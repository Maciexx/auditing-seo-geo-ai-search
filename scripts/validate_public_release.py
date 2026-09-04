"""Check the publishable tree and optional reachable history without logging client terms.

Supply private names/domains locally with --private-term. Do not store them in this repository.
This complements CI secret/fixture checks; it does not identify unknown clients automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

PRIVATE_PARTS = {"clients", "audit-output", "owned-input", ".staging", "output", "tmp"}
GENERATED = {
    "audit.json",
    "evidence.jsonl",
    "delivery.json",
    "ai-prompts.json",
    "client-report-data.json",
    "report-draft.json",
    "diagnostics.json",
}


def fabricated_diagnostic_example(path: Path, value: object) -> bool:
    """A narrow fixture declaration, never an exemption for a copied run directory."""
    if (
        path.parts[:1] != ("fixtures",)
        or path.name != "diagnostics.example.json"
        or not isinstance(value, dict)
        or value.get("synthetic") is not True
        or not isinstance(value.get("domain"), str)
        or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.example", value["domain"])
    ):
        return False

    def safe(item: object) -> bool:
        if isinstance(item, dict):
            return all(safe(key) and safe(child) for key, child in item.items())
        if isinstance(item, list):
            return all(safe(child) for child in item)
        if not isinstance(item, str):
            return True
        # IDNA recognizes these dot variants. Tokenize Unicode labels (including
        # combining marks) before checking nested bindings and quoted domains.
        sample = unicodedata.normalize("NFC", item.translate(str.maketrans("。．｡", "...")))
        label = r"[^\s/.:@?#<>\"'()\[\]{},;!\\]+"
        for domain in re.findall(rf"{label}(?:\.{label})+", sample):
            if re.fullmatch(r"[0-9.]+", domain):
                continue
            try:
                hostname = domain.encode("idna").decode("ascii").lower()
            except UnicodeError:
                return False
            if not hostname.endswith(".example"):
                return False
        for url in re.findall(r"https?://[^\s<>\"']+", sample, flags=re.IGNORECASE):
            try:
                parsed = urlsplit(url)
                _ = parsed.port
                hostname = (parsed.hostname or "").encode("idna").decode("ascii").lower()
            except (ValueError, UnicodeError):
                return False
            if (
                parsed.username is not None
                or parsed.password is not None
                or "\\" in url
                or not hostname.endswith(".example")
            ):
                return False
        return True

    return safe(value)


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True).stdout


def normalized(value: str) -> str:
    return re.sub(r"[\W_]", "", value.casefold())


def path_label(name: str, terms: tuple[str, ...]) -> str:
    if any(term in normalized(name) for term in terms):
        return f"path-sha256:{hashlib.sha256(name.encode()).hexdigest()}"
    return name


def inspect(name: str, payload: bytes, terms: tuple[str, ...]) -> list[str]:
    path = Path(name)
    issues = []
    private_path = any(term in normalized(name) for term in terms)
    label = path_label(name, terms)
    if private_path:
        issues.append(f"configured private identifier in path: {label}")
    if (
        PRIVATE_PARTS.intersection(path.parts)
        or path.name in GENERATED
        or any(
            parts[0] == "diagnostics" and re.fullmatch(r"run-[1-9][0-9]*", parts[2])
            for parts in zip(path.parts, path.parts[1:], path.parts[2:], strict=False)
        )
        or path.suffix.lower() in {".pdf", ".csv", ".xlsx"}
        or path.parts[:2] == ("docs", "superpowers")
    ):
        issues.append(f"private/generated artifact: {label}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        if path.name == "diagnostics.example.json":
            issues.append(f"undeclared diagnostic output: {label}")
        return issues
    folded = normalized(text)
    if any(term in folded for term in terms):
        issues.append(f"configured private identifier: {label}")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        diagnostic_shape = isinstance(value, dict) and (
            {"binding", "module_states"} <= value.keys()
            or {"binding", "algorithm_sha256", "files"} <= value.keys()
        )
        if (diagnostic_shape or path.name == "diagnostics.example.json") and not (
            fabricated_diagnostic_example(path, value)
        ):
            issues.append(f"undeclared diagnostic output: {label}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--private-term", action="append", default=[])
    parser.add_argument("--history", action="store_true", help="scan every ancestor of HEAD")
    args = parser.parse_args()
    terms = tuple(normalized(term) for term in args.private_term)
    if any(not term for term in terms):
        parser.error("private terms cannot be empty")
    issues = []
    for raw_path in git(args.root, "ls-files", "-z").split(b"\0"):
        if not raw_path:
            continue
        name = raw_path.decode()
        path = args.root / name
        if path.is_symlink():
            issues.append(f"tracked symlink requires review: {path_label(name, terms)}")
        elif path.is_file():
            issues.extend(inspect(name, path.read_bytes(), terms))
    if args.history:
        for commit in git(args.root, "rev-list", "HEAD").decode().splitlines():
            message = git(args.root, "show", "-s", "--format=%B", commit)
            if any(term in normalized(message.decode()) for term in terms):
                issues.append(f"history {commit}: configured private identifier in commit message")
            for raw_name in git(args.root, "ls-tree", "-r", "--name-only", "-z", commit).split(
                b"\0"
            ):
                if not raw_name:
                    continue
                name = raw_name.decode()
                payload = git(args.root, "show", f"{commit}:{name}")
                issues.extend(
                    f"history {commit}: {issue}" for issue in inspect(name, payload, terms)
                )
    for issue in sorted(set(issues)):
        print(issue)
    if issues:
        return 1
    print("Release boundary passed (configured terms and artifact paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
