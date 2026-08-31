#!/usr/bin/env python3
"""Validate the repository's Agent Skill package contract for CI."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ALLOWED_FRONTMATTER = {"name", "description", "license", "allowed-tools", "metadata"}


def validate_skill(root: Path) -> list[str]:
    errors: list[str] = []
    skill_path = root / "SKILL.md"
    if not skill_path.is_file():
        return ["SKILL.md not found"]
    content = skill_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---(?:\n|$)", content, re.DOTALL)
    if match is None:
        return ["SKILL.md has invalid YAML frontmatter"]
    frontmatter = yaml.safe_load(match.group(1))
    if not isinstance(frontmatter, dict):
        return ["SKILL.md frontmatter must be a mapping"]
    unexpected = set(frontmatter) - ALLOWED_FRONTMATTER
    if unexpected:
        errors.append(f"unexpected frontmatter keys: {sorted(unexpected)}")
    name = frontmatter.get("name")
    if not isinstance(name, str) or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
        errors.append("skill name must use lowercase hyphen-case")
    elif len(name) > 64:
        errors.append("skill name exceeds 64 characters")
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("skill description is required")
    elif len(description) > 1024 or "<" in description or ">" in description:
        errors.append("skill description violates length or character constraints")
    agent_path = root / "agents" / "openai.yaml"
    if not agent_path.is_file():
        errors.append("agents/openai.yaml not found")
    else:
        agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
        interface = agent.get("interface") if isinstance(agent, dict) else None
        if not isinstance(interface, dict):
            errors.append("agents/openai.yaml requires an interface mapping")
        else:
            for field in ("display_name", "short_description", "default_prompt"):
                if not isinstance(interface.get(field), str) or not interface[field].strip():
                    errors.append(f"agents/openai.yaml requires interface.{field}")
    return errors


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path.cwd()
    errors = validate_skill(root.resolve())
    if errors:
        print("Skill validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Skill is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
