# Contributing

Keep each branch focused on one change. Small pull requests are easier to review and verify.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Run the checks before opening a pull request:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/validate_skill.py .
```

## Fixtures and private data

Use synthetic fixtures with `.example` domains only. Do not add client names, domains, addresses, facts, screenshots, evidence, reports, credentials, or authenticated data.

## Evidence and scoring changes

New or changed rules must include a source, verification date, evidence level, confidence, expiry, and tests. Scoring changes must retain traceability and preserve explicit unavailable states, coverage, and confidence.
