"""Fail when a repository-local Markdown link points to a missing path."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REMOTE_SCHEMES = {"http", "https", "mailto", "tel", "data"}


def markdown_files() -> list[Path]:
    """Return tracked-style Markdown sources while skipping local environments."""

    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.md")
        if not EXCLUDED_PARTS.intersection(path.relative_to(PROJECT_ROOT).parts)
    )


def normalized_target(raw_target: str) -> str:
    """Remove optional titles and GitHub angle brackets from a link target."""

    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def missing_local_links(path: Path) -> list[tuple[str, Path]]:
    """Return missing local targets referenced by one Markdown file."""

    failures: list[tuple[str, Path]] = []
    text = path.read_text(encoding="utf-8")
    for match in LINK_PATTERN.finditer(text):
        raw_target = normalized_target(match.group(1))
        parsed = urlparse(raw_target)
        if parsed.scheme.lower() in REMOTE_SCHEMES or raw_target.startswith("#"):
            continue

        relative_target = unquote(parsed.path)
        if not relative_target:
            continue
        if relative_target.startswith("/"):
            resolved = PROJECT_ROOT / relative_target.lstrip("/")
        else:
            resolved = path.parent / relative_target
        resolved = resolved.resolve()

        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            failures.append((raw_target, resolved))
            continue
        if not resolved.exists():
            failures.append((raw_target, resolved))
    return failures


def main() -> int:
    """Validate every repository-local link and report actionable failures."""

    files = markdown_files()
    failures: list[tuple[Path, str, Path]] = []
    for path in files:
        failures.extend(
            (path, raw_target, resolved)
            for raw_target, resolved in missing_local_links(path)
        )

    if failures:
        for source, raw_target, resolved in failures:
            source_name = source.relative_to(PROJECT_ROOT)
            print(f"{source_name}: missing {raw_target!r} -> {resolved}")
        return 1

    print(f"Validated local Markdown links in {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
