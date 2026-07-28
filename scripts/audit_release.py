#!/usr/bin/env python3
"""Fail if common private-release artefacts or machine-specific paths are present."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).parents[1]
SKIP = {".git", "__pycache__", "tools"}
SELF = Path(__file__).resolve()
NAME_PATTERNS = ("CL" + "AUDE", "AG" + "ENTS", "PIPELINE_DOSSIER", "METHODS_DRAFT", "corrections")
TEXT_PATTERNS = {
    "absolute local path": re.compile("/" + "(Users|home)/[^/]+/"),
    "cluster path": re.compile("/" + "nfs/"),
    "OpenAI secret": re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
}


def public_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        yield path


def main():
    findings = []
    for path in public_files():
        if path.resolve() == SELF:
            continue
        if any(token.lower() in path.name.lower() for token in NAME_PATTERNS):
            findings.append(f"private-looking filename: {path.relative_to(ROOT)}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"unexpected binary file: {path.relative_to(ROOT)}")
            continue
        for description, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{description}: {path.relative_to(ROOT)}")
    if findings:
        print("Release audit failed:", *findings, sep="\n- ")
        sys.exit(1)
    print("Release audit passed: no scanned private artefacts or absolute paths found.")


if __name__ == "__main__":
    main()
