"""Files copied into the Hugging Face Docker Space.

The API loads ``models/pm25_model.joblib`` and ``models/metadata.json`` at
import time. It does not read the CSV or the reports directory.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

REPO_ID = "Pandaisop/aqi-predictor"
SPACE_PAGE = "https://huggingface.co/spaces/Pandaisop/aqi-predictor"
SPACE_HEALTH_URL = "https://pandaisop-aqi-predictor.hf.space/health"
SPACE_DOCS_URL = "https://pandaisop-aqi-predictor.hf.space/docs"

# Directories and files at the repository root that the running container needs.
SHIPPED_PATHS = (
    "Dockerfile",
    "requirements.txt",
    "app",
    "src",
    "models",
)
README_SOURCE = Path("deploy/huggingface/README.md")
README_DEST = "README.md"

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

# Hugging Face Space thumbnail colors.
# https://huggingface.co/docs/hub/spaces-config-reference
SPACE_COLORS = frozenset(
    {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
)

# Card license identifiers.
# https://huggingface.co/docs/hub/repositories-licenses
SPACE_LICENSES = frozenset(
    {
        "apache-2.0",
        "mit",
        "openrail",
        "bigscience-openrail-m",
        "creativeml-openrail-m",
        "bigscience-bloom-rail-1.0",
        "bigcode-openrail-m",
        "afl-3.0",
        "artistic-2.0",
        "bsl-1.0",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-3-clause-clear",
        "c-uda",
        "cc",
        "cc0-1.0",
        "cc-by-2.0",
        "cc-by-2.5",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
        "cc-by-nc-2.0",
        "cc-by-nc-3.0",
        "cc-by-nc-4.0",
        "cc-by-nd-4.0",
        "cc-by-nc-nd-3.0",
        "cc-by-nc-nd-4.0",
        "cc-by-nc-sa-2.0",
        "cc-by-nc-sa-3.0",
        "cc-by-nc-sa-4.0",
        "cdla-sharing-1.0",
        "cdla-permissive-1.0",
        "cdla-permissive-2.0",
        "wtfpl",
        "ecl-2.0",
        "epl-1.0",
        "epl-2.0",
        "etalab-2.0",
        "eupl-1.1",
        "eupl-1.2",
        "agpl-3.0",
        "gfdl",
        "gpl",
        "gpl-2.0",
        "gpl-3.0",
        "lgpl",
        "lgpl-2.1",
        "lgpl-3.0",
        "isc",
        "h-research",
        "intel-research",
        "lppl-1.3c",
        "ms-pl",
        "apple-ascl",
        "apple-amlr",
        "mpl-2.0",
        "odc-by",
        "odbl",
        "openmdw-1.0",
        "openmdw-1.1",
        "openrail++",
        "osl-3.0",
        "postgresql",
        "ofl-1.1",
        "ncsa",
        "unlicense",
        "zlib",
        "pddl",
        "lgpl-lr",
        "deepfloyd-if-license",
        "fair-noncommercial-research-license",
        "llama2",
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "llama4",
        "grok2-community",
        "gemma",
        "unknown",
        "other",
    }
)

REQUIRED_FRONT_MATTER = {
    "title": "India PM2.5 Predictor",
    "emoji": "🌫️",
    "colorFrom": "blue",
    "colorTo": "green",
    "sdk": "docker",
    "app_port": "7860",
    "pinned": "false",
    "license": "unknown",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_front_matter(text: str) -> dict[str, str]:
    """Parse the flat YAML block at the top of the Space README."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Space README is missing YAML front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("Space README front matter is not closed") from exc
    parsed: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Invalid front matter line: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Invalid front matter line: {line}")
        parsed[key] = value
    return parsed


def validate_space_readme(text: str) -> dict[str, str]:
    """Raise ValueError when the Space card front matter is not valid."""
    meta = parse_front_matter(text)
    missing = [key for key in REQUIRED_FRONT_MATTER if key not in meta]
    if missing:
        raise ValueError(f"Space README front matter is missing: {', '.join(missing)}")
    if not meta["title"]:
        raise ValueError("Space README title is empty")
    if not meta["emoji"]:
        raise ValueError("Space README emoji is empty")
    if meta["colorFrom"] not in SPACE_COLORS:
        raise ValueError(f"Invalid colorFrom: {meta['colorFrom']}")
    if meta["colorTo"] not in SPACE_COLORS:
        raise ValueError(f"Invalid colorTo: {meta['colorTo']}")
    if meta["sdk"] != "docker":
        raise ValueError(f"Space sdk must be docker, got {meta['sdk']}")
    if meta["app_port"] != "7860":
        raise ValueError(f"Space app_port must be 7860, got {meta['app_port']}")
    if meta["pinned"] != "false":
        raise ValueError(f"Space pinned must be false, got {meta['pinned']}")
    if meta["license"] not in SPACE_LICENSES:
        raise ValueError(f"Invalid Space license: {meta['license']}")
    for key, expected in REQUIRED_FRONT_MATTER.items():
        if meta[key] != expected:
            raise ValueError(f"Space README {key} must be {expected!r}, got {meta[key]!r}")
    return meta


def iter_shipped_files(root: Path | None = None) -> list[tuple[Path, str]]:
    """Return ``(source, path_in_space)`` pairs for the upload set."""
    root = repo_root() if root is None else root
    files: list[tuple[Path, str]] = []
    for rel in SHIPPED_PATHS:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Required Space path is missing: {rel}")
        if path.is_file():
            files.append((path, rel))
            continue
        found = False
        for file in sorted(path.rglob("*")):
            if not file.is_file():
                continue
            relative_parts = file.relative_to(root).parts
            if any(part in SKIP_DIR_NAMES for part in relative_parts):
                continue
            if file.suffix in SKIP_SUFFIXES:
                continue
            files.append((file, file.relative_to(root).as_posix()))
            found = True
        if not found:
            raise FileNotFoundError(f"Required Space directory is empty: {rel}")
    readme = root / README_SOURCE
    if not readme.is_file():
        raise FileNotFoundError(f"Required Space README is missing: {README_SOURCE}")
    validate_space_readme(readme.read_text(encoding="utf-8"))
    files.append((readme, README_DEST))
    files.sort(key=lambda item: item[1])
    return files


def stage_space(root: Path, dest: Path) -> list[str]:
    """Copy the upload set into ``dest`` and return Space-relative paths."""
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    shipped: list[str] = []
    for source, relative in iter_shipped_files(root):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        shipped.append(relative)
    return shipped


def stale_delete_patterns(remote_files: list[str], shipped: set[str]) -> list[str]:
    """Remote paths to remove. ``README.md`` is never included.

    Patterns are exact file paths. Glob characters are rejected so a remote
    name cannot expand to ``README.md``.
    """
    patterns: list[str] = []
    for path in sorted(remote_files):
        if path == README_DEST or path in shipped:
            continue
        if path == "" or any(char in path for char in "*?[]"):
            raise RuntimeError(f"Refusing unsafe delete pattern: {path!r}")
        if fnmatch.fnmatchcase(README_DEST, path):
            raise RuntimeError("Refusing to delete README.md")
        patterns.append(path)
    return patterns
