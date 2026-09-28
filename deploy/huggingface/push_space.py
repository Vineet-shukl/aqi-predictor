"""Upload the serving bundle to the Hugging Face Docker Space.

The token is read from the ``HF_TOKEN`` environment variable and is never printed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from deploy.huggingface.space_bundle import (
    README_DEST,
    REPO_ID,
    SPACE_PAGE,
    iter_shipped_files,
    repo_root,
    stage_space,
    stale_delete_patterns,
)

MISSING_TOKEN_MESSAGE = (
    "HF_TOKEN is missing or empty. Add a fine-grained Hugging Face write token "
    "as the GitHub Actions repository secret HF_TOKEN. "
    "This deploy does not continue without it."
)


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token is None or token.strip() == "":
        print(MISSING_TOKEN_MESSAGE, file=sys.stderr)
        raise SystemExit(1)
    return token.strip()


def redact(text: str, token: str) -> str:
    if token and token in text:
        return text.replace(token, "[REDACTED]")
    return text


def push_space(token: str, root: Path | None = None) -> list[str]:
    """Create the Space if needed and upload the serving files.

    Returns the Space-relative paths that were uploaded.
    """
    from huggingface_hub import HfApi

    root = repo_root() if root is None else root
    shipped_pairs = iter_shipped_files(root)
    shipped = [relative for _, relative in shipped_pairs]
    api = HfApi()
    try:
        api.create_repo(
            REPO_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token,
        )
        try:
            remote = api.list_repo_files(REPO_ID, repo_type="space", token=token)
        except Exception as exc:
            print(
                "Could not list existing Space files "
                f"({redact(str(exc), token)}). Uploading without deletions.",
                file=sys.stderr,
            )
            remote = []
        patterns = stale_delete_patterns(list(remote), set(shipped))
        if README_DEST in patterns:
            print("Refusing to delete README.md from the Space.", file=sys.stderr)
            raise SystemExit(1)
        with tempfile.TemporaryDirectory(prefix="hf-space-") as tmp:
            staging = Path(tmp)
            staged = stage_space(root, staging)
            if staged != shipped:
                raise RuntimeError("Staged upload set does not match the shipped file list")
            api.upload_folder(
                folder_path=staging,
                repo_id=REPO_ID,
                repo_type="space",
                token=token,
                commit_message="Deploy FastAPI Space from main",
                delete_patterns=patterns,
            )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Hugging Face deploy failed: {redact(str(exc), token)}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Uploaded {len(shipped)} files to {SPACE_PAGE}")
    for relative in shipped:
        print(f"  {relative}")
    return shipped


def main() -> None:
    token = require_token()
    push_space(token)


if __name__ == "__main__":
    main()
