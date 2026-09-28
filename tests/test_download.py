import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_dataset_matches_pinned_checksum():
    result = subprocess.run(
        [sys.executable, str(ROOT / "data" / "download.py"), "--check"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "rows 29531" in result.stdout
