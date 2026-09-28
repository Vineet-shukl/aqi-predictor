import os
import subprocess
import sys
from pathlib import Path

import pytest

from deploy.huggingface.push_space import MISSING_TOKEN_MESSAGE, redact, require_token
from deploy.huggingface.smoke_space import health_is_ok
from deploy.huggingface.space_bundle import (
    iter_shipped_files,
    repo_root,
    stage_space,
    stale_delete_patterns,
    validate_space_readme,
)
from deploy.huggingface.verify_upload import verify

ROOT = repo_root()

EXPECTED_SPACE_FILES = [
    "Dockerfile",
    "README.md",
    "app/__init__.py",
    "app/main.py",
    "models/metadata.json",
    "models/pm25_model.joblib",
    "requirements.txt",
    "src/__init__.py",
    "src/aqi.py",
    "src/estimators.py",
    "src/features.py",
    "src/inference.py",
    "src/train.py",
]


def test_space_readme_front_matter_is_valid():
    text = (ROOT / "deploy" / "huggingface" / "README.md").read_text(encoding="utf-8")
    meta = validate_space_readme(text)
    assert meta["sdk"] == "docker"
    assert meta["app_port"] == "7860"
    assert meta["pinned"] == "false"
    assert meta["license"] == "unknown"
    assert meta["title"] == "India PM2.5 Predictor"
    assert meta["emoji"] == "🌫️"
    assert meta["colorFrom"] == "blue"
    assert meta["colorTo"] == "green"


def test_upload_set_is_only_the_serving_files():
    relative = [path for _, path in iter_shipped_files(ROOT)]
    assert relative == EXPECTED_SPACE_FILES
    assert "data/city_day.csv" not in relative
    assert not any(path.startswith("reports/") for path in relative)


def test_stage_space_copies_readme_and_model(tmp_path: Path):
    shipped = stage_space(ROOT, tmp_path)
    assert shipped == EXPECTED_SPACE_FILES
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    source = (ROOT / "deploy" / "huggingface" / "README.md").read_text(encoding="utf-8")
    assert readme == source
    assert (tmp_path / "models" / "pm25_model.joblib").is_file()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "reports").exists()


def test_delete_patterns_never_include_readme():
    remote = [
        "README.md",
        "Dockerfile",
        "app/main.py",
        "data/city_day.csv",
        "old.py",
        ".gitattributes",
        "src/train.py",
    ]
    shipped = set(EXPECTED_SPACE_FILES)
    patterns = stale_delete_patterns(remote, shipped)
    assert "README.md" not in patterns
    assert patterns == [".gitattributes", "data/city_day.csv", "old.py"]


def test_delete_patterns_reject_globs():
    with pytest.raises(RuntimeError, match="unsafe delete pattern"):
        stale_delete_patterns(["*"], set(EXPECTED_SPACE_FILES))


def test_missing_hf_token_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        require_token()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert MISSING_TOKEN_MESSAGE in captured.err


def test_empty_hf_token_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "   ")
    with pytest.raises(SystemExit) as exc:
        require_token()
    assert exc.value.code == 1


def test_token_value_is_not_printed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    secret = "hf_test_token_should_not_appear"
    monkeypatch.setenv("HF_TOKEN", secret)
    token = require_token()
    captured = capsys.readouterr()
    assert token == secret
    assert secret not in captured.out
    assert secret not in captured.err
    assert redact(f"failed with {secret}", secret) == "failed with [REDACTED]"


def test_readme_documents_the_live_space():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/spaces/Pandaisop/aqi-predictor" in readme
    assert "https://pandaisop-aqi-predictor.hf.space/docs" in readme
    assert "auto-deployed from `main`" in readme


def test_health_classifier_accepts_only_ok_json():
    assert health_is_ok(200, '{"status":"ok","model_name":"random_forest"}')
    assert not health_is_ok(503, '{"status":"ok"}')
    assert not health_is_ok(200, "<html>Building</html>")
    assert not health_is_ok(200, '{"status":"starting"}')


def test_staged_upload_serves_health_and_predict():
    shipped = verify(ROOT)
    assert shipped == EXPECTED_SPACE_FILES


def test_push_script_fails_without_token():
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    completed = subprocess.run(
        [sys.executable, "-m", "deploy.huggingface.push_space"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "HF_TOKEN is missing or empty" in completed.stderr
    assert completed.stdout == ""
    assert "hf_" not in completed.stderr
