"""Check that the Space upload set can serve /health and /predict by itself."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from deploy.huggingface.space_bundle import repo_root, stage_space


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _curl_json(method: str, url: str, body_path: Path | None = None) -> tuple[int, dict]:
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--max-time",
        "30",
        "-w",
        "\n%{http_code}",
        "-X",
        method,
        url,
    ]
    if body_path is not None:
        command[1:1] = ["-H", "Content-Type: application/json", "--data-binary", f"@{body_path}"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"curl failed for {url}")
    output = completed.stdout
    if "\n" not in output:
        raise RuntimeError(f"curl returned no status code for {url}")
    payload, _, status_text = output.rpartition("\n")
    status = int(status_text)
    try:
        parsed = json.loads(payload) if payload else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Response from {url} is not JSON: {payload[:200]}") from exc
    return status, parsed


def _wait_for_health(url: str, proc: subprocess.Popen[str], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            status, body = _curl_json("GET", url)
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(0.4)
            continue
        if status == 200 and body.get("status") == "ok":
            return body
        last_error = f"HTTP {status}: {body}"
        time.sleep(0.4)
    raise RuntimeError(f"Server did not become healthy ({last_error})")


def _stop_process(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=10)
    return f"{stdout or ''}{stderr or ''}"


def serve_and_check(staging: Path, root: Path, port: int) -> None:
    """Start uvicorn with ``staging`` as the only application directory."""
    request_path = root / "reports" / "api_example_request.json"
    example = json.loads((root / "reports" / "api_example.json").read_text(encoding="utf-8"))
    metadata = json.loads((staging / "models" / "metadata.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(staging)
    env["PYTHONUNBUFFERED"] = "1"
    env["PORT"] = str(port)
    env.pop("HF_TOKEN", None)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        command,
        cwd=staging,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        health = _wait_for_health(f"{base}/health", proc, timeout=60)
        if health["model_name"] != metadata["best_model"]:
            raise RuntimeError(f"Unexpected model_name: {health}")
        if health["n_cities"] != len(metadata["cities"]):
            raise RuntimeError(f"Unexpected n_cities: {health}")
        if health["target"] != metadata["target"]:
            raise RuntimeError(f"Unexpected target: {health}")
        status, predicted = _curl_json("POST", f"{base}/predict", request_path)
        if status != 200:
            raise RuntimeError(f"/predict returned HTTP {status}: {predicted}")
        if predicted != example["response"]:
            raise RuntimeError(f"/predict response did not match the saved example: {predicted}")
    except Exception as exc:
        logs = _stop_process(proc)
        raise RuntimeError(f"{exc}\nserver log:\n{logs}") from None
    else:
        _stop_process(proc)
    print(f"Upload set served /health and /predict on {base}")


def docker_build_and_check(staging: Path, root: Path) -> None:
    """Build the image from the upload set and curl the container."""
    tag = "aqi-predictor-space"
    name = "aqi-predictor-space-verify"
    host_port = free_port()
    subprocess.run(["docker", "build", "-t", tag, str(staging)], check=True)
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", f"{host_port}:7860", tag],
        check=True,
    )
    try:
        request_path = root / "reports" / "api_example_request.json"
        example = json.loads((root / "reports" / "api_example.json").read_text(encoding="utf-8"))
        deadline = time.monotonic() + 90
        health: dict | None = None
        last_error = "container did not respond"
        while time.monotonic() < deadline:
            try:
                status, body = _curl_json("GET", f"http://127.0.0.1:{host_port}/health")
            except RuntimeError as exc:
                last_error = str(exc)
                time.sleep(1)
                continue
            if status == 200 and body.get("status") == "ok":
                health = body
                break
            last_error = f"HTTP {status}: {body}"
            time.sleep(1)
        if health is None:
            logs = subprocess.run(
                ["docker", "logs", name],
                check=False,
                capture_output=True,
                text=True,
            )
            raise RuntimeError(
                f"Container was not healthy ({last_error}).\n{logs.stdout}\n{logs.stderr}"
            )
        status, predicted = _curl_json(
            "POST",
            f"http://127.0.0.1:{host_port}/predict",
            request_path,
        )
        if status != 200 or predicted != example["response"]:
            raise RuntimeError(f"Container /predict failed: HTTP {status} {predicted}")
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    print(f"Docker image {tag} built from the upload set and served /health and /predict")


def verify(root: Path | None = None, *, docker: bool = False, port: int | None = None) -> list[str]:
    root = repo_root() if root is None else root
    with tempfile.TemporaryDirectory(prefix="hf-space-verify-") as tmp:
        staging = Path(tmp)
        shipped = stage_space(root, staging)
        print("Staged Space files:")
        for relative in shipped:
            print(f"  {relative}")
        serve_and_check(staging, root, port if port is not None else free_port())
        if docker:
            docker_build_and_check(staging, root)
    return shipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Also docker build the staged directory and curl the container",
    )
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    try:
        verify(docker=args.docker, port=args.port)
    except Exception as exc:
        print(f"Space upload verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
