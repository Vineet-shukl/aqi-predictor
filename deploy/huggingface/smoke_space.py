"""Poll the live Space health URL. Warn, and exit 0, if it is still building."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

from deploy.huggingface.space_bundle import SPACE_HEALTH_URL

DEFAULT_TIMEOUT_SECONDS = 600
POLL_SECONDS = 20


def health_is_ok(status_code: int, body: str) -> bool:
    if status_code != 200:
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def fetch_health(url: str, timeout: float = 20) -> tuple[int, str]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return exc.code, detail
    except urllib.error.URLError as exc:
        return 0, str(exc.reason)


def poll_health(
    url: str = SPACE_HEALTH_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Return True when the Space answers /health with status ok."""
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        status, body = fetch_health(url)
        if health_is_ok(status, body):
            print(f"Space is healthy (HTTP {status}): {body.strip()}")
            return True
        snippet = " ".join(body.split())[:180]
        print(f"Attempt {attempt}: HTTP {status} {snippet}".rstrip())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(POLL_SECONDS, remaining))


def main() -> None:
    ready = poll_health()
    if ready:
        return
    message = (
        f"GET {SPACE_HEALTH_URL} did not return status ok within "
        f"{DEFAULT_TIMEOUT_SECONDS // 60} minutes. The Space may still be building."
    )
    print(f"::warning title=Space still building::{message}")
    print(message, file=sys.stderr)


if __name__ == "__main__":
    main()
