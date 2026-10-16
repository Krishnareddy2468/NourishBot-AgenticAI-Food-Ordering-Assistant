#!/usr/bin/env python3
"""Non-mutating deployment smoke test for the public frontend and backend."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def request(url: str) -> tuple[int, dict[str, str], bytes, float]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return response.status, dict(response.headers), response.read(200_000), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read(200_000), time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    api = args.api.rstrip("/")
    frontend = args.frontend.rstrip("/")
    checks = []

    for name, url, statuses in (
        ("backend_liveness", f"{api}/api/health", {200}),
        ("backend_readiness", f"{api}/api/ready", {200}),
        ("unauthenticated_staff_denied", f"{api}/api/v1/analytics/operations", {401}),
        ("frontend_home", f"{frontend}/", {200}),
        ("frontend_login", f"{frontend}/login", {200}),
        ("frontend_customer_dashboard", f"{frontend}/account", {200}),
    ):
        try:
            status, headers, body, elapsed = request(url)
            passed = status in statuses and bool(body)
            if name.startswith("backend_"):
                passed = passed and headers.get("X-Content-Type-Options", "").lower() == "nosniff"
            checks.append({"name": name, "passed": passed, "status": status, "duration_ms": round(elapsed * 1000, 2)})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "error": exc.__class__.__name__})

    result = {
        "passed": all(item["passed"] for item in checks),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
