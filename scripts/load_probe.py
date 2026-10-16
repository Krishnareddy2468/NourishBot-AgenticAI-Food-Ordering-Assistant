#!/usr/bin/env python3
"""Bounded, read-only latency and error-rate probe for deployment validation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


def hit(url: str) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            response.read(4096)
            return response.status, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        return exc.code, (time.perf_counter() - started) * 1000
    except Exception:
        return 0, (time.perf_counter() - started) * 1000


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Use a read-only endpoint such as /api/health")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-p95-ms", type=float, default=1000)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--report")
    args = parser.parse_args()
    if not 1 <= args.requests <= 10_000 or not 1 <= args.concurrency <= 100:
        parser.error("requests must be 1..10000 and concurrency must be 1..100")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda _value: hit(args.url), range(args.requests)))
    statuses = [status for status, _elapsed in results]
    latencies = [elapsed for _status, elapsed in results]
    failures = sum(1 for status in statuses if status < 200 or status >= 400)
    error_rate = failures / len(results)
    p95 = percentile(latencies, .95)
    passed = p95 <= args.max_p95_ms and error_rate <= args.max_error_rate
    report = {
        "passed": passed,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "requests": len(results),
        "concurrency": args.concurrency,
        "status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses))},
        "latency_ms": {"mean": round(statistics.mean(latencies), 2), "p95": round(p95, 2), "max": round(max(latencies), 2)},
        "error_rate": round(error_rate, 4),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
