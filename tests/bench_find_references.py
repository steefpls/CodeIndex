"""Benchmark find_references latency on representative symbols.

Run with: PYTHONPATH=. python tests/bench_find_references.py

Measures p50/p95 latencies and file counts for find_references across
mainapp (with Unity coverage enabled). This determines whether the
current Python-based file scanning approach is acceptable or whether
a ripgrep backend is needed.

Decision criteria (from workstream F):
  - p95 < 2s for common symbols with max_results <= 50 -> KEEP current impl
  - p95 >= 2s -> consider rg backend
"""

import statistics
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.tools.references import find_references

# Representative symbols to benchmark
_SYMBOLS = [
    ("IRobotDriver", "mainapp"),
    ("MonoBehaviour", "mainapp"),
    ("Transform", "mainapp"),
    ("CalibrationManager", "mainapp"),
    ("OnEnable", "mainapp"),
    ("MeshRenderer", "mainapp"),
    ("HandleCalibration", "mainapp"),
    ("NetworkHandler", "mainapp"),
    ("IDisposable", "mainapp"),
    ("SerializeField", "mainapp"),
]

_ITERATIONS = 3


def run_benchmark():
    print("=== find_references benchmark ===\n")
    print(f"Symbols: {len(_SYMBOLS)}, iterations per symbol: {_ITERATIONS}\n")

    all_latencies = []
    results_table = []

    for symbol, repo in _SYMBOLS:
        latencies = []
        result_count = 0
        for _ in range(_ITERATIONS):
            t0 = time.perf_counter()
            result = find_references(symbol, repo, max_results=50)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            # Count result lines (rough proxy for matches)
            result_count = len(result.split("\n")) - 1

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else max(latencies)
        all_latencies.extend(latencies)
        results_table.append((symbol, repo, avg, p95, result_count))

    # Print results
    print(f"{'Symbol':<25} {'Repo':<10} {'Avg (s)':<10} {'p95 (s)':<10} {'Results':<10}")
    print("-" * 65)
    for symbol, repo, avg, p95, count in results_table:
        print(f"{symbol:<25} {repo:<10} {avg:<10.3f} {p95:<10.3f} {count:<10}")

    # Overall stats
    overall_p50 = statistics.median(all_latencies)
    overall_p95 = sorted(all_latencies)[int(len(all_latencies) * 0.95)]
    overall_max = max(all_latencies)

    print(f"\n--- Overall ({len(all_latencies)} measurements) ---")
    print(f"  p50: {overall_p50:.3f}s")
    print(f"  p95: {overall_p95:.3f}s")
    print(f"  max: {overall_max:.3f}s")

    # Decision
    print(f"\n--- Decision ---")
    if overall_p95 < 2.0:
        print(f"  KEEP current Python implementation (p95={overall_p95:.3f}s < 2.0s threshold)")
        print(f"  No rg backend needed at current scale.")
    else:
        print(f"  CONSIDER rg backend (p95={overall_p95:.3f}s >= 2.0s threshold)")
        print(f"  Performance may degrade further as Unity coverage expands.")


if __name__ == "__main__":
    run_benchmark()
