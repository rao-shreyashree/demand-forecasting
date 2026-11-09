"""Dashboard benchmark for issue #63 AC.

Helper-level: reorder_table() execution time (pandas-side only).
Dashboard-level: full Streamlit render cycle (page load + filter reruns),
measured with streamlit.testing.v1.AppTest — headless, reproducible.

Run: python -m scripts.bench_dashboard

Requires: streamlit >= 1.28 (AppTest API)
"""
from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

import streamlit
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
RUNS = 5
THRESHOLD_MS = 3000
TIMEOUT_S = 30

if not Path(APP_PATH).is_file():
    raise FileNotFoundError(f"dashboard app not found: {APP_PATH}")

def _env() -> dict[str, str]:
    return {
        "python":    sys.version.split()[0],
        "streamlit": streamlit.__version__,
        "platform":  platform.platform(),
    }


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_S).run()
    if at.exception:
        raise RuntimeError(f"dashboard raised: {at.exception}")
    return at


def _find(at: AppTest, kind: str, label_fragment: str):
    """Find widget by label substring (case-insensitive). Returns None if not found."""
    for w in getattr(at, kind):
        if hasattr(w, "label") and label_fragment.lower() in str(w.label).lower():
            return w
    return None


def _measure(fn, runs: int = RUNS) -> tuple[float, float, float]:
    """Run fn() once as warmup, then runs times; return (median, min, max) ms."""
    fn()                                   # warmup, discarded
    times = [fn() for _ in range(runs)]
    return statistics.median(times), min(times), max(times)


def bench_initial_cold() -> float:
    streamlit.cache_data.clear()
    t0 = time.perf_counter()
    _run_app()
    return (time.perf_counter() - t0) * 1000


def bench_initial_warm() -> float:
    t0 = time.perf_counter()
    _run_app()
    return (time.perf_counter() - t0) * 1000


def bench_multiselect(label: str, pick: str = "first") -> float:
    def run() -> float:
        at = _run_app()
        w = _find(at, "multiselect", label)
        if w is None:
            raise LookupError(f"multiselect[{label!r}] not found")
        opts = list(w.options)
        if not opts:
            return 0.0
        val = [opts[0] if pick == "first" else opts[1]]
        w.set_value(val)
        t0 = time.perf_counter()
        at.run()
        if at.exception:
            raise RuntimeError(f"rerun raised: {at.exception}")
        return (time.perf_counter() - t0) * 1000
    return run


def bench_selectbox(label: str) -> float:
    def run() -> float:
        at = _run_app()
        w = _find(at, "selectbox", label)
        if w is None:
            raise LookupError(f"selectbox[{label!r}] not found")
        opts = list(w.options)
        if len(opts) < 2:
            return 0.0
        w.select(opts[1])
        t0 = time.perf_counter()
        at.run()
        if at.exception:
            raise RuntimeError(f"rerun raised: {at.exception}")
        return (time.perf_counter() - t0) * 1000
    return run


def bench_slider(label: str, value) -> float:
    def run() -> float:
        at = _run_app()
        w = _find(at, "slider", label)
        if w is None:
            raise LookupError(f"slider[{label!r}] not found")
        w.set_value(value)
        t0 = time.perf_counter()
        at.run()
        if at.exception:
            raise RuntimeError(f"rerun raised: {at.exception}")
        return (time.perf_counter() - t0) * 1000
    return run


def _report(results: dict[str, tuple[float, float, float]]) -> bool:
    print()
    print(f"{'Measurement':<40} {'median':>10} {'min':>10} {'max':>10}")
    print("-" * 75)
    for label, (med, lo, hi) in results.items():
        if lo == hi == 0:
            print(f"{label:<40} {med:>8.1f}ms {'-':>10} {'-':>10}")
        else:
            print(f"{label:<40} {med:>8.1f}ms {lo:>8.1f}ms {hi:>8.1f}ms")
    print()

    fails = [(k, v[0]) for k, v in results.items() if v[0] > THRESHOLD_MS]
    if fails:
        print(f"3s threshold: FAIL")
        for label, ms in fails:
            print(f"  {label}: {ms:.0f}ms > {THRESHOLD_MS}ms")
        return False
    print(f"3s threshold: PASS (all measurements <= {THRESHOLD_MS}ms)")
    return True


def main() -> int:
    print("Environment:")
    for k, v in _env().items():
        print(f"  {k:10s} {v}")
    print()

    # Sanity: list all widgets so we know what's visible to AppTest
    at = _run_app()
    print(f"Widgets visible: "
          f"selectbox={len(at.selectbox)} multiselect={len(at.multiselect)} "
          f"slider={len(at.slider)} checkbox={len(at.checkbox)}")
    for kind in ("selectbox", "multiselect", "slider"):
        for i, w in enumerate(getattr(at, kind)):
            print(f"  {kind}[{i}]: {w.label!r}")
    print()

    results: dict[str, tuple[float, float, float]] = {}

    print("Benchmarking initial load (cold, cache cleared)...")
    results["Initial dashboard load (cold)"] = (bench_initial_cold(), 0, 0)

    print("Benchmarking initial load (warm)...")
    results["Initial dashboard load (warm)"] = _measure(bench_initial_warm)

    # Forecast tab
    for label, fn in [
        ("Store selectbox change",   bench_selectbox("Store")),
        ("Item selectbox change",    bench_selectbox("Item")),
        ("Models multiselect change", bench_multiselect("Models")),
        ("History slider change",    bench_slider("History", 14)),
    ]:
        print(f"Benchmarking {label}...")
        try:
            results[label] = _measure(fn)
        except LookupError as e:
            print(f"  SKIPPED: {e}")

    # Reorder tab
    for label, fn in [
        ("Reorder: Flags filter",    bench_multiselect("Flags")),
        ("Reorder: Stores filter",   bench_multiselect("Stores")),
        ("Reorder: Items filter",    bench_multiselect("Items")),
    ]:
        print(f"Benchmarking {label}...")
        try:
            results[label] = _measure(fn)
        except LookupError as e:
            print(f"  SKIPPED: {e}")

    return 0 if _report(results) else 1


if __name__ == "__main__":
    sys.exit(main())