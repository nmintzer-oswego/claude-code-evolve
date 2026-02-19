"""
R5-compliant measurement protocol for CodeEvolve evaluators.

Protocol (validated in R5):
- 5 warmup calls (untimed, warms CPU cache)
- GC disabled during measurement window
- 10 timed runs using time.perf_counter()
- Median as estimator (wins 3/6 regimes, robust to outliers)

With this protocol, MDE is 1.03x even in the noisiest (1ms) regime.

Source: Promoted from Research/R6_EndToEnd/measure.py (no logic changes).
"""

import gc
import statistics
import time

WARMUP_RUNS = 5
MEASUREMENT_RUNS = 10


def measure_runtime(func, args, warmup=WARMUP_RUNS, runs=MEASUREMENT_RUNS):
    """
    Measure function runtime using R5 protocol.

    Args:
        func: The function to time.
        args: Tuple of arguments to pass to func.
        warmup: Number of untimed warmup calls.
        runs: Number of timed measurement runs.

    Returns:
        Median time in seconds.
    """
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Measurement
    gc.disable()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    gc.enable()

    return statistics.median(times)
