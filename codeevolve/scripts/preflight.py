"""
CodeEvolve Preflight Check (SRD § 3.6).

Validates all inputs before evolution begins:
1. Parses the function file — identifies the target function
2. Validates the test suite — all tests must pass on the original
3. Checks benchmark viability — warns if function runs in <1ms
4. Measures baseline performance — establishes timing reference
5. Presents summary and asks for developer confirmation

Exit codes:
  0 = confirmed, proceed with evolution
  1 = abort (parse error / test failure)
  2 = user cancelled

On success (exit 0), prints a JSON blob to stdout with baseline data
for build_harness.py to consume.

Usage:
    python preflight.py <function_file> <test_file>
        [--function <name>] [--objective time|space|balanced] [--iterations N]
"""

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add scripts dir to path for measure_performance
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from measure_performance import measure_runtime

WARMUP_RUNS = 5
MEASUREMENT_RUNS = 10

# Synthetic benchmark sizes used when no large test inputs are found
SYNTHETIC_SIZES = [1000, 5000, 10000]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Parse function file
# ─────────────────────────────────────────────────────────────────────────────

def find_functions(filepath: str) -> list[str]:
    """Return names of all top-level functions in a Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        print(f"  ERROR: Cannot parse {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            and isinstance(node.col_offset, int) and node.col_offset == 0]


def get_function_linecount(filepath: str, func_name: str) -> int:
    """Return approximate line count of a named function."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return (node.end_lineno or node.lineno) - node.lineno + 1
    return 0


def count_params(filepath: str, func_name: str) -> int:
    """Return parameter count of a named function."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return len(node.args.args)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Validate test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(test_file: str) -> tuple[bool, int, str]:
    """
    Run pytest on the test file.
    Returns (all_passed, test_count, output).
    """
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short', '-q'],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    # Count tests from pytest output
    test_count = 0
    for line in output.splitlines():
        # pytest summary line: "5 passed" or "3 passed, 2 failed"
        if 'passed' in line or 'failed' in line or 'error' in line:
            import re
            nums = re.findall(r'(\d+)\s+(passed|failed|error)', line)
            for n, _ in nums:
                test_count += int(n)
            break

    passed = result.returncode == 0
    return passed, test_count, output


# ─────────────────────────────────────────────────────────────────────────────
# Steps 3-4: Benchmark viability + baseline measurement
# ─────────────────────────────────────────────────────────────────────────────

def load_function(filepath: str, func_name: str):
    """Dynamically load a function from a file."""
    spec = importlib.util.spec_from_file_location("target_module", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def _try_extract_call(node, func_name: str):
    """Extract literal args from a Call node if it matches func_name. Returns tuple or None."""
    if not isinstance(node, ast.Call):
        return None
    call_name = ''
    if isinstance(node.func, ast.Name):
        call_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        call_name = node.func.attr
    if call_name != func_name:
        return None
    try:
        return tuple(ast.literal_eval(a) for a in node.args)
    except (ValueError, TypeError):
        return None


def extract_test_inputs(test_file: str, func_name: str) -> list:
    """
    Extract input args from test file using AST. Handles multiple patterns:
      - assert func(args) == expected
      - assert func(args)[key] == expected   (subscript)
      - result = func(args)
    Returns list of arg tuples as literal values.
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=test_file)
    except SyntaxError:
        return []

    inputs = []

    def _add(args):
        if args is not None and args not in inputs:
            inputs.append(args)

    for node in ast.walk(tree):
        # Pattern 1: assert func(args) == expected
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                left = test.left
                # Direct call: assert func(args) == x
                _add(_try_extract_call(left, func_name))
                # Subscript: assert func(args)[key] == x
                if isinstance(left, ast.Subscript):
                    _add(_try_extract_call(left.value, func_name))

        # Pattern 2: result = func(args)
        elif isinstance(node, ast.Assign):
            _add(_try_extract_call(node.value, func_name))

    return inputs


def measure_baseline(func, inputs: list, func_name: str) -> dict:
    """
    Measure baseline timing across available inputs.
    Returns dict with timing info and regime assessment.
    """
    if not inputs:
        # No inputs extracted — use a simple single-element call if possible
        return {"measurements": [], "regime": "unknown", "max_ms": 0.0}

    measurements = []
    for args in inputs:
        try:
            t = measure_runtime(func, args)
            measurements.append({"args_size": _estimate_size(args), "median_s": t, "median_ms": t * 1000})
        except Exception:
            pass

    if not measurements:
        return {"measurements": [], "regime": "unknown", "max_ms": 0.0}

    max_ms = max(m["median_ms"] for m in measurements)
    if max_ms >= 100:
        regime = "EXCELLENT (>=100ms)"
    elif max_ms >= 10:
        regime = "GOOD (10-100ms)"
    elif max_ms >= 1:
        regime = "OK (1-10ms)"
    else:
        regime = "NOISY (<1ms) - consider scaling inputs"

    return {"measurements": measurements, "regime": regime, "max_ms": max_ms}


def _estimate_size(args) -> int:
    """Estimate the 'size' of an input tuple (largest collection length)."""
    sizes = []
    for a in args:
        if hasattr(a, '__len__'):
            sizes.append(len(a))
        elif isinstance(a, (int, float)):
            sizes.append(abs(int(a)))
    return max(sizes) if sizes else 1


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Summary display and confirmation
# ─────────────────────────────────────────────────────────────────────────────

def display_summary(func_name: str, param_count: int, line_count: int,
                    test_count: int, objective: str, iterations: int,
                    baseline: dict) -> None:
    """Print the preflight summary box."""
    regime = baseline.get("regime", "unknown")
    measurements = baseline.get("measurements", [])

    # Estimate timing per iteration
    max_ms = baseline.get("max_ms", 0.0)
    if max_ms >= 100:
        iter_time = "~15s"
    elif max_ms >= 10:
        iter_time = "~8s"
    else:
        iter_time = "~5.5s"

    total_min_s = iterations * 5.5
    total_max_s = iterations * 15.0
    total_min_m = total_min_s / 60
    total_max_m = total_max_s / 60
    cost_min = iterations * 0.0014
    cost_max = iterations * 0.0044

    print()
    print("+----------------------------------------------+")
    print("|  CodeEvolve Preflight Check                  |")
    print("+----------------------------------------------+")
    print(f"|  Function:    {func_name} ({param_count} params, {line_count} lines)".ljust(47) + "|")
    print(f"|  Test suite:  {test_count} tests, all passing".ljust(47) + "|")
    print(f"|  Objective:   {objective}".ljust(47) + "|")
    print("|                                              |")
    if measurements:
        print("|  Baseline performance:                       |")
        for m in measurements[:4]:  # show at most 4
            size = m["args_size"]
            ms = m["median_ms"]
            line = f"|    n={size}: {ms:.1f}ms/call"
            print(line.ljust(47) + "|")
    print(f"|  Measurement regime: {regime}".ljust(47) + "|")
    print("|                                              |")
    print("|  Plan:                                       |")
    print(f"|    Iterations:  {iterations}".ljust(47) + "|")
    print(f"|    Est. time:   ~{total_min_m:.0f}-{total_max_m:.0f} min".ljust(47) + "|")
    print(f"|    Est. cost:   ~${cost_min:.2f}-${cost_max:.2f}".ljust(47) + "|")
    print("|                                              |")
    print("|  Proceed? [Y/n/configure]                    |")
    print("+----------------------------------------------+")


def get_confirmation() -> str:
    """Read user confirmation from stdin. Returns 'y', 'n', or 'configure'."""
    try:
        answer = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 'n'
    if answer in ('', 'y', 'yes'):
        return 'y'
    elif answer in ('n', 'no'):
        return 'n'
    elif answer.startswith('c'):
        return 'configure'
    return 'y'  # default yes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CodeEvolve preflight check — validates inputs before evolution"
    )
    parser.add_argument('function_file', help="Path to Python file containing the function")
    parser.add_argument('test_file', help="Path to pytest-compatible test file")
    parser.add_argument('--function', help="Target function name (required if multiple functions)")
    parser.add_argument('--objective', choices=['time', 'space', 'balanced'], default='time',
                        help="Optimization objective. Only 'time' is supported in v1.0.")
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--yes', action='store_true', help="Skip confirmation prompt (non-interactive)")
    args = parser.parse_args()

    function_file = os.path.abspath(args.function_file)
    test_file = os.path.abspath(args.test_file)

    print(f"\nCodeEvolve Preflight Check")
    print(f"Function file: {function_file}")
    print(f"Test file:     {test_file}")
    print()

    # ── Objective guard ───────────────────────────────────────────────────────
    if args.objective in ('space', 'balanced'):
        print(f"ERROR: --objective {args.objective} is not supported in v1.0.", file=sys.stderr)
        print("  Space and balanced objectives require a validated tracemalloc measurement", file=sys.stderr)
        print("  protocol (noise profile, MDE, GC interaction) that has not yet been", file=sys.stderr)
        print("  researched. This is planned for v2.0.", file=sys.stderr)
        print("  Use --objective time (default) for now.", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: Parse function file ──────────────────────────────────────────
    print("Step 1/5: Parsing function file...")
    funcs = find_functions(function_file)
    if not funcs:
        print(f"  ERROR: No top-level functions found in {function_file}", file=sys.stderr)
        sys.exit(1)

    if args.function:
        func_name = args.function
        if func_name not in funcs:
            print(f"  ERROR: Function '{func_name}' not found. Available: {funcs}", file=sys.stderr)
            sys.exit(1)
    elif len(funcs) == 1:
        func_name = funcs[0]
    else:
        print(f"  Multiple functions found: {funcs}", file=sys.stderr)
        print(f"  Use --function <name> to specify which to optimize.", file=sys.stderr)
        sys.exit(1)

    param_count = count_params(function_file, func_name)
    line_count = get_function_linecount(function_file, func_name)
    print(f"  OK: function '{func_name}' ({param_count} params, {line_count} lines)")

    # ── Step 2: Validate test suite ──────────────────────────────────────────
    print("Step 2/5: Running tests against original function...")
    passed, test_count, test_output = run_tests(test_file)
    if not passed:
        print(f"  ERROR: Tests failed on the original function. Fix tests before optimizing.", file=sys.stderr)
        print(test_output[:1000], file=sys.stderr)
        sys.exit(1)
    print(f"  OK: {test_count} tests, all passing")

    # ── Step 3: Benchmark viability ──────────────────────────────────────────
    print("Step 3/5: Extracting test inputs for baseline measurement...")
    try:
        func = load_function(function_file, func_name)
    except Exception as e:
        print(f"  ERROR: Could not load function: {e}", file=sys.stderr)
        sys.exit(1)

    inputs = extract_test_inputs(test_file, func_name)
    if not inputs:
        print(f"  WARNING: Could not extract test inputs via AST.")
        print(f"  Will use synthetic benchmark inputs (list of integers).")
        # Generate synthetic inputs for a list-processing function
        import random
        random.seed(42)
        inputs = []
        for n in SYNTHETIC_SIZES:
            lst = list(range(n))
            random.shuffle(lst)
            inputs.append((lst,))
    else:
        print(f"  Found {len(inputs)} test input(s) via AST extraction")

    # ── Step 4: Baseline measurement ─────────────────────────────────────────
    print("Step 4/5: Measuring baseline performance...")
    baseline = measure_baseline(func, inputs, func_name)

    regime = baseline.get("regime", "unknown")
    max_ms = baseline.get("max_ms", 0.0)
    print(f"  Max measured: {max_ms:.2f}ms/call")
    print(f"  Regime: {regime}")

    if max_ms < 1.0:
        print()
        print("  WARNING: Function runs in <1ms on all test inputs.")
        print("  Measurement noise will be high (CV ~10%). For reliable evolution,")
        print("  consider providing larger benchmark inputs (aim for >1ms/call).")
        print("  You can continue anyway, but speedup detection may be noisy.")

    # ── Step 5: Summary & confirmation ───────────────────────────────────────
    print("Step 5/5: Summary")
    display_summary(func_name, param_count, line_count, test_count,
                    args.objective, args.iterations, baseline)

    if args.yes:
        answer = 'y'
        print("  (--yes flag: auto-confirming)")
    else:
        answer = get_confirmation()

    if answer == 'n':
        print("  Aborted.")
        sys.exit(2)
    elif answer == 'configure':
        print("  Run /project:configure to adjust settings, then retry.")
        sys.exit(2)

    # ── Output baseline JSON for build_harness.py ─────────────────────────────
    output = {
        "function_name": func_name,
        "function_file": function_file,
        "test_file": test_file,
        "param_count": param_count,
        "line_count": line_count,
        "test_count": test_count,
        "objective": args.objective,
        "iterations": args.iterations,
        "baseline": baseline,
        "extracted_inputs": [list(inp) for inp in inputs[:10]],  # pass up to 10 inputs
    }
    print("\n__PREFLIGHT_JSON__")
    print(json.dumps(output))
    print("__PREFLIGHT_JSON_END__")


if __name__ == '__main__':
    main()
