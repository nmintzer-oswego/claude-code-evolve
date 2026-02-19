"""
CodeEvolve Harness Builder (SRD § 3.5).

Takes the developer-provided function file and test file and packages them
into a directory structure that OpenEvolve can consume:

  .codeevolve/runs/<run-id>/
  ├── initial.py          # Original function with EVOLVE-BLOCK markers
  ├── evaluator.py        # Correctness tests + R5 performance measurement
  ├── config.yaml         # OpenEvolve configuration
  ├── claude_code_llm.py  # LLM adapter (copied from scripts/)
  └── results/            # Created by OpenEvolve during the run

Usage:
    python build_harness.py <function_file> <test_file>
        --run-id <id>
        --objective time|space|balanced
        [--iterations N]
        [--function <name>]
        [--preflight-json <json_string>]
"""

import argparse
import ast
import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
CODEEVOLVE_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = CODEEVOLVE_DIR.parent
CONFIG_DEFAULT = CODEEVOLVE_DIR / 'config' / 'default_config.yaml'


# ─────────────────────────────────────────────────────────────────────────────
# Extract function source
# ─────────────────────────────────────────────────────────────────────────────

def extract_function_source(filepath: str, func_name: str) -> tuple[str, str]:
    """
    Extract the source of a named function from a file.
    Returns (imports_section, function_source).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)
    lines = source.splitlines()

    # Collect top-level imports
    import_lines = []
    func_node = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_lines.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node

    if func_node is None:
        raise ValueError(f"Function '{func_name}' not found in {filepath}")

    # Extract function lines (ast gives 1-based line numbers)
    start = func_node.lineno - 1
    end = func_node.end_lineno
    func_lines = lines[start:end]
    func_source = '\n'.join(func_lines)

    imports_section = '\n'.join(l for l in import_lines if l)
    return imports_section, func_source


# ─────────────────────────────────────────────────────────────────────────────
# Write initial.py with EVOLVE-BLOCK markers
# ─────────────────────────────────────────────────────────────────────────────

def write_initial_py(run_dir: Path, imports: str, func_source: str, func_name: str) -> None:
    """Write initial.py with EVOLVE-BLOCK markers around the target function."""
    content_parts = [f'"""{func_name} - Initial program for CodeEvolve evolution."""\n']
    if imports:
        content_parts.append(imports)
        content_parts.append('')
    content_parts.append('')
    content_parts.append('# EVOLVE-BLOCK-START')
    content_parts.append(func_source)
    content_parts.append('# EVOLVE-BLOCK-END')
    content_parts.append('')

    with open(run_dir / 'initial.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_parts))


# ─────────────────────────────────────────────────────────────────────────────
# Extract test cases from test file (AST-only approach)
# ─────────────────────────────────────────────────────────────────────────────

def extract_test_cases(test_file: str, func_name: str) -> list[tuple]:
    """
    Parse test file with AST looking for: assert func(args) == expected
    Returns list of ((args...), expected) tuples using ast.literal_eval.
    Only cases where all args and expected are literals are included.
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=test_file)
    except SyntaxError:
        return []

    cases = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        # Pattern: assert func(args) == expected
        if not isinstance(test, ast.Compare):
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue

        left = test.left
        comparators = test.comparators
        if not comparators:
            continue

        # Handle: assert func(args) == expected  AND  assert expected == func(args)
        call_node = None
        expected_node = None
        if isinstance(left, ast.Call):
            call_node = left
            expected_node = comparators[0]
        elif isinstance(comparators[0], ast.Call):
            call_node = comparators[0]
            expected_node = left

        if call_node is None:
            continue

        # Check function name matches
        call_name = ''
        if isinstance(call_node.func, ast.Name):
            call_name = call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            call_name = call_node.func.attr

        if call_name != func_name:
            continue

        # Try to extract literal args
        try:
            args = tuple(ast.literal_eval(a) for a in call_node.args)
            # Handle keyword args that pass lists/dicts as positional
            expected = ast.literal_eval(expected_node)
            cases.append((args, expected))
        except (ValueError, TypeError):
            pass

    return cases


def infer_compare_output(cases: list[tuple]) -> str:
    """
    Infer a compare_output function based on the types of expected values.
    Returns Python source for the compare_output function.

    Uses exact equality by default. Only uses sort-comparison when the expected
    values are sets (Python set literals), because lists are ordered and sorting
    would mask wrong-order results (e.g. productExceptSelf, prefix sums, etc.).
    """
    if not cases:
        return "def compare_output(result, expected):\n    return result == expected\n"

    # Only sort-compare if the expected values are actual Python sets (unordered)
    for _, expected in cases:
        if isinstance(expected, set):
            return (
                "def compare_output(result, expected):\n"
                "    \"\"\"Sort both before comparing — expected values are unordered sets.\"\"\"\n"
                "    try:\n"
                "        return sorted(result) == sorted(expected)\n"
                "    except TypeError:\n"
                "        return result == expected\n"
            )
    # Default: exact equality (preserves order for lists, tuples, scalars)
    return "def compare_output(result, expected):\n    return result == expected\n"


def format_test_cases(cases: list[tuple]) -> str:
    """Format test cases as Python source for the evaluator."""
    if not cases:
        return (
            "# TODO: Add test cases in this format:\n"
            "# TEST_CASES = [\n"
            "#     ((arg1, arg2, ...), expected_output),\n"
            "#     ...\n"
            "# ]\n"
            "TEST_CASES = []\n"
        )
    lines = ["TEST_CASES = ["]
    for args, expected in cases:
        lines.append(f"    ({repr(args)}, {repr(expected)}),")
    lines.append("]")
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Derive benchmark sizes from baseline data
# ─────────────────────────────────────────────────────────────────────────────

def derive_benchmark_sizes(baseline: dict, extracted_inputs: list) -> list[int]:
    """
    Derive benchmark input sizes that put the function in >1ms regime,
    while keeping total evaluator wall time safely under 90s.

    Budget: evaluator timeout is 120s. Each size runs 15 calls (5 warmup +
    10 measured) for both baseline and evolved — so 30 calls per size.
    Target: each call <1s so 3 sizes × 30 calls = 90s worst case.
    For an O(n²) brute baseline: n=500 ≈ 1ms, n=1500 ≈ 9ms, n=3000 ≈ 36ms.
    Safe cap: n=1500 keeps a single call under 10ms (well within budget).
    """
    measurements = baseline.get("measurements", []) if baseline else []
    max_ms = baseline.get("max_ms", 0.0) if baseline else 0.0

    # Hard cap: each benchmark call must complete in <1s on the brute baseline.
    # For O(n²) on modern hardware, n≈3000 is roughly 50ms — safe upper bound.
    MAX_SAFE_SIZE = 3000

    if measurements:
        sizes = [m["args_size"] for m in measurements if m["args_size"] > 0]
        if sizes:
            largest = max(sizes)
            if max_ms >= 1.0:
                # Already in measurable regime — use existing sizes, capped
                return sorted(set([
                    min(MAX_SAFE_SIZE, max(100, largest)),
                    min(MAX_SAFE_SIZE, max(500, largest * 2)),
                    min(MAX_SAFE_SIZE, max(1000, largest * 4)),
                ]))
            else:
                # Sub-1ms — scale up to hit >1ms, capped at safe max
                scale = min(MAX_SAFE_SIZE, max(10, int(1.0 / max(max_ms, 0.001) * largest)))
                return sorted(set([scale, min(MAX_SAFE_SIZE, scale * 2), min(MAX_SAFE_SIZE, scale * 4)]))

    # Fallback when no baseline data: conservative sizes that work for O(n²)
    # within the 120s evaluator timeout.
    return [500, 1000, 1500]


def generate_benchmark_input_source(extracted_inputs: list, sizes: list[int]) -> str:
    """
    Generate source for the generate_benchmark_input(n) function.
    Uses the first extracted input as a template if it's a list.
    Falls back to generating a shuffled list of integers.
    """
    # Check if the first input is a single list — common case
    first_input = extracted_inputs[0] if extracted_inputs else None
    if first_input and len(first_input) == 1 and isinstance(first_input[0], list):
        # Single-list input: generate a shuffled list of size n
        return textwrap.dedent("""\
            def generate_benchmark_input(n):
                \"\"\"Generate deterministic benchmark input of size n.\"\"\"
                import random
                random.seed(n)
                lst = list(range(n))
                random.shuffle(lst)
                return (lst,)
            """)
    elif first_input and len(first_input) == 2 and isinstance(first_input[0], list):
        # Two-arg input where first is a list: pass second arg from largest test case
        second_arg = first_input[1]
        return textwrap.dedent(f"""\
            def generate_benchmark_input(n):
                \"\"\"Generate deterministic benchmark input of size n.\"\"\"
                import random
                random.seed(n)
                lst = list(range(n))
                random.shuffle(lst)
                return (lst, {repr(second_arg)})
            """)
    else:
        # Generic fallback: generate a list of integers
        return textwrap.dedent("""\
            def generate_benchmark_input(n):
                \"\"\"Generate deterministic benchmark input of size n (list of integers).\"\"\"
                import random
                random.seed(n)
                lst = list(range(n))
                random.shuffle(lst)
                return (lst,)
            """)


# ─────────────────────────────────────────────────────────────────────────────
# Write evaluator.py
# ─────────────────────────────────────────────────────────────────────────────

EVALUATOR_TEMPLATE = '''\
"""
Generated evaluator for CodeEvolve.
Function: {func_name}
Objective: {objective}

OpenEvolve calls evaluate(program_path) → Dict[str, float].
Two-stage gate: correctness must pass before performance is measured.

Auto-generated by build_harness.py. You can edit TEST_CASES manually
if auto-extraction missed any cases.
"""

import importlib.util
import os
import sys
from typing import List, Tuple, Dict, Optional, Any

# Add codeevolve/scripts to path for measure_performance
# Resolved at runtime so this evaluator is portable across machines.
# evaluator.py lives at .codeevolve/runs/<run-id>/evaluator.py
# Three levels up (.codeevolve/runs/<run-id> → .codeevolve/runs → .codeevolve → project root)
# then down into codeevolve/scripts/.
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'codeevolve', 'scripts'))

from measure_performance import measure_runtime

# === Problem-specific config ===
FUNCTION_NAME = {func_name_repr}

{test_cases_source}

# Benchmark sizes (chosen to be in >1ms regime for reliable measurement)
BENCHMARK_SIZES = {benchmark_sizes}


{compare_output_source}

{generate_benchmark_input_source}

{inline_brute_source}

# === Generic evaluator logic ===

def _load_module(filepath):
    """Dynamically load a Python module, sanitizing Unicode issues."""
    with open(filepath, \'r\', encoding=\'utf-8\') as f:
        code = f.read()
    # Sanitize Unicode minus sign (R4 finding: LLM sometimes emits U+2212)
    sanitized = code.replace(\'\\u2212\', \'-\')
    if sanitized != code:
        with open(filepath, \'w\', encoding=\'utf-8\') as f:
            f.write(sanitized)
    spec = importlib.util.spec_from_file_location("evolved_module", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(program_path):
    """Evaluate an evolved implementation of {func_name}."""
    try:
        mod = _load_module(program_path)
    except Exception:
        return {{"combined_score": 0.0, "correctness": 0.0, "avg_speedup": 0.0}}

    if not hasattr(mod, FUNCTION_NAME):
        return {{"combined_score": 0.0, "correctness": 0.0, "avg_speedup": 0.0}}

    func = getattr(mod, FUNCTION_NAME)

    # === STAGE 1: CORRECTNESS ===
    if not TEST_CASES:
        # No test cases extracted — skip correctness gate, score on performance only
        correctness = 1.0
    else:
        passed = 0
        for args, expected in TEST_CASES:
            try:
                result = func(*args)
                if compare_output(result, expected):
                    passed += 1
            except Exception:
                return {{
                    "combined_score": 0.0,
                    "correctness": float(passed) / len(TEST_CASES),
                    "avg_speedup": 0.0,
                }}
        correctness = float(passed) / len(TEST_CASES)
        if passed < len(TEST_CASES):
            return {{"combined_score": 0.0, "correctness": correctness, "avg_speedup": 0.0}}

    # === STAGE 2: PERFORMANCE (R5 protocol) ===
    speedups = []
    for n in BENCHMARK_SIZES:
        args = generate_benchmark_input(n)
        baseline_time = measure_runtime(inline_brute, args)
        evolved_time = measure_runtime(func, args)
        if evolved_time > 0:
            speedups.append(baseline_time / evolved_time)

    avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0

    return {{
        "combined_score": avg_speedup,
        "correctness": correctness,
        "avg_speedup": avg_speedup,
    }}
'''


def write_evaluator_py(run_dir: Path, func_name: str, objective: str,
                       imports: str, func_source: str,
                       cases: list[tuple], baseline: dict,
                       extracted_inputs: list) -> None:
    """Generate and write evaluator.py to the run directory."""
    benchmark_sizes = derive_benchmark_sizes(baseline, extracted_inputs)
    compare_output_src = infer_compare_output(cases)
    test_cases_src = format_test_cases(cases)
    gen_input_src = generate_benchmark_input_source(extracted_inputs, benchmark_sizes)

    # inline_brute = the original function, renamed to inline_brute
    inline_brute_src = func_source.replace(
        f'def {func_name}(', 'def inline_brute(', 1
    )
    inline_brute_src = f"# Inline baseline for fair measurement (original implementation)\n{inline_brute_src}"

    content = EVALUATOR_TEMPLATE.format(
        func_name=func_name,
        func_name_repr=repr(func_name),
        objective=objective,
        test_cases_source=test_cases_src,
        benchmark_sizes=repr(benchmark_sizes),
        compare_output_source=compare_output_src,
        generate_benchmark_input_source=gen_input_src,
        inline_brute_source=inline_brute_src,
    )

    with open(run_dir / 'evaluator.py', 'w', encoding='utf-8') as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────────────
# Write config.yaml
# ─────────────────────────────────────────────────────────────────────────────

def write_config_yaml(run_dir: Path, iterations: int) -> None:
    """Copy default config, overriding max_iterations."""
    with open(CONFIG_DEFAULT, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace max_iterations line
    content = re.sub(
        r'^max_iterations:\s*\d+',
        f'max_iterations: {iterations}',
        content,
        flags=re.MULTILINE,
    )

    with open(run_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CodeEvolve harness builder — packages function + tests into OpenEvolve run directory"
    )
    parser.add_argument('function_file', help="Path to Python file containing the function")
    parser.add_argument('test_file', help="Path to pytest-compatible test file")
    parser.add_argument('--run-id', required=True, help="Unique run identifier")
    parser.add_argument('--objective', choices=['time', 'space', 'balanced'], default='time',
                        help="Optimization objective. Only 'time' is supported in v1.0.")
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--function', help="Target function name (required if multiple functions)")
    parser.add_argument('--preflight-json', help="JSON string output from preflight.py")
    args = parser.parse_args()

    function_file = os.path.abspath(args.function_file)
    test_file = os.path.abspath(args.test_file)

    # Parse preflight data if provided
    preflight_data = {}
    if args.preflight_json:
        try:
            preflight_data = json.loads(args.preflight_json)
        except json.JSONDecodeError:
            pass

    if args.objective in ('space', 'balanced'):
        print(f"ERROR: --objective {args.objective} is not supported in v1.0.", file=sys.stderr)
        print("  Space and balanced objectives require a validated tracemalloc measurement", file=sys.stderr)
        print("  protocol that has not yet been researched. Planned for v2.0.", file=sys.stderr)
        print("  Use --objective time (default).", file=sys.stderr)
        sys.exit(1)

    func_name = args.function or preflight_data.get('function_name')
    if not func_name:
        # Auto-detect if single function
        with open(function_file, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source)
        funcs = [n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.col_offset == 0]
        if len(funcs) == 1:
            func_name = funcs[0]
        else:
            print(f"ERROR: Multiple functions found ({funcs}). Use --function.", file=sys.stderr)
            sys.exit(1)

    baseline = preflight_data.get('baseline', {})
    extracted_inputs = preflight_data.get('extracted_inputs', [])

    # Create run directory
    run_dir = PROJECT_ROOT / '.codeevolve' / 'runs' / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'results').mkdir(exist_ok=True)

    print(f"Building harness in: {run_dir}")

    # Extract function
    print(f"  Extracting function '{func_name}'...")
    imports, func_source = extract_function_source(function_file, func_name)

    # Write initial.py
    print("  Writing initial.py...")
    write_initial_py(run_dir, imports, func_source, func_name)

    # Extract test cases
    print("  Extracting test cases via AST...")
    cases = extract_test_cases(test_file, func_name)
    if cases:
        print(f"  Extracted {len(cases)} test case(s)")
    else:
        print("  WARNING: No test cases extracted via AST.")
        print("  The generated evaluator.py has a TODO marker — fill in TEST_CASES manually.")

    # Write evaluator.py
    print("  Writing evaluator.py...")
    write_evaluator_py(run_dir, func_name, args.objective, imports, func_source,
                       cases, baseline, extracted_inputs)

    # Write config.yaml
    print("  Writing config.yaml...")
    write_config_yaml(run_dir, args.iterations)

    # Copy claude_code_llm.py into run dir
    print("  Copying claude_code_llm.py...")
    shutil.copy(SCRIPTS_DIR / 'claude_code_llm.py', run_dir / 'claude_code_llm.py')

    print(f"\nHarness built: {run_dir}")
    print(f"  initial.py    — {func_name} with EVOLVE-BLOCK markers")
    print(f"  evaluator.py  — {len(cases)} test cases, {derive_benchmark_sizes(baseline, extracted_inputs)} benchmark sizes")
    print(f"  config.yaml   — {args.iterations} iterations, objective={args.objective}")

    # Output run dir path for the calling slash command
    print(f"\n__RUN_DIR__\n{run_dir}\n__RUN_DIR_END__")


if __name__ == '__main__':
    main()
