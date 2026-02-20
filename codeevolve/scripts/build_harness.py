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

    Detects the input shape from the first extracted input and generates a
    matching scalable benchmark generator. Falls back to a shuffled list of
    integers only when no extracted inputs are available.
    """
    first_input = extracted_inputs[0] if extracted_inputs else None

    # Pattern: (int, list-of-lists, int) — graph/adjacency problems
    # e.g. shortestPath(n, edges, src)
    if (first_input and len(first_input) == 3
            and isinstance(first_input[0], int)
            and isinstance(first_input[1], list)
            and first_input[1] and isinstance(first_input[1][0], list)
            and isinstance(first_input[2], int)):
        return textwrap.dedent("""\
            def generate_benchmark_input(n):
                \"\"\"Generate deterministic (n, edges, src) graph input of scale n.\"\"\"
                import random
                random.seed(n)
                edges = []
                for i in range(n - 1):
                    edges.append([i, i + 1, random.randint(1, 100)])
                for i in range(0, n - 10, 5):
                    edges.append([i, i + 10, random.randint(1, 50)])
                    edges.append([i, i + 7, random.randint(1, 50)])
                return (n, edges, 0)
            """)

    # Pattern: (list,) — single-list input
    if first_input and len(first_input) == 1 and isinstance(first_input[0], list):
        return textwrap.dedent("""\
            def generate_benchmark_input(n):
                \"\"\"Generate deterministic benchmark input of size n.\"\"\"
                import random
                random.seed(n)
                lst = list(range(n))
                random.shuffle(lst)
                return (lst,)
            """)

    # Pattern: (list, scalar) — list + second arg
    if first_input and len(first_input) == 2 and isinstance(first_input[0], list):
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

    # Generic fallback: generate a list of integers.
    # NOTE: This may not match the function's actual signature for multi-arg functions.
    # If speedup measurement is nonsensical, manually edit generate_benchmark_input in evaluator.py.
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
Two-stage gate: correctness (pytest) must pass before performance is measured.

Auto-generated by build_harness.py.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
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

# Path to the user's pytest test file (absolute, set at harness-build time)
TEST_FILE = {test_file_repr}

# Name of the module the test file imports from (stem of the original function file)
MODULE_NAME = {module_name_repr}

# Benchmark sizes (chosen to be in >1ms regime for reliable measurement)
BENCHMARK_SIZES = {benchmark_sizes}


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
    # === STAGE 1: CORRECTNESS — run pytest against evolved program ===
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Make the evolved program importable under the original module name
            shutil.copy(program_path, os.path.join(tmpdir, MODULE_NAME + \'.py\'))
            result = subprocess.run(
                [sys.executable, \'-m\', \'pytest\', TEST_FILE,
                 \'--tb=no\', \'-q\', \'--no-header\'],
                cwd=tmpdir,
                capture_output=True,
                timeout=60,
            )
            correctness = 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return {{"combined_score": 0.0, "correctness": 0.0, "avg_speedup": 0.0}}

    if correctness < 1.0:
        return {{"combined_score": 0.0, "correctness": correctness, "avg_speedup": 0.0}}

    # === STAGE 2: PERFORMANCE (R5 protocol) ===
    try:
        mod = _load_module(program_path)
    except Exception:
        return {{"combined_score": 0.0, "correctness": correctness, "avg_speedup": 0.0}}

    if not hasattr(mod, FUNCTION_NAME):
        return {{"combined_score": 0.0, "correctness": correctness, "avg_speedup": 0.0}}

    func = getattr(mod, FUNCTION_NAME)

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
                       test_file: str, module_name: str,
                       baseline: dict, extracted_inputs: list) -> None:
    """Generate and write evaluator.py to the run directory."""
    benchmark_sizes = derive_benchmark_sizes(baseline, extracted_inputs)
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
        test_file_repr=repr(test_file),
        module_name_repr=repr(module_name),
        benchmark_sizes=repr(benchmark_sizes),
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

    # The module name the test file imports from is the stem of the function file.
    # e.g. solution.py → 'solution', so the evaluator copies evolved.py as solution.py
    module_name = Path(function_file).stem

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

    # Detect benchmark input shape
    benchmark_sizes = derive_benchmark_sizes(baseline, extracted_inputs)
    gen_src = generate_benchmark_input_source(extracted_inputs, benchmark_sizes)
    input_shape = "graph (n, edges, src)" if "edges" in gen_src else \
                  "list" if "(lst,)" in gen_src else "unknown"

    # Write evaluator.py
    print("  Writing evaluator.py...")
    write_evaluator_py(run_dir, func_name, args.objective, imports, func_source,
                       test_file, module_name, baseline, extracted_inputs)

    # Write config.yaml
    print("  Writing config.yaml...")
    write_config_yaml(run_dir, args.iterations)

    # Copy claude_code_llm.py into run dir
    print("  Copying claude_code_llm.py...")
    shutil.copy(SCRIPTS_DIR / 'claude_code_llm.py', run_dir / 'claude_code_llm.py')

    print(f"\nHarness built: {run_dir}")
    print(f"  initial.py    — {func_name} with EVOLVE-BLOCK markers")
    print(f"  evaluator.py  — pytest correctness gate, {input_shape} benchmark ({benchmark_sizes})")
    print(f"  config.yaml   — {args.iterations} iterations, objective={args.objective}")

    # Output run dir path for the calling slash command
    print(f"\n__RUN_DIR__\n{run_dir}\n__RUN_DIR_END__")


if __name__ == '__main__':
    main()
