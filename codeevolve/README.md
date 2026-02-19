# CodeEvolve

Evolutionary code optimization for Python functions, powered by Claude Code.

Give CodeEvolve a Python function and a test suite. It runs evolutionary search — LLM-generated mutations, correctness gating, performance measurement — and hands back a faster implementation. All LLM calls route through your Claude Code subscription. No separate API key.

**Empirically validated:** 5/5 test problems evolved from brute-force to known-optimal algorithms within 10 iterations. 49x–1316x speedups. 100% correctness preservation. $0.14–$0.44/problem.

---

## Prerequisites

- **Claude Code CLI ≥ 1.0.33** with an active subscription — [install](https://claude.ai/download)
- **Python ≥ 3.10**
- **openevolve** and **pytest** (installed below)

## Installation

```bash
# 1. Clone or download this repo
git clone https://github.com/<your-repo>/codeevolve.git
cd your-project

# 2. Install Python dependencies
pip install openevolve==0.2.26 pytest

# 3. Load the plugin
claude --plugin-dir ./codeevolve
```

That's it. The three commands are now available in your Claude Code session.

---

## Usage

### Optimize a function

```
/codeevolve:optimize my_function.py test_my_function.py
```

With options:
```
/codeevolve:optimize my_function.py test_my_function.py --iterations 30
/codeevolve:optimize my_function.py test_my_function.py --function sort_items --iterations 10 --yes
```

### Check run progress

```
/codeevolve:status
```

### Configure defaults

```
/codeevolve:configure
```

---

## What your function must look like

Your function must be:
- **Self-contained** — no I/O, network, or database access
- **Pure computation** — deterministic: same inputs always produce the same output
- **Importable** — the file can be imported as a standalone Python module

Good candidates: sorting, search, graph algorithms, string processing, mathematical computation.

Not supported in v1.0: functions with file I/O, HTTP requests, database queries, random state, class methods that depend on instance state, or functions that import third-party packages not available in the evaluator.

## What your test file must look like

Use a standard pytest-compatible test file:

```python
from my_function import two_sum

def test_basic():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]

def test_negative():
    assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]

def test_large():
    nums = list(range(1000))
    assert two_sum(nums, 999) == [0, 999]
```

**For best results:**
- Use `assert func(args) == expected` directly — CodeEvolve's AST extractor picks these up automatically
- Include at least one test with large inputs (aim for >1ms per call) for reliable performance measurement
- Cover edge cases: empty inputs, single elements, duplicates, negatives

If your tests use fixtures or complex setup, the AST extractor may not capture them. In that case, CodeEvolve will warn you and generate an evaluator with a `# TODO: fill in TEST_CASES` marker — you can edit it manually before running evolution.

---

## How it works

```
/codeevolve:optimize function.py tests.py
        │
        ▼
preflight.py     — validate function + tests, measure baseline performance
        │
        ▼
build_harness.py — generate OpenEvolve run directory (initial.py, evaluator.py, config.yaml)
        │
        ▼
run_evolution.py — OpenEvolve loop: LLM mutations → correctness gate → performance measurement
  └── claude_code_llm.py — routes all LLM calls through `claude -p` (stateless, ~5–15s/iter)
        │
        ▼
Results: best_program.py + optimization_report.md in .codeevolve/runs/<run-id>/results/
```

Run outputs are saved to `.codeevolve/runs/<run-id>/` in your project directory.

---

## Costs

| Function size | Time/iteration | Cost/100 iterations |
|---|---|---|
| Small (~20 lines) | ~5.5s | ~$1.40 |
| Medium (~100 lines) | ~8s | ~$2.80 |
| Large (~300 lines) | ~15s | ~$4.38 |

Default: 20 iterations → $0.28–$0.88. Most problems converge in 5–10 iterations.

---

## Limitations (v1.0)

- Python only (TypeScript/JavaScript planned for v1.1)
- Self-contained functions only — no external dependencies, I/O, or side effects
- Developer must provide test suite — auto-generation planned for v2.0
- Optimization objective: time (wall-clock speedup) only — space/balanced deferred to v2.0
- Test case extraction uses AST patterns; complex fixtures may require manual editing of the generated evaluator

---

## Troubleshooting

**"WARNING: No test cases extracted"**
Your test file uses fixtures or patterns the AST extractor doesn't recognize. Open the generated `.codeevolve/runs/<run-id>/evaluator.py` and fill in `TEST_CASES` manually following the format shown in the `# TODO` comment.

**Evaluator times out**
The benchmark inputs are too large for the brute-force baseline. Edit `BENCHMARK_SIZES` in the generated `evaluator.py` to use smaller values (aim for <1s per call on the original function).

**Preflight fails: tests don't pass on the original**
All tests must pass on the starting function before evolution begins. Fix your tests or function before running `/codeevolve:optimize`.

**Cost shows $0.0000**
Cost tracking reads `total_cost_usd` from the Claude Code JSON response. If your version of Claude Code doesn't include this field, cost will show as zero — this is cosmetic and doesn't affect evolution.
