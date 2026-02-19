# CodeEvolve — Claude Code Extension

CodeEvolve uses evolutionary optimization to make Python functions faster. It runs OpenEvolve (v0.2.26) with all LLM calls routed through the Claude Code API (`claude -p` subprocess calls), keeping costs within the developer's Claude Code subscription.

## What This Extension Does

Given a function and a test suite, CodeEvolve:
1. Validates the function and tests (preflight check)
2. Packages them into an OpenEvolve run directory
3. Runs evolutionary optimization (LLM-generated mutations, correctness gating, performance measurement)
4. Presents the best result with a diff, speedup, and accept/reject option

## Available Commands

When loaded as a plugin (`claude --plugin-dir ./codeevolve`), commands are namespaced:
- `/codeevolve:optimize <function-file> <test-file>` — Run optimization (main command)
- `/codeevolve:status` — Show progress of current/last run
- `/codeevolve:configure` — Configure defaults (objective, iterations, thresholds)

When used standalone (files in `.claude/commands/` at project root):
- `/optimize`, `/status`, `/configure` (no namespace)

## Project Structure

```
codeevolve/
├── .claude/commands/       # Slash command definitions
├── scripts/
│   ├── preflight.py        # Input validation + baseline measurement
│   ├── build_harness.py    # OpenEvolve run directory assembly
│   ├── run_evolution.py    # OpenEvolve orchestrator
│   ├── claude_code_llm.py  # LLM adapter (routes all calls through claude -p)
│   └── measure_performance.py  # R5 measurement protocol
└── config/
    └── default_config.yaml # OpenEvolve defaults

.codeevolve/               # Created at project root during runs
├── config.yaml            # Project-level settings (from /project:configure)
└── runs/
    └── <run-id>/          # One directory per optimization run
        ├── initial.py     # Original function + EVOLVE-BLOCK markers
        ├── evaluator.py   # Correctness tests + fitness measurement
        ├── config.yaml    # OpenEvolve config for this run
        └── results/       # Evolution output (best_program.py, report, logs)
```

## Key Constraints

- **Self-contained functions only** — Functions must not have I/O, network, or database access
- **Developer provides tests** — The test suite is the correctness oracle; tests must pass on the original before evolution starts
- **Python 3.10+** — All scripts require Python 3.10 or later
- **OpenEvolve dependency** — The `lib/openevolve_pkg/` directory must exist at the project root (already installed)

## Architecture Notes

- All LLM calls use `claude -p` in stateless mode (no session persistence)
- Prompts are piped via stdin (never CLI args) — Windows compatibility requirement
- Each iteration runs in ~5–15s; a 20-iteration run costs ~$0.14–$0.44
- R6 empirical validation: 5/5 problems evolved to optimal within 10 iterations, 49x–1316x speedups

## Running Scripts Directly

```bash
# Preflight check only
python codeevolve/scripts/preflight.py my_func.py test_my_func.py

# Build harness only (after preflight)
python codeevolve/scripts/build_harness.py my_func.py test_my_func.py --run-id test_run --objective time

# Run evolution on an existing run directory
python codeevolve/scripts/run_evolution.py --run-dir .codeevolve/runs/test_run
```
