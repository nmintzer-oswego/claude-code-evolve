"""
CodeEvolve Evolution Runner (SRD § 3.2.3).

Wires OpenEvolve (v0.2.26) with the ClaudeCodeLLM adapter and runs the
evolutionary optimization loop on a pre-built run directory.

The run directory must already contain (built by build_harness.py):
  - initial.py       — original function with EVOLVE-BLOCK markers
  - evaluator.py     — correctness tests + R5 performance measurement
  - config.yaml      — OpenEvolve configuration
  - claude_code_llm.py — LLM adapter

All LLM calls route through `claude -p` subprocess (stateless, R1-validated).
Results are written to <run-dir>/results/.

Source: Adapted from Research/R6_EndToEnd/run_problem.py.
Key changes:
- Accepts --run-dir <path> instead of --problem <name>
- No hardcoded PROBLEMS list
- Writes results to <run-dir>/results/

Usage:
    python run_evolution.py --run-dir <path> [--iterations N]
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows — OpenEvolve logs emoji (⚠️, ✅) that
# cp1252 (the default Windows console encoding) cannot encode.
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Reconfigure stdout/stderr to UTF-8 if they are open in text mode
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add the run directory's claude_code_llm.py to path at runtime
# (run_dir is passed as CLI arg; we import after parsing)

def setup_logging(run_dir: Path):
    """Configure logging to both console and file."""
    log_file = run_dir / 'results' / 'evolution.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Use UTF-8 stream handler to handle emoji from OpenEvolve on Windows
    stream_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stdout, 'buffer') else sys.stdout
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            stream_handler,
            logging.FileHandler(log_file, encoding='utf-8'),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


async def run_evolution(run_dir: Path, iterations: int = None) -> dict:
    """
    Run OpenEvolve on a pre-built run directory.

    Args:
        run_dir: Path to the run directory created by build_harness.py
        iterations: Override max_iterations from config.yaml (optional)

    Returns:
        Dict with: status, iterations_completed, elapsed_seconds,
                   best_score, correctness, avg_speedup, cost_usd,
                   best_program, best_metrics, error
    """
    logger = setup_logging(run_dir)

    # Add the local openevolve install to path if present (vendored copy).
    # Falls back to system-installed openevolve (pip install openevolve==0.2.26).
    project_root = run_dir.parent.parent.parent  # .codeevolve/runs/<id> → project root
    lib_path = project_root / 'lib' / 'openevolve_pkg'
    if lib_path.exists():
        sys.path.insert(0, str(lib_path))

    # Add run_dir to path for claude_code_llm.py
    sys.path.insert(0, str(run_dir))

    from openevolve.config import Config
    from openevolve.controller import OpenEvolve
    from claude_code_llm import create_claude_code_llm

    config_path = run_dir / 'config.yaml'
    initial_program = str(run_dir / 'initial.py')
    evaluator = str(run_dir / 'evaluator.py')
    output_dir = str(run_dir / 'results')

    logger.info("=" * 60)
    logger.info(f"CodeEvolve Evolution: {run_dir.name}")
    logger.info("OpenEvolve + Claude Code API")
    logger.info("=" * 60)

    # Load config
    config = Config.from_yaml(config_path)
    if iterations is not None:
        config.max_iterations = iterations

    # Inject ClaudeCodeLLM as the init_client for all model configs
    for model_cfg in config.llm.models:
        model_cfg.init_client = create_claude_code_llm
    for model_cfg in config.llm.evaluator_models:
        model_cfg.init_client = create_claude_code_llm

    logger.info(f"Initial program: {initial_program}")
    logger.info(f"Evaluator:       {evaluator}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info(f"Max iterations:  {config.max_iterations}")

    result = {
        "run_dir": str(run_dir),
        "status": "failed",
        "iterations_completed": 0,
        "early_stopped": False,
        "elapsed_seconds": 0,
        "time_per_iteration": 0,
        "best_score": 0.0,
        "correctness": 0.0,
        "avg_speedup": 0.0,
        "cost_usd": 0.0,
        "best_program": "",
        "best_metrics": {},
        "error": None,
    }

    start_time = time.time()

    try:
        evolve = OpenEvolve(
            config=config,
            initial_program_path=initial_program,
            evaluation_file=evaluator,
            output_dir=output_dir,
        )

        best = await evolve.run()
        elapsed = time.time() - start_time

        if best is None:
            logger.error("Evolution returned no valid programs!")
            result["status"] = "no_valid_programs"
            result["elapsed_seconds"] = elapsed
            return result

        best_metrics = best.metrics
        best_score = best_metrics.get('combined_score', best_metrics.get('score', 0.0))

        # Collect total cost from all LLM instances
        # OpenEvolve exposes ensembles as llm_ensemble and llm_evaluator_ensemble,
        # each with a .models list of ClaudeCodeLLM instances that track total_cost_usd.
        total_cost = 0.0
        try:
            for ensemble_attr in ('llm_ensemble', 'llm_evaluator_ensemble'):
                ensemble = getattr(evolve, ensemble_attr, None)
                if ensemble is not None:
                    for llm in getattr(ensemble, 'models', []):
                        if hasattr(llm, 'total_cost_usd'):
                            total_cost += llm.total_cost_usd
        except Exception:
            pass

        # Determine iterations completed from checkpoint
        iters_completed = config.max_iterations
        checkpoint_file = Path(output_dir) / 'checkpoint.json'
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    ckpt = json.load(f)
                iters_completed = ckpt.get('iteration', config.max_iterations)
            except Exception:
                pass

        result.update({
            "status": "completed",
            "iterations_completed": iters_completed,
            "early_stopped": iters_completed < config.max_iterations,
            "elapsed_seconds": elapsed,
            "time_per_iteration": elapsed / max(iters_completed, 1),
            "best_score": best_score,
            "correctness": best_metrics.get('correctness', 0.0),
            "avg_speedup": best_metrics.get('avg_speedup', best_score),
            "cost_usd": total_cost,
            "best_program": best.code,
            "best_metrics": {k: float(v) for k, v in best_metrics.items()},
        })

        logger.info("=" * 60)
        logger.info("EVOLUTION COMPLETE")
        logger.info(f"Best score:      {best_score:.2f}x")
        logger.info(f"Correctness:     {best_metrics.get('correctness', 0.0)}")
        logger.info(f"Total time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"Iterations:      {iters_completed}/{config.max_iterations}")
        logger.info(f"Cost:            ${total_cost:.4f}")
        logger.info("=" * 60)

        # Write results summary
        results_path = Path(output_dir) / 'optimization_report.md'
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"# CodeEvolve Optimization Report\n\n")
            f.write(f"**Run ID:** {run_dir.name}\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Results\n\n")
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Speedup | {best_score:.2f}x |\n")
            f.write(f"| Correctness | {best_metrics.get('correctness', 0.0):.0%} |\n")
            f.write(f"| Iterations | {iters_completed}/{config.max_iterations} |\n")
            f.write(f"| Early stopped | {iters_completed < config.max_iterations} |\n")
            f.write(f"| Total time | {elapsed:.1f}s ({elapsed/60:.1f} min) |\n")
            f.write(f"| Cost | ${total_cost:.4f} |\n\n")
            f.write(f"## Best Program\n\n```python\n{best.code}\n```\n")
        logger.info(f"Report written to {results_path}")

        # Also write best program to a standalone file
        best_path = Path(output_dir) / 'best_program.py'
        with open(best_path, 'w', encoding='utf-8') as f:
            f.write(best.code)

    except Exception as e:
        elapsed = time.time() - start_time
        result["elapsed_seconds"] = elapsed
        result["error"] = str(e)
        logger.error(f"Evolution failed after {elapsed:.1f}s: {e}", exc_info=True)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CodeEvolve evolution runner — runs OpenEvolve on a pre-built run directory"
    )
    parser.add_argument('--run-dir', required=True,
                        help="Path to run directory created by build_harness.py")
    parser.add_argument('--iterations', type=int, default=None,
                        help="Override max_iterations from config.yaml")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(run_evolution(run_dir, args.iterations))

    # Print summary to stdout for the calling slash command
    print(f"\n{'='*60}")
    print(f"EVOLUTION RESULT")
    print(f"Status:      {result['status']}")
    if result['status'] == 'completed':
        print(f"Speedup:     {result['avg_speedup']:.2f}x")
        print(f"Correctness: {result['correctness']:.0%}")
        print(f"Iterations:  {result['iterations_completed']}")
        print(f"Time:        {result['elapsed_seconds']:.1f}s")
        print(f"Cost:        ${result['cost_usd']:.4f}")
        if result['early_stopped']:
            print(f"Note:        Early stopped (converged before max iterations)")
    elif result['error']:
        print(f"Error:       {result['error']}")
    print(f"{'='*60}")

    # Output JSON for the slash command to parse
    print(f"\n__RESULT_JSON__")
    print(json.dumps(result))
    print(f"__RESULT_JSON_END__")

    return result


if __name__ == '__main__':
    main()
