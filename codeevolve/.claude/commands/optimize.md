# Optimize

Run CodeEvolve to evolve a Python function to be faster or more memory-efficient.

The user's message contains the arguments after `/optimize`. Parse them:
- First path argument = function file (Python file containing the function to optimize)
- Second path argument = test file (pytest-compatible test file)
- `--objective time|space|balanced` (default: time)
- `--iterations N` (default: 20)
- `--function <name>` (only needed if the file has multiple top-level functions)
- `--yes` (skip the confirmation prompt)

If the user did not provide both a function file and a test file, ask for the missing ones before proceeding.

---

## Step 1: Preflight check

Run with the arguments the user provided (substitute actual values for FUNCTION_FILE, TEST_FILE, etc.):

```bash
python codeevolve/scripts/preflight.py FUNCTION_FILE TEST_FILE --objective OBJECTIVE --iterations ITERATIONS
```

Show the full output to the user. If the script exits non-zero, stop and report the error. Do not proceed to Step 2.

Extract the JSON blob printed between the lines `__PREFLIGHT_JSON__` and `__PREFLIGHT_JSON_END__` — you will pass it to the next step.

If the output contains "WARNING: No test cases extracted", warn the user that the correctness gate will be disabled and ask whether to proceed or edit the evaluator first.

## Step 2: Build the run harness

Generate a run ID using the current date and time in `YYYYMMDD_HHMMSS` format. Then run:

```bash
python codeevolve/scripts/build_harness.py FUNCTION_FILE TEST_FILE --run-id RUN_ID --objective OBJECTIVE --iterations ITERATIONS --preflight-json 'PREFLIGHT_JSON'
```

Show the output. Extract the run directory path printed between `__RUN_DIR__` and `__RUN_DIR_END__`.

## Step 3: Run evolution

```bash
python codeevolve/scripts/run_evolution.py --run-dir RUN_DIR
```

Tell the user evolution is running and will take ~2–6 minutes. Show the output as it arrives. Extract the result JSON printed between `__RESULT_JSON__` and `__RESULT_JSON_END__`.

## Step 4: Present results

**If avg_speedup > 1.1 and correctness == 1.0:**

Show a clear summary (speedup, iterations, time, cost). Read `RUN_DIR/results/best_program.py` and show it. Show a diff against the original function. Ask the user:
- **Accept** — replace the function in the original file in-place (use AST to edit only the function, not the whole file)
- **Reject** — discard, keep original
- **Iterate** — run more iterations (ask how many, then repeat Step 3)

**If avg_speedup <= 1.1:**

Report no significant improvement found. Suggest trying `--iterations 50` or a different `--objective`.

**If status is "failed" or "no_valid_programs":**

Read the last 20 lines of `RUN_DIR/results/evolution.log` and report the error. Common causes: empty TEST_CASES, evaluator timeout, import errors in evolved code.
