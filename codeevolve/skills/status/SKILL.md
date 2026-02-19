---
name: status
description: Show progress of the most recent CodeEvolve optimization run.
disable-model-invocation: true
---

# Status

Show the progress of the most recent CodeEvolve optimization run.

```bash
python -c "
import os, json
from pathlib import Path
runs_dir = Path('.codeevolve/runs')
if not runs_dir.exists() or not list(runs_dir.iterdir()):
    print('No runs found.')
else:
    latest = sorted(runs_dir.iterdir())[-1]
    print(f'Run: {latest.name}')
    ckpt = latest / 'results' / 'checkpoint.json'
    if ckpt.exists():
        c = json.loads(ckpt.read_text())
        print(f'Iteration: {c.get(\"iteration\", \"?\")}')
        print(f'Best score: {c.get(\"best_score\", \"?\")}')
    report = latest / 'results' / 'optimization_report.md'
    if report.exists():
        print(report.read_text()[:800])
    log = latest / 'results' / 'evolution.log'
    if log.exists():
        lines = log.read_text(encoding='utf-8').splitlines()
        print('\nRecent log:')
        print('\n'.join(lines[-15:]))
"
```

Show the output to the user. If there are no runs, tell them to run `/codeevolve:optimize <function-file> <test-file>` to start.
