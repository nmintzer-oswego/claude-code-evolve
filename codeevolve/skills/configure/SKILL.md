---
name: configure
description: Configure CodeEvolve default settings (iterations, speedup threshold).
disable-model-invocation: true
---

# Configure

Configure CodeEvolve default settings. Writes to `.codeevolve/config.yaml` at the project root.

```bash
python -c "
from pathlib import Path
p = Path('.codeevolve/config.yaml')
if p.exists():
    print('Current settings:')
    print(p.read_text())
else:
    print('No config file yet.')
"
```

Show the current settings (or note that none exist yet). Then ask the user two questions (objective is fixed at `time` in v1.0):

1. **Note on objective** — In v1.0, the only supported objective is `time` (wall-clock speedup). Space and balanced objectives require a validated memory measurement protocol and are planned for v2.0. Always write `default_objective: time`.

2. **Default iteration count** — How many evolution iterations per run? Default: `20`. R6 finding: most problems reach optimal within 10 iterations; 20 gives headroom. Range: 10–100.

3. **Minimum speedup threshold** — Results below this are reported as "no significant improvement". Default: `1.1`. R5 finding: measurement noise floor is 1.03x; 1.1x is safely above it.

After the user answers, write `.codeevolve/config.yaml`:

```bash
python -c "
from pathlib import Path
Path('.codeevolve').mkdir(exist_ok=True)
Path('.codeevolve/config.yaml').write_text('''# CodeEvolve project configuration
default_objective: time
default_iterations: ITERATIONS
min_speedup_threshold: THRESHOLD
''')
print('Saved.')
"
```

Confirm: "Settings saved to `.codeevolve/config.yaml`. These will be used as defaults for future `/codeevolve:optimize` runs."
