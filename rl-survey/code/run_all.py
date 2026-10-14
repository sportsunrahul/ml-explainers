"""Run every chapter exercise and save its printed output for the page.

usage: .venv/bin/python code/run_all.py [substring ...]
Each code/chNN-<slug>/{numpy_ex.py,torch_ex.py} runs with its own folder as cwd and code/kit on
PYTHONPATH (the page tells the reader to save the kit files next to the script). Output goes to
code/out/<folder>--<script>.txt. A non-zero exit stops the run: the page must never show code that fails.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
PY = CODE.parent / '.venv' / 'bin' / 'python'
OUT = CODE / 'out'
OUT.mkdir(exist_ok=True)
wanted = sys.argv[1:]
env = dict(os.environ, PYTHONPATH=str(CODE / 'kit'), PYTHONHASHSEED='0')
for folder in sorted(CODE.glob('ch*')):
    for name in ('numpy_ex.py', 'torch_ex.py'):
        script = folder / name
        tag = f'{folder.name}--{name}'
        if not script.exists() or (wanted and not any(w in tag for w in wanted)):
            continue
        t0 = time.time()
        r = subprocess.run([str(PY), name], cwd=folder, env=env, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            sys.exit(f'FAILED {tag} after {dt:.0f}s\n{r.stdout}\n{r.stderr}')
        (OUT / f'{tag}.txt').write_text(r.stdout)
        print(f'ok {tag} {dt:.1f}s')
