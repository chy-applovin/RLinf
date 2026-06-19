#!/usr/bin/env python3
"""Build an executed-notebook-friendly launch smoke report for TACO A1/A2/A3 runs."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name('taco_a123_launch_smoke.ipynb')

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
}

nb.cells = [
    nbf.v4.new_markdown_cell(
        "# TACO Allegro A1/A2/A3 Launch Smoke Report\n\n"
        "This notebook records the launch evidence for the three RL experiments started on 2026-06-19. "
        "It reads the persisted run directories, command files, W&B URLs, process status, and latest log metrics."
    ),
    nbf.v4.new_code_cell(r"""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path('/root/RLinf')
RUNS = [
    {
        'label': 'A1',
        'pid': 313798,
        'config': 'taco_allegro_ppo_flow_a1_single_step_hand',
        'gpus': '0,1',
        'mode': 'single_step + object + hand reward',
        'log_dir': ROOT / 'logs/20260619-004711-a1-singlestep-object-hand-10k',
        'wandb': 'https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/2n4xat4r',
    },
    {
        'label': 'A2',
        'pid': 313799,
        'config': 'taco_allegro_ppo_flow_a2_curriculum_object',
        'gpus': '2,3',
        'mode': 'demo-start horizon curriculum + object reward',
        'log_dir': ROOT / 'logs/20260619-004711-a2-demo-start-curriculum-object-10k',
        'wandb': 'https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/ohayni13',
    },
    {
        'label': 'A3',
        'pid': 313800,
        'config': 'taco_allegro_ppo_flow_a3_curriculum_hand',
        'gpus': '4,5',
        'mode': 'demo-start horizon curriculum + object + hand reward',
        'log_dir': ROOT / 'logs/20260619-004711-a3-demo-start-curriculum-object-hand-10k',
        'wandb': 'https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/4tka9pch',
    },
]

print('executed_at_utc:', datetime.now(timezone.utc).isoformat(timespec='seconds'))
print('root:', ROOT)
"""),
    nbf.v4.new_code_cell(r"""
def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)


def read_text(path: Path, limit_chars: int | None = None) -> str:
    if not path.exists():
        return ''
    data = path.read_text(errors='replace')
    if limit_chars is not None and len(data) > limit_chars:
        data = data[-limit_chars:]
    return strip_ansi(data)


def ps_status(pid: int) -> str:
    proc = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'pid=,stat=,etime=,cmd='],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or f'{pid} not running'


def latest_metric(text: str, key: str) -> str | None:
    values = re.findall(rf'{re.escape(key)}=([^\s│]+)', text)
    return values[-1] if values else None


def latest_step(text: str) -> str | None:
    values = re.findall(r'Global Step:\s*([0-9]+/[0-9]+)', text)
    if values:
        return values[-1]
    values = re.findall(r'Global Step\s+([0-9]+/[0-9]+)', text)
    return values[-1] if values else None

METRIC_KEYS = [
    'episode_len',
    'curriculum_horizon_steps',
    'single_step_frame',
    'demo_start_frame',
    'return',
    'reward',
    'success_once',
    'tool_pos_err_final_m',
    'target_pos_err_final_m',
    'hand_qpos_err_final',
    'critic/explained_variance',
]

rows = []
for run in RUNS:
    log_dir = run['log_dir']
    run_log = read_text(log_dir / 'run.log', limit_chars=500_000)
    metrics_log = read_text(log_dir / 'metrics.log', limit_chars=500_000)
    text = run_log + '\n' + metrics_log
    row = {
        'label': run['label'],
        'pid': run['pid'],
        'status': ps_status(run['pid']),
        'gpus': run['gpus'],
        'config': run['config'],
        'mode': run['mode'],
        'log_dir': str(log_dir),
        'command_file': str(log_dir / 'command.txt'),
        'resolved_config': str(log_dir / 'tensorboard/config.yaml'),
        'wandb': run['wandb'],
        'latest_step': latest_step(text),
    }
    for key in METRIC_KEYS:
        value = latest_metric(text, key)
        if value is not None:
            row[key] = value
    rows.append(row)

print(json.dumps(rows, indent=2))
"""),
    nbf.v4.new_code_cell(r"""
for run in RUNS:
    print('=' * 100)
    print(run['label'], run['mode'])
    print('W&B:', run['wandb'])
    print('command:')
    print(read_text(run['log_dir'] / 'command.txt').strip())
    print('\nlatest run.log tail:')
    tail = '\n'.join(read_text(run['log_dir'] / 'run.log', limit_chars=80_000).splitlines()[-35:])
    print(tail)
"""),
]

nbf.write(nb, OUT)
print(OUT)
