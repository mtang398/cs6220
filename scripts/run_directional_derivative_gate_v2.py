"""Serialize the diagnostic, retain its command receipt, and charge the campaign."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
STEM = 'directional-derivative-preconditioning-v2'
receipt = ROOT / f'results/{STEM}-command.json'
log = ROOT / f'results/{STEM}-command.log'
ledger = ROOT / 'docs/method-revision.json'
assert not receipt.exists() and not log.exists()
before = json.loads(ledger.read_text())
assert before['campaign_command_wall_seconds'] - before['consumed_command_seconds'] >= 150
cmd = ['timeout', '--signal=TERM', '--kill-after=5s', '145s', 'flock', '-w', '1',
       '/tmp/cs6220-project-benchmark.lock', sys.executable,
       str(ROOT/'scripts/check_directional_derivative_preconditioning_v2.py')]
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           BLIS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
record = dict(command=cmd, cwd=str(ROOT), started_at=datetime.now(timezone.utc).isoformat(),
              blas_threads=1, external_timeout_seconds=145, script_cpu_limit_seconds=[110,115],
              source_sha256=sha256((ROOT/'scripts/check_directional_derivative_preconditioning_v2.py').read_bytes()).hexdigest(),
              campaign_consumed_before=before['consumed_command_seconds'])
receipt.write_text(json.dumps(record, indent=2)+'\n')
t = perf_counter()
with log.open('w') as stream:
    completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
elapsed = perf_counter()-t
record.update(returncode=completed.returncode, wall_seconds=elapsed,
              finished_at=datetime.now(timezone.utc).isoformat())
now = json.loads(ledger.read_text())
assert now['consumed_command_seconds'] == before['consumed_command_seconds']
now['consumed_command_seconds'] += elapsed
now['events'].append(dict(experiment=f'results/{STEM}.json', command_receipt=str(receipt.relative_to(ROOT)),
    status='executed_pending_audit' if completed.returncode==0 else 'command_failed_pending_diagnosis',
    elapsed_seconds=elapsed, returncode=completed.returncode, scientific_success=False))
ledger.write_text(json.dumps(now, indent=2, ensure_ascii=False)+'\n')
record['campaign_consumed_after'] = now['consumed_command_seconds']
receipt.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
raise SystemExit(completed.returncode)
