"""Run the official full four-benchmark evaluation with a persistent batch log.

Run through: bash tools/fresh_run.sh -u tools/run_fresh_full.py
The fresh wrapper supplies isolated cache locations and removes proxies.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "outputs/evaluation/fresh_full_v1"
output.mkdir(parents=True, exist_ok=True)
(output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
command = ["bash", "run.sh", "--out-dir", str(output),
           "--gpu-ids", "0,1,2,3,4,5,6,7", "--resume"]
print(f"Full evaluation log: {output / 'batch.log'}", flush=True)
with (output / "batch.log").open("a", buffering=1) as log:
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
(output / "exit_code.txt").write_text(str(result.returncode) + "\n")
print(f"Full evaluation exit code: {result.returncode}", flush=True)
sys.exit(result.returncode)
