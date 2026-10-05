"""Run the official full four-benchmark evaluation with a persistent batch log.

Run through: bash tools/fresh_run.sh -u tools/run_fresh_full.py
The fresh wrapper supplies isolated cache locations and removes proxies.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation/fresh_full_v1"))
parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
parser.add_argument("--order", default="robotwin_clean,libero,robotwin_rand,libero_plus")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
output = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
command = ["bash", "run.sh", "--out-dir", str(output),
           "--gpu-ids", args.gpu_ids, "--order", args.order, "--resume"]
if args.dry_run:
    sys.exit(subprocess.call([*command, "--dry-run"], cwd=ROOT))
output.mkdir(parents=True, exist_ok=True)
(output / "exit_code.txt").unlink(missing_ok=True)
(output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
print(f"Full evaluation log: {output / 'batch.log'}", flush=True)
with (output / "batch.log").open("a", buffering=1) as log:
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
(output / "exit_code.txt").write_text(str(result.returncode) + "\n")
print(f"Full evaluation exit code: {result.returncode}", flush=True)
sys.exit(result.returncode)
