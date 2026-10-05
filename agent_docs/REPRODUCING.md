# Reproducing Copper-Policy from scratch

Read [../AGENTS.md](../AGENTS.md) first. This guide captures the setup and checks
performed on Linux with eight RTX 5090 GPUs. The agent helpers must be present
in your checkout: local additions become available to other clones only after
they are committed and published. Do not assume the canonical upstream already
contains these helpers.

## 1. Confirm scope, current releases and hardware

Distinguish setup, smoke evaluation, full evaluation and training. When the user
asks only for commands, prepare commands without starting workloads.
The default agent workflow is to prepare the environment, complete actual smoke
tests, report their results, and give the user full-evaluation and monitoring
commands. The user starts the full run. Do not launch a long full evaluation
unless the user separately and explicitly requests execution or supervision.

Check the current README, project page and their actual download links:

- Source: https://github.com/Mark4551124015/Copper-Policy
- Project: https://zexinfeng-cn.github.io/works/copper-policy/
- Policies: https://huggingface.co/Mark455/Copper-Policy
- LIBERO-Plus assets: https://huggingface.co/datasets/Sylvest/LIBERO-plus
- RoboTwin assets: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0

The verified source revision was `57cc8ea0a78b61b665814c37fabedac872db9af1` on
2026-10-05. Both policies were released; training code and training datasets were
still pending. Recheck before making availability claims: an older checkout said
policies were unavailable, but the updated README and project page linked them.

System prerequisites: Linux x86_64, uv, ffmpeg, a C++ compiler, NVIDIA drivers,
a compatible CUDA toolkit and ImageMagick shared libraries. Inspect
`nvidia-smi`, `nvcc --version`, `df -h .` and `command -v uv ffmpeg`.
The tested driver was 580.173.02; the lockfile used Python 3.10.16 and
PyTorch 2.9.0+cu130. RTX 5090 builds used CUDA 13.0 and `12.0+PTX`.
Allow at least 100 GB for setup and additional space for evaluation videos.
For other hardware, adjust CUDA/PyTorch sources and GPU architecture deliberately;
regenerate `uv.lock` if needed and record deviations.

## 2. Use a new checkout, Python and dependencies

Start from a remote checkout in a new directory. Do not copy a previous project,
activate an existing conda environment, reuse caches or link local assets.

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
git clone https://github.com/Mark4551124015/Copper-Policy.git Copper-Policy-fresh
cd Copper-Policy-fresh
git rev-parse HEAD

unset PYTHONPATH PYTHONHOME UV_PROJECT_ENVIRONMENT
export UV_CACHE_DIR="$PWD/.fresh-cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.fresh-python"
export UV_NO_CACHE=1
export UV_LINK_MODE=copy
export HF_HOME="$PWD/.fresh-cache/huggingface"
export TORCH_HOME="$PWD/.fresh-cache/torch"
export MPLCONFIGDIR="$PWD/.fresh-cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.fresh-cache/xdg"
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST='12.0+PTX'
export MAX_JOBS=4

uv sync --frozen --all-extras --inexact --no-cache \
  --link-mode copy --python-preference only-managed
```

Use `bash tools/fresh_run.sh ...` for subsequent Python commands. It changes
to the project root, clears proxies and inherited Python paths, and uses the new
`.venv` with isolated caches. CUDA and architecture accept environment overrides.
When these helpers are absent from the upstream version you cloned, use
`.venv/bin/python` with the exports above; the helper files need to be distributed
with the repository revision used by the next agent.

Sandbox DNS failures or invisible GPUs do not establish that the host is broken.
Request tool-mediated execution outside the sandbox if the session permits it.
Do not bypass restrictions or alter DNS to evade sandbox policy.

## 3. Download sources, encoders and policies

```bash
bash tools/fresh_run.sh -m third_party.setup_sources all
bash tools/fresh_run.sh -m tools.download_weights all --yes
bash tools/fresh_run.sh -m tools.download_weights all --check

cd pretrained_weights/copper_policy
sha256sum -c SHA256SUMS
cd ../..
```

The current `all` downloader obtains three encoders and both policy checkpoints.
For an older downloader, inspect upstream updates or download the policies and
release metadata directly from the official Hugging Face repository into
`pretrained_weights/copper_policy/`. Do not substitute existing local models.

The verified policy sizes were 4,056,018,675 bytes for LIBERO and 4,056,161,265
bytes for RoboTwin. Prefer the corresponding release's `release_manifest.json`
and SHA256SUMS over assuming these sizes will never change. Do not load a policy
whose checksum fails.

## 4. Prepare LIBERO and LIBERO-Plus

LIBERO includes its required simulator assets. Download the extra Plus archive:

```bash
.venv/bin/hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset --local-dir third_party/LIBERO-plus
unzip -oq third_party/LIBERO-plus/assets.zip -d third_party/LIBERO-plus/libero/libero
```

The tested archive contained a long upstream-machine prefix. Normalize the
downloaded directory location without linking external assets:

```bash
bash tools/fresh_run.sh - <<'PY'
from pathlib import Path
import shutil
root = Path("third_party/LIBERO-plus/libero/libero")
target = root / "assets"
if not target.exists():
    candidates = [p for p in root.rglob("assets") if (p / "new_objects").is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Inspect downloaded archive layout: {candidates}")
    shutil.move(str(candidates[0]), str(target))
assert (target / "new_objects").is_dir()
print("LIBERO-Plus assets ready:", target)
PY

bash tools/fresh_run.sh tools/patch_libero_init_states.py
```

The patch is repeatable. PyTorch >=2.6 defaults to weights-only loading, whereas
official LIBERO initial states contain NumPy objects. This patch changes only
official simulator initial-state loading to `weights_only=False`; policy loading
is unchanged. Do not apply a global weights-only override.

## 5. Prepare RoboTwin, cuRobo and rendering

Keep the environment exports from step 2 active:

```bash
bash tools/setup_robotwin.sh

cd third_party/RoboTwin/assets
../../../.venv/bin/python _download.py
unzip -oq background_texture.zip
unzip -oq embodiments.zip
unzip -oq objects.zip
cd ..
../../.venv/bin/python script/update_embodiment_config_path.py
cd ../..
```

The setup script builds cuRobo, applies SAPIEN/MPLib compatibility patches and
installs checksum-verified official OIDN 2.3.3 for Blackwell rendering. Preserve
RoboTwin `c3ddfa8b97d5519efa828b075999bd0006778e5e` and cuRobo
`d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (v0.7.8); a newer simulator main
may expose an incompatible API. Generate embodiment configuration paths for this
checkout using the official script. Keep archives for inspection rather than
running the upstream asset script's removal commands.

Retain `--inexact` on subsequent uv sync operations so cuRobo is not removed.
Rebuild it after changing Python or PyTorch.

## 6. Validate setup and run actual smoke episodes

First check sources, assets, imports and CUDA execution:

```bash
bash tools/fresh_run.sh -m third_party.setup_sources all --check
bash tools/fresh_run.sh -m evaluation.run libero --check
bash tools/fresh_run.sh -m evaluation.run libero-plus --check
bash tools/fresh_run.sh -m evaluation.run robotwin --check
bash tools/fresh_run.sh -c 'import torch; import curobo.curobolib.geom_cu; import sapien; from wand.image import Image; x=torch.randn(64,64,device="cuda"); y=x@x; torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), y.shape)'
```

Next run actual policy inference, actions and result generation. These commands
run sequentially on one GPU, with one episode per selected task and compilation
disabled. Do not duplicate a user-managed workload.

```bash
bash tools/fresh_run.sh -c 'from pathlib import Path; import json; p=Path("outputs/fresh_smoke/one_libero_task.json"); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps([["libero_spatial",0]]))'

bash tools/fresh_run.sh -m evaluation.libero_mot.eval_tasks \
  --ckpt pretrained_weights/copper_policy/libero/policy.pt \
  --out-dir outputs/fresh_smoke/libero_policy --gpu-ids 0 \
  --task-manifest outputs/fresh_smoke/one_libero_task.json --smoke --no-compile

bash tools/fresh_run.sh -m evaluation.libero_mot.eval_tasks --backend libero-plus \
  --ckpt pretrained_weights/copper_policy/libero/policy.pt \
  --out-dir outputs/fresh_smoke/libero_plus_policy --gpu-ids 0 \
  --task-manifest outputs/fresh_smoke/one_libero_task.json --smoke --no-compile

bash tools/fresh_run.sh -m evaluation.robotwin.eval_tasks \
  --ckpt pretrained_weights/copper_policy/robotwin/policy.pt \
  --out-dir outputs/fresh_smoke/robotwin_policy --gpu-ids 0 \
  --eval-num-episodes 1 --tasks adjust_bottle --no-compile

bash tools/fresh_run.sh -m evaluation.robotwin.eval_tasks \
  --ckpt pretrained_weights/copper_policy/robotwin/policy.pt \
  --out-dir outputs/fresh_smoke/robotwin_rand_policy --gpu-ids 0 \
  --task-config demo_randomized --eval-num-episodes 1 --tasks adjust_bottle --no-compile
```

The historical smoke run used LIBERO seed 7 and RoboTwin seed 0 on physical GPUs
0, 2, 1 and 3 respectively. The first three benchmarks succeeded in 1/1 episodes;
randomized succeeded in 0/1, with all four evaluations completing normally.
Record your own outcomes. A failed task is not necessarily a broken environment.
Imports, asset checks and dry runs do not replace actual policy episodes.
Reusing an output directory can skip completed episodes; choose a new directory
for an independent experiment, without deleting the user's existing results.

## 7. Hand full-evaluation commands to the user

After smoke testing, report the task selection, episode counts, settings and
outcomes, then provide the commands below for the user to run. Adapt GPU choices
and output directories first. Do not execute these full-evaluation commands as
part of the default agent workflow.

The user can start or resume the reference eight-GPU evaluation with:

```bash
bash tools/fresh_run.sh -u tools/run_fresh_full.py
```

This calls upstream `run.sh` with compilation and videos enabled, in this order:
RoboTwin clean (50×50), LIBERO (40×50), RoboTwin randomized (50×50) and LIBERO-Plus
(10,030×1). Expect hours of execution. Change the GPU list and output path in
`tools/run_fresh_full.py` when the user requests another configuration.
Alternatively, with the step-2 exports active, run
`bash run.sh --gpu-ids 0 --out-dir outputs/evaluation/my_run --resume` for one GPU.

Give the user these commands to monitor from another terminal:

```bash
tail -f outputs/evaluation/fresh_full_v1/batch.log
cat outputs/evaluation/fresh_full_v1/status.tsv
```

Worker logs are in each benchmark's `attempt_N/persistent_gpu*.log` and final
results in its `summary.json`. Verify task coverage, episode counts, exit codes
and `all_tasks_completed=true`. A log line saying Success is not full completion.
`exit_code.txt` describes the most recent exited launcher and can remain stale
while a resumed run is active. A single episode cannot establish the paper rate.

If asked to stop, identify this dispatcher and its worker process tree; do not
globally kill Python processes. Retain results and logs for resuming. The historical
full run was started and then stopped at the user's request; it was not completed.

## 8. Leave a reviewable handoff

Record source SHAs, dependencies, checksum validation, patches, commands and
result paths. Local `FRESH_REPRODUCTION.md` and `fresh_environment.json`, if present,
are historical experiment records and intentionally ignored by Git. They are not
prerequisites for another agent and must not be mistaken for its current state.

Track the generic instructions and helpers. Keep environments, caches, nested
upstream clones, weights, assets and results out of Git. Inspect
`git check-ignore -v <file>` and `git status --short`; the root ignore file uses
an explicit allowlist. Do not commit or push without user authorization.

If training artifacts are still unreleased, report that limitation explicitly.
Released policy evaluation can proceed independently; successful inference does
not establish training reproduction.
