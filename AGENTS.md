# Instructions for reproducing Copper-Policy

Read [agent_docs/REPRODUCING.md](agent_docs/REPRODUCING.md) before installing or
evaluating. Optional local experiment records in `agent_docs/` are ignored by
Git and are not prerequisites. Record your own revisions, compatibility fixes
and outcomes instead of assuming the historical results.
Any coding agent can follow these instructions without a plugin or skill.

## Required workflow

1. Verify current remote releases, hardware and the requested benchmarks.
2. Prepare the isolated environment, download sources and artifacts, verify
   checksums, and apply the documented compatibility fixes.
3. Perform setup checks and actual single-episode smoke evaluations for the
   requested benchmarks. Resolve execution errors and inspect saved summaries.
   A completed unsuccessful episode is a valid smoke outcome.
4. Report the smoke task selection, settings, outcomes and result paths. State
   that smoke results do not establish full benchmark success rates.
5. Provide ready-to-run full-evaluation, resume and log-monitoring commands,
   adapted to the user's hardware and output directory. Hand execution over to
   the user; do not launch full evaluation as part of the reproduction setup.

The agent's default deliverable is a prepared environment, verified smoke
results and concrete commands for the user. Starting or supervising a full run
requires a separate explicit user request; a general request to reproduce the
project is not authorization to start that long-running workload.

## User scope and isolation

- Honor the current user's scope. If asked for commands, prepare commands; do
  not launch evaluations. Do not duplicate a user-managed GPU workload.
- For fresh reproduction, clone upstream sources and download Python, packages,
  models and assets into a new directory. Do not reuse another local checkout,
  conda environment, model or cache. Do not use `ln -s` or `--link-local`.
  System drivers, CUDA, compilers, uv and ffmpeg remain system prerequisites.
- Use `.venv` and repository-local managed Python. `tools/fresh_run.sh` clears proxies
  and inherited Python paths, disables user site packages and isolates caches.
  Keep `--inexact` on uv sync to retain separately built cuRobo; use copy mode.
- Never edit or kill unrelated jobs. Identify this run's process tree before
  stopping it. The evaluation dispatcher forwards termination to workers.

## Verify sources and permissions

- Check the current upstream README, project page and linked Hugging Face
  repository before declaring artifacts unavailable. Record the git SHA and
  distinguish historical release status from current availability.
- Official policies: `https://huggingface.co/Mark455/Copper-Policy`.
  In the recorded current version, `tools.download_weights all --yes` downloads
  encoders AND policies. Verify policies against `SHA256SUMS` before loading.
- Clear upper- and lower-case proxy variables before reproduction networking.
  Sandbox DNS errors or invisible GPUs do not prove the host is broken. Use
  execution-tool approval if permitted; never bypass environment restrictions.
- Follow existing session authorization. Ask only for genuinely missing scope,
  hardware constraints or tool-required approvals.

## Setup details

- Keep RoboTwin pinned by `third_party.setup_sources`, and cuRobo pinned by
  `tools/setup_robotwin.sh`. Do not silently install a newer simulator API.
- Recorded RTX 5090 setup: CUDA 13.0, architecture `12.0+PTX`. Inspect actual
  hardware before using these; the wrapper permits CUDA and architecture overrides.
- LIBERO-Plus assets can contain a long upstream-machine path prefix. Move the
  downloaded assets into the correct package location; do not create empty
  directories or point at unrelated local assets merely to pass a check.
- Run `tools/patch_libero_init_states.py` after cloning. It allows NumPy loading
  only for official simulator initial-state files. Do not globally disable
  weights-only loading or change the policy checkpoint loader for this issue.
- Apply upstream RoboTwin compatibility patches and OIDN checksums, then run its
  embodiment configuration-path script as documented in the guide.

## Validate and report

- Follow layered checks and four actual single-episode commands in the guide.
  Imports, directory checks and dry runs do not prove real inference works.
- A normally completed failed episode is a valid benchmark result. Inspect
  `summary.json`, per-task results and logs; do not require every task to succeed.
- Report task selection, episode count, seed and compilation mode. Never call
  a single-episode result a paper success-rate reproduction. Full evaluation
  requires all requested tasks and episodes and `all_tasks_completed=true`.
- Record revisions, package versions, patches, commands and output paths. Keep
  historical outcomes distinct from subsequent runs. Training requires released
  training code and datasets; successful inference does not establish training.
- The root `.gitignore` is an allowlist. Ensure support files are tracked; keep
  environments, caches, upstream clones, weights, assets and outputs out of Git.
