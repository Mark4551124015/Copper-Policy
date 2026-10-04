"""Select a bundled simulator source and forward evaluation arguments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from third_party.setup_sources import source_dir, source_ready


ROOT = Path(__file__).resolve().parents[1]


def _has_option(args: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


def _prepare_libero_env(backend: str) -> dict[str, str]:
    source = source_dir(backend).resolve()
    package = source / "libero" / "libero"
    config_dir = ROOT / "third_party" / ".libero_config" / backend
    config_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    config = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(source / "libero" / "datasets"),
        "assets": str(package / "assets"),
    }
    temporary = config_dir / f".config.{os.getpid()}.yaml"
    temporary.write_text(yaml.safe_dump(config), encoding="utf-8")
    temporary.replace(config_dir / "config.yaml")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source), str(ROOT), env.get("PYTHONPATH", ""))))
    env["LIBERO_CONFIG_PATH"] = str(config_dir)
    env["COPPER_LIBERO_BACKEND"] = backend
    env.setdefault("MUJOCO_GL", "egl")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("libero", "libero-plus", "robotwin"))
    parser.add_argument("--check", action="store_true", help="Verify source and assets without evaluating or using a GPU")
    args, forwarded = parser.parse_known_args()
    source = source_dir(args.backend)
    if not source_ready(args.backend):
        raise SystemExit(
            f"{args.backend} source missing at {source}. "
            f"Run 'python -m third_party.setup_sources {args.backend}' first."
        )
    if args.backend in ("libero", "libero-plus"):
        package = source / "libero" / "libero"
        required = [package / "assets", package / "bddl_files", package / "init_files"]
        if args.backend == "libero-plus":
            required.append(package / "assets" / "new_objects")
        missing = [str(path) for path in required if not path.is_dir()]
        if missing:
            hint = " Download LIBERO-Plus assets.zip from https://huggingface.co/datasets/Sylvest/LIBERO-plus" if args.backend == "libero-plus" else ""
            raise SystemExit(f"{args.backend} required directories missing: {', '.join(missing)}.{hint}")
    else:
        assets = source / "assets" / "objects"
        if not assets.is_dir():
            raise SystemExit(f"RoboTwin assets missing at {assets}; run its script/_download_assets.sh")
    if args.check:
        if args.backend in ("libero", "libero-plus"):
            subprocess.run([sys.executable, "-c",
                            "from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; print('Simulator imports: PASS')"],
                           cwd=ROOT, env=_prepare_libero_env(args.backend), check=True)
        print(f"{args.backend}: source and required asset directories ready ({source})")
        return
    if not _has_option(forwarded, "--ckpt"):
        parser.error("Evaluation requires --ckpt")
    if args.backend == "robotwin":
        command = [sys.executable, str(ROOT / "evaluation" / "robotwin" / "eval_robotwin_mot.py")]
        if not _has_option(forwarded, "--robotwin-root"):
            forwarded.extend(("--robotwin-root", str(source)))
        env = os.environ.copy()
    else:
        command = [sys.executable, str(ROOT / "evaluation" / "libero_mot" / "eval_libero.py")]
        env = _prepare_libero_env(args.backend)
        if not _has_option(forwarded, "--description-source"):
            forwarded.extend(("--description-source", "mapping"))
        if args.backend == "libero-plus":
            for name, value in (
                ("--num-trials", "1"),
                ("--description-source", "mapping"),
                ("--out-dir", "outputs/libero_plus"),
            ):
                if not _has_option(forwarded, name):
                    forwarded.extend((name, value))
    if not _has_option(forwarded, "--t5-dir"):
        forwarded.extend(("--t5-dir", "pretrained_weights/text_encoder/wan22_ti2v_5b"))
    from wan_va.modules.backbone_presets import vision_cache_dir, vision_cache_ready

    if not _has_option(forwarded, "--dino-model") and vision_cache_ready("dinov2_with_registers_large"):
        forwarded.extend(("--dino-model", str(vision_cache_dir("dinov2_with_registers_large"))))
    raise SystemExit(subprocess.call([*command, *forwarded], cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
