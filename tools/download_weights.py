"""Download/check Copper-Policy checkpoints and encoder presets without loading models."""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from wan_va.modules.backbone_presets import (
    WORLD_ENCODER_PRESETS, VISION_ENCODER_PRESETS, TEXT_ENCODER_PRESETS,
    CACHE_ROOT, world_cache_dir, world_cache_ready, vision_cache_dir, vision_cache_ready,
    text_cache_dir, text_cache_ready,
)
def cache_world(name: str) -> None:
    preset = WORLD_ENCODER_PRESETS[name]
    root = world_cache_dir(name)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    if not (source / "hubconf.py").is_file():
        if source.exists():
            raise RuntimeError(f"Incomplete source checkout at {source}; repair it before retrying")
        subprocess.run(["git", "clone", "--depth", "1", preset.source_repo, str(source)], check=True)
    checkpoint = root / preset.checkpoint_name
    if not checkpoint.is_file():
        partial = checkpoint.with_suffix(checkpoint.suffix + ".partial")
        urllib.request.urlretrieve(preset.checkpoint_url, partial)
        partial.replace(checkpoint)
    if not world_cache_ready(name):
        raise RuntimeError(f"World encoder cache incomplete: {root}")

def cache_vision(name: str) -> None:
    from huggingface_hub import snapshot_download

    preset = VISION_ENCODER_PRESETS[name]
    root = vision_cache_dir(name)
    snapshot_download(
        repo_id=preset.model_id,
        local_dir=str(root),
        allow_patterns=["config.json", "model.safetensors", "model.safetensors.index.json", "model-*.safetensors"],
    )
    if not vision_cache_ready(name):
        raise RuntimeError(f"Vision encoder cache incomplete: {root}")

def cache_text(name: str) -> None:
    from huggingface_hub import snapshot_download

    preset = TEXT_ENCODER_PRESETS[name]
    root = text_cache_dir(name)
    snapshot_download(
        repo_id=preset.model_id,
        local_dir=str(root),
        allow_patterns=["tokenizer/*", "text_encoder/config.json", "text_encoder/model.safetensors*", "text_encoder/model-*.safetensors"],
    )
    if not text_cache_ready(name):
        raise RuntimeError(f"Text encoder cache incomplete: {root}")

def ensure_one(kind: str, name: str, *, yes: bool, check: bool) -> None:
    root = {"world": world_cache_dir, "vision": vision_cache_dir, "text": text_cache_dir}[kind](name)
    ready = {"world": world_cache_ready, "vision": vision_cache_ready, "text": text_cache_ready}[kind](name)
    print(f"{kind} encoder {name}: {'cached' if ready else 'missing'} ({root})")
    if ready:
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = {"world": 5_500_000_000, "vision": 1_500_000_000, "text": 11_500_000_000}[kind]
    free_bytes = shutil.disk_usage(root.parent).free
    gib = 1024 ** 3
    print(f"Estimated download/storage: {estimated_bytes / gib:.2f} GiB; free: {free_bytes / gib:.2f} GiB")
    if check:
        return
    if estimated_bytes > free_bytes:
        raise SystemExit("Not enough free disk space for the pretrained encoder cache.")
    if not yes:
        if not sys.stdin.isatty():
            raise SystemExit("Cache missing. Run interactively or pass --yes to download.")
        answer = input("Download this pretrained encoder into the named cache directory? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Cache remains missing; download skipped.")
    {"world": cache_world, "vision": cache_vision, "text": cache_text}[kind](name)
    print(f"Cache ready: {root}")
POLICY_REPO = "Mark455/Copper-Policy"
POLICY_BYTES = {"libero": 4_056_018_675, "robotwin": 4_056_161_265}


def ensure_policy(name: str, *, yes: bool, check: bool) -> None:
    root = CACHE_ROOT / "copper_policy"
    checkpoint = root / name / "policy.pt"
    ready = checkpoint.is_file() and checkpoint.stat().st_size == POLICY_BYTES[name]
    print(f"policy {name}: {'cached' if ready else 'missing or incomplete'} ({checkpoint})")
    if ready or check:
        return
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    print(f"Download/storage: {POLICY_BYTES[name] / 1024 ** 3:.2f} GiB; free: {free_bytes / 1024 ** 3:.2f} GiB")
    if free_bytes < POLICY_BYTES[name]:
        raise SystemExit("Not enough free disk space for the policy checkpoint.")
    if not yes:
        if not sys.stdin.isatty():
            raise SystemExit("Policy missing. Run interactively or pass --yes to download.")
        if input(f"Download {name} from {POLICY_REPO}? [y/N] ").strip().lower() not in {"y", "yes"}:
            raise SystemExit("Policy download skipped.")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=POLICY_REPO, local_dir=str(root),
                      allow_patterns=[f"{name}/policy.pt", f"{name}/policy.json", "SHA256SUMS", "release_manifest.json"])
    if not checkpoint.is_file() or checkpoint.stat().st_size != POLICY_BYTES[name]:
        raise RuntimeError(f"Policy checkpoint incomplete: {checkpoint}")
    print(f"Policy ready: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("world", "vision", "text", "policy", "all"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.kind == "policy":
        if args.name not in (None, "all", *POLICY_BYTES):
            parser.error("policy selection must be libero, robotwin, or all")
        for name in POLICY_BYTES if args.name in (None, "all") else (args.name,):
            ensure_policy(name, yes=args.yes, check=args.check)
        return
    presets = {"world": WORLD_ENCODER_PRESETS, "vision": VISION_ENCODER_PRESETS, "text": TEXT_ENCODER_PRESETS}
    selected = {"world": os.getenv("COPPER_WORLD_ENCODER", "vjepa2_1_vit_large_384"),
                "vision": os.getenv("COPPER_VISION_ENCODER", "dinov2_with_registers_large"),
                "text": os.getenv("COPPER_TEXT_ENCODER", "wan22_ti2v_5b")}
    kinds = tuple(presets) if args.kind == "all" else (args.kind,)
    if args.kind == "all" and args.name:
        parser.error("all does not take a preset name")
    for kind in kinds:
        name = args.name or selected[kind]
        if name not in presets[kind]:
            parser.error(f"Unknown {kind} preset {name!r}; choices: {tuple(presets[kind])}")
        ensure_one(kind, name, yes=args.yes, check=args.check)
    if args.kind == "all":
        for name in POLICY_BYTES:
            ensure_policy(name, yes=args.yes, check=args.check)

if __name__ == "__main__":
    main()
