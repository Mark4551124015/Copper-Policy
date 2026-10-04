"""Place upstream simulator source checkouts under third_party/."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


THIRD_PARTY = Path(__file__).resolve().parent
SOURCES = {
    "libero": ("LIBERO", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", "libero/libero/benchmark/__init__.py"),
    "libero-plus": ("LIBERO-plus", "https://github.com/sylvestf/LIBERO-plus.git", "libero/libero/benchmark/__init__.py"),
    "robotwin": ("RoboTwin", "https://github.com/RoboTwin-Platform/RoboTwin.git", "script/eval_policy.py"),
}
REVISIONS = {
    # This is the simulator revision used for the published inference checks.
    # Newer main uses scripts/ and env_cfg/ plus a different evaluation API.
    "robotwin": "c3ddfa8b97d5519efa828b075999bd0006778e5e",
}


def checkout_revision(name: str, path: Path) -> None:
    revision = REVISIONS.get(name)
    if revision is None:
        return
    # Preserve edits and downloaded assets when repairing a fresh main clone.
    changes = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if changes.strip():
        raise SystemExit(f"Tracked source changes at {path}; cannot switch to {revision}")
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)


def source_dir(name: str) -> Path:
    return THIRD_PARTY / SOURCES[name][0]


def source_ready(name: str) -> bool:
    return (source_dir(name) / SOURCES[name][2]).is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=(*SOURCES, "all"))
    parser.add_argument("--check", action="store_true", help="Only check source paths")
    parser.add_argument("--link-local", type=Path, help="Link an existing checkout; valid with one backend only")
    args = parser.parse_args()
    if args.link_local and args.backend == "all":
        parser.error("--link-local requires one backend")
    names = tuple(SOURCES) if args.backend == "all" else (args.backend,)
    for name in names:
        path = source_dir(name)
        if source_ready(name):
            print(f"{name}: ready ({path})")
            continue
        if path.exists() or path.is_symlink():
            if not args.check and not path.is_symlink() and (path / ".git").exists() and name in REVISIONS:
                checkout_revision(name, path)
                if source_ready(name):
                    print(f"{name}: ready ({path}; revision {REVISIONS[name]})")
                    continue
            raise SystemExit(
                f"Incompatible source checkout at {path}. "
                f"Expected entry point: {SOURCES[name][2]}. "
                "Rerun without --check to select the supported revision."
            )
        if args.check:
            print(f"{name}: missing ({path})")
            continue
        if args.link_local:
            target = args.link_local.expanduser().resolve()
            if not (target / SOURCES[name][2]).is_file():
                parser.error(f"Existing checkout lacks {SOURCES[name][2]}: {target}")
            path.symlink_to(target, target_is_directory=True)
        else:
            command = ["git", "clone", "--filter=blob:none", SOURCES[name][1], str(path)]
            subprocess.run(command, check=True)
            checkout_revision(name, path)
            if name == "robotwin":
                subprocess.run(["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], check=True)
        if not source_ready(name):
            raise RuntimeError(f"Source checkout does not contain expected entry point: {path}")
        print(f"{name}: ready ({path})")


if __name__ == "__main__":
    main()
