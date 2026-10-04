"""Apply the two dependency fixes from RoboTwin's pinned installation script."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
environment = ROOT / ".venv"
if Path(sys.prefix).resolve() != environment.resolve():
    raise SystemExit("Run this script in the project's uv .venv")

for module, relative, old, new in (
    ("sapien", "wrapper/urdf_loader.py", 'with open(urdf_file, "r") as f:',
     'with open(urdf_file, "r", encoding="utf-8") as f:'),
    ("sapien", "wrapper/urdf_loader.py", 'with open(srdf_file, "r") as f:',
     'with open(srdf_file, "r", encoding="utf-8") as f:'),
    ("mplib", "planner.py", 'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
     'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:'),
):
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"Missing package: {module}")
    target = Path(next(iter(spec.submodule_search_locations))) / relative
    if not target.resolve().is_relative_to(environment.resolve()):
        raise SystemExit(f"Refusing to patch a package outside the project environment: {target}")
    text = target.read_text()
    if old in text:
        # uv may hardlink installed files to its shared cache. Replace the
        # directory entry atomically so other environments remain unchanged.
        temporary = target.with_name(target.name + ".copper-patch")
        temporary.write_text(text.replace(old, new))
        temporary.replace(target)
        print(f"Patched {module}/{relative}")
    elif new in text:
        print(f"Already patched: {module}/{relative}")
    else:
        raise SystemExit(f"Expected upstream patch location missing: {target}")
