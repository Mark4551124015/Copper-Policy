"""Allow loading official LIBERO NumPy initial states with PyTorch >= 2.6."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for backend in ("LIBERO", "LIBERO-plus"):
    path = ROOT / "third_party" / backend / "libero/libero/benchmark/__init__.py"
    original = path.read_text()
    # The initial-state files come from the freshly cloned official simulator
    # repositories, and contain NumPy arrays rather than pure tensor weights.
    old = "torch.load(init_states_path)"
    new = "torch.load(init_states_path, weights_only=False)"
    if old not in original and new not in original:
        raise RuntimeError(f"Unexpected initial-state loader: {path}")
    if old in original:
        path.write_text(original.replace(old, new))
    print(f"Initial-state loader ready: {backend}")
