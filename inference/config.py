"""Inference defaults; checkpoint architecture and embedded normalization win.

Historical registry aliases are retained for existing checkpoints and launchers.
"""
import json
from pathlib import Path
import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent
VA_CONFIGS = {}
for alias, name in (("config_libero_train", "libero"),
                    ("config_robotwin_train", "robotwin"),
                    ("config_realbot_train", "realrobot")):
    cfg = EasyDict(json.loads((ROOT / "config_base.json").read_text()))
    cfg.update(json.loads((ROOT / f"{name}.json").read_text()))
    cfg.param_dtype = torch.bfloat16
    VA_CONFIGS[alias] = cfg
