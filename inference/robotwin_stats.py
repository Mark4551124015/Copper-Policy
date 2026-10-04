from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np



ROBOTWIN_QPOS_ACTION_MODE_ALIASES = {
    "abs": "abs_qpos",
    "absolute": "abs_qpos",
    "absolute_qpos": "abs_qpos",
    "abs_qpos": "abs_qpos",
    "qpos": "abs_qpos",
    "joint": "abs_qpos",
    "rel": "rel_qpos",
    "relative": "rel_qpos",
    "relative_qpos": "rel_qpos",
    "rel_qpos": "rel_qpos",
    "delta": "rel_qpos",
    "delta_qpos": "rel_qpos",
}

ACTION_SPACE_STATS_CONTAINERS = (
    "action_spaces",
    "action_space_stats",
)




def normalize_robotwin_qpos_action_mode(value: Any) -> str:
    key = str(value or "abs_qpos").strip().lower().replace("-", "_")
    if key not in ROBOTWIN_QPOS_ACTION_MODE_ALIASES:
        raise ValueError(
            f"Unsupported robotwin_qpos_action_mode={value!r}. "
            f"Expected one of {sorted(ROBOTWIN_QPOS_ACTION_MODE_ALIASES)}"
        )
    return ROBOTWIN_QPOS_ACTION_MODE_ALIASES[key]


def _stats_leaf_has_norm_params(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        ("mean" in value and "std" in value)
        or ("min" in value and "max" in value)
        or ("q01" in value and "q99" in value)
    )


def _action_space_key_candidates(mode: str, action_horizon: int | None = None) -> tuple[str, ...]:
    keys = [mode]
    if action_horizon is not None:
        horizon = int(action_horizon)
        keys.extend(
            [
                f"{mode}_h{horizon}",
                f"{mode}_horizon{horizon}",
                f"{mode}_{horizon}",
            ]
        )
    return tuple(dict.fromkeys(keys))


def validate_action_space_stats_horizon(
    stats: Mapping[str, Any],
    *,
    mode: str,
    action_horizon: int | None,
) -> None:
    if action_horizon is None:
        return
    expected = int(action_horizon)
    for key in ("horizon", "action_horizon"):
        if key not in stats:
            continue
        actual = int(np.asarray(stats[key]).reshape(-1)[0])
        if actual != expected:
            raise ValueError(
                f"RobotWin {mode} action stats were computed for horizon={actual}, "
                f"but config action_horizon={expected}."
            )


def select_action_space_stats(
    stats: Mapping[str, Any],
    *,
    qpos_action_mode: Any,
    action_horizon: int | None,
    allow_top_level: bool = True,
) -> Mapping[str, Any]:
    mode = normalize_robotwin_qpos_action_mode(qpos_action_mode)

    for container_key in ACTION_SPACE_STATS_CONTAINERS:
        container = stats.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for action_key in _action_space_key_candidates(mode, action_horizon):
            value = container.get(action_key)
            if _stats_leaf_has_norm_params(value):
                validate_action_space_stats_horizon(value, mode=mode, action_horizon=action_horizon)
                return value

    for action_key in _action_space_key_candidates(mode, action_horizon):
        value = stats.get(action_key)
        if _stats_leaf_has_norm_params(value):
            validate_action_space_stats_horizon(value, mode=mode, action_horizon=action_horizon)
            return value

    if allow_top_level and _stats_leaf_has_norm_params(stats):
        validate_action_space_stats_horizon(stats, mode=mode, action_horizon=action_horizon)
        return stats

    raise KeyError(
        f"Could not find RobotWin action stats for mode={mode!r}, horizon={action_horizon!r}. "
        f"Available top-level keys: {sorted(stats.keys())}"
    )




def _lookup_feature_stats(stats: Mapping[str, Any], feature_key: str) -> Mapping[str, Any]:
    if "." in feature_key:
        cur: Any = stats
        for part in feature_key.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, Mapping):
            return cur

    if feature_key in stats and isinstance(stats[feature_key], Mapping):
        return stats[feature_key]

    fallback_paths = (
        ("state", "default"),
        ("observation", "state"),
    )
    for path in fallback_paths:
        cur: Any = stats
        for part in path:
            if not isinstance(cur, Mapping) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, Mapping):
            return cur

    raise KeyError(
        f"Could not find stats for {feature_key!r}. "
        f"Top-level keys: {sorted(stats.keys())}"
    )

























def resolve_robotwin_feature_stats(dataset_path, feature_key, dataset_variant=None, dataset_repo_allowlist=None):
    """Read existing normalization metadata for legacy checkpoint compatibility."""
    path = Path(dataset_path) / "global_stats.json"
    return dict(_lookup_feature_stats(json.loads(path.read_text()), feature_key))


def resolve_robotwin_action_space_stats(dataset_path, *, qpos_action_mode, action_horizon,
                                       dataset_variant=None, dataset_repo_allowlist=None):
    """Read existing action metadata; no dataset processing is performed."""
    payload = json.loads((Path(dataset_path) / "global_stats.json").read_text())
    mode = normalize_robotwin_qpos_action_mode(qpos_action_mode)
    try:
        return dict(select_action_space_stats(payload, qpos_action_mode=mode,
                                             action_horizon=action_horizon, allow_top_level=False))
    except KeyError:
        if mode != "abs_qpos":
            raise
        return dict(_lookup_feature_stats(payload, "action"))
