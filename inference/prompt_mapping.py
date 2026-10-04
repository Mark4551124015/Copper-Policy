"""Exact plaintext prompt lookup; task identities are used only offline."""
from functools import lru_cache
import json
import os
from pathlib import Path

DEFAULT_PROMPT_MAP = Path(__file__).with_name("libero_prompt_mapping.json")


@lru_cache(maxsize=8)
def load_prompt_mapping(path):
    path = Path(path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 2:
        raise ValueError(f"Unsupported prompt mapping version: {path}")
    return payload


def resolve_training_prompt(benchmark_prompt, *, task_name="", backend=None, mapping_path=None):
    """Look up the exact benchmark text, preserving language perturbations.

    Task names only identify language perturbations to bypass substitution.
    Neither task indices nor names select a mapping entry.
    """
    if "_language_" in task_name:
        return benchmark_prompt
    backend = backend or os.environ.get("COPPER_LIBERO_BACKEND")
    mapping = load_prompt_mapping(str(mapping_path or DEFAULT_PROMPT_MAP))
    for candidate in (backend,) if backend else ("libero", "libero-plus"):
        text = mapping.get("prompt_to_training", {}).get(candidate, {}).get(benchmark_prompt)
        if isinstance(text, str) and text.strip():
            return text
    return benchmark_prompt
