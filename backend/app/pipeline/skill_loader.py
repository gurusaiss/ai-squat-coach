from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

SKILL_PATH = Path(__file__).resolve().parents[3] / "skill" / "squat_skill.yaml"


@lru_cache(maxsize=1)
def load_skill() -> dict:
    with open(SKILL_PATH, "r") as f:
        return yaml.safe_load(f)


def criteria_by_id() -> dict:
    skill = load_skill()
    return {c["id"]: c for c in skill["criteria"]}


def not_assessable_list() -> list:
    return load_skill()["not_assessable_from_side_view"]
