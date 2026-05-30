"""Isaac Sim environment backdrops — warehouse / room / office USDs.

Tier 2 of `plans/isaac-asset-diversification.md`. Each clip samples one
backdrop uniformly from (discovered envs + "none"). When a real env is
picked, the USD brings its own authored floor and lights, so the caller
skips the procedural `GroundPlane` and HDRI/fallback distant light to
avoid Z-fighting on the floor and double-bright scenes. When "none" is
picked, behavior matches the pre-Tier-2 pipeline exactly.

Discovery follows the same pattern as `asset_library.py`: every candidate
path is `omni.client.stat`-ed at import time, and any path missing on the
local mirror is dropped with a warning. So a partial sync degrades — it
doesn't crash.

Caveat (plan §7 q4): each env USD authors its floor at world `z=0` and
that assumption is shared with `ClipConfig.init_height_range`. If a future
env places its floor elsewhere, bodies will fall through and we'll need a
per-env body-anchor override in `CANDIDATE_ENVIRONMENTS`.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import omni.client
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage


logger = logging.getLogger(__name__)


# (name, relative_path) — uniform sampling for v0.1. Add weights here when
# evidence shows one env dominates the visual distribution.
CANDIDATE_ENVIRONMENTS = [
    ("warehouse",                "/Isaac/Environments/Simple_Warehouse/warehouse.usd"),
    ("warehouse_full",           "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"),
    ("warehouse_with_forklifts", "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"),
    ("warehouse_shelves",        "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"),
    ("simple_room",              "/Isaac/Environments/Simple_Room/simple_room.usd"),
    ("office",                   "/Isaac/Environments/Office/office.usd"),
]

ENV_PRIM_PATH = "/World/environment"
NONE_NAME = "none"


def _resolve(assets_root: Optional[str], candidates) -> list:
    """Validate each candidate path against the live asset server.

    Returns `[(name, url), ...]` for paths that resolved; logs a warning
    per dropped path.
    """
    if assets_root is None:
        logger.warning(
            "environments: get_assets_root_path() returned None — Isaac asset "
            "server unreachable. Environment backdrops disabled."
        )
        return []
    valid = []
    for name, rel in candidates:
        url = f"{assets_root}{rel}"
        result, _ = omni.client.stat(url)
        if result == omni.client.Result.OK:
            valid.append((name, url))
        else:
            logger.warning("environments: dropping %s (not found at %s)", name, url)
    return valid


@dataclass(frozen=True)
class EnvironmentLibrary:
    """Validated environment backdrops with their resolved URLs."""

    assets_root: Optional[str] = None
    environments: list = field(default_factory=list)  # [(name, url), ...]

    @property
    def is_empty(self) -> bool:
        return not self.environments

    @property
    def names(self) -> list:
        return [n for n, _ in self.environments]


def discover() -> EnvironmentLibrary:
    """Eager discovery at import time — mirrors `asset_library.discover`."""
    assets_root = get_assets_root_path(skip_check=True)
    envs = _resolve(assets_root, CANDIDATE_ENVIRONMENTS)
    logger.info("environments: discovered %d backdrops", len(envs))
    return EnvironmentLibrary(assets_root=assets_root, environments=envs)


library = discover()


def setup_random_environment(rng, force_name: Optional[str] = None) -> Optional[dict]:
    """Sample one env uniformly from (discovered + "none") and reference it.

    `force_name` pins the choice — pass `"none"` for procedural ground or any
    name in `library.names` to load that specific env. Passing a name not in
    `library.names + ["none"]` raises ValueError. None (default) = random.

    Returns a dict for `meta.json["scene"]["environment"]` when a real env is
    loaded, or `None` when "none" was sampled (or the library is empty).

    Caller contract: a non-None return means the env brings its own floor +
    lights, so the caller MUST skip the procedural `GroundPlane` and the
    HDRI/fallback distant light to avoid Z-fighting and double-bright
    scenes (plan §7 q3).
    """
    if force_name is not None:
        if force_name == NONE_NAME:
            return None
        for name, url in library.environments:
            if name == force_name:
                add_reference_to_stage(usd_path=url, prim_path=ENV_PRIM_PATH)
                return {"name": name, "url": url}
        raise ValueError(
            f"environment {force_name!r} not in discovered set "
            f"{library.names + [NONE_NAME]!r}"
        )

    if library.is_empty:
        return None

    # Uniform over (discovered envs + "none") — "none" is one option among
    # equals, giving the encoder a mix of richly-textured and bare scenes.
    n_real = len(library.environments)
    idx = int(rng.integers(0, n_real + 1))
    if idx == n_real:
        return None

    name, url = library.environments[idx]
    add_reference_to_stage(usd_path=url, prim_path=ENV_PRIM_PATH)
    return {"name": name, "url": url}
