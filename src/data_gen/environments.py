"""Isaac Sim environment backdrops — warehouse / room / grid-room USDs.

Tier 2 of `plans/data-gen/isaac-asset-diversification.md`. Each clip samples one
backdrop uniformly from the discovered envs. Every env USD brings its own
authored floor and lights, so the caller skips the procedural `GroundPlane`
and HDRI/fallback distant light to avoid Z-fighting on the floor and
double-bright scenes. The pre-Tier-2 procedural-ground path is no longer
sampled at random (it rendered a textureless white void); it is reachable
only via `setup_random_environment(force_name="none")` for debugging.

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
#
# office.usd is deliberately excluded: it renders its floor textureless (the
# MDL/material binding silently falls back to a default white shader on our
# local asset mirror), which is worse than the procedural ground. Warehouse +
# simple_room resolve their materials correctly, so they stay. Re-add office
# here once its floor material resolves (or ship the missing MDL).
CANDIDATE_ENVIRONMENTS = [
    ("warehouse",                "/Isaac/Environments/Simple_Warehouse/warehouse.usd"),
    ("warehouse_full",           "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"),
    ("warehouse_with_forklifts", "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"),
    ("warehouse_shelves",        "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"),
    ("simple_room",              "/Isaac/Environments/Simple_Room/simple_room.usd"),
    # Grid rooms — the lightweight / bare-scene backdrop that REPLACES the old
    # procedural "none" ground. That path rendered a textureless white floor
    # blended into a white procedural dome (the clip_00000 void). Each grid
    # room ships an authored floor + lights at z=0 and gives strong
    # translation / parallax cues without warehouse clutter, so the encoder
    # still sees sparse scenes — just not a featureless void.
    ("grid_default",             "/Isaac/Environments/Grid/default_environment.usd"),
    ("gridroom_black",           "/Isaac/Environments/Grid/gridroom_black.usd"),
    ("gridroom_curved",          "/Isaac/Environments/Grid/gridroom_curved.usd"),
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

    # Uniform over discovered envs only — no "none" in the random draw. The
    # bare-scene niche the procedural ground used to fill is now served by the
    # Grid rooms (sparse, but with an authored floor + lights instead of the
    # textureless white void of clip_00000). The procedural-ground + sky path
    # survives solely for an explicit `force_name="none"` (debugging /
    # `--environment none`); it is no longer sampled at random.
    idx = int(rng.integers(0, len(library.environments)))
    name, url = library.environments[idx]
    add_reference_to_stage(usd_path=url, prim_path=ENV_PRIM_PATH)
    return {"name": name, "url": url}
