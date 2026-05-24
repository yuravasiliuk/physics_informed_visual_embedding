"""Isaac Sim asset library — curated USD paths organized by category.

Step 5.5 of `plans/scene-diversification.md`. Provides three categorized
lists of USD asset URLs discovered via `get_assets_root_path()`. On import,
every candidate path is checked against the live Isaac asset server;
missing paths are logged and dropped from the active set rather than
crashing the run, so a partially-installed Isaac asset tree degrades
gracefully.

Categories (plan §4):
    static_props   — scaffolding (pallets, racks, mounts, KLT bins)
                     → load with `UsdPhysics.CollisionAPI` only.
    static_robots  — visually-interesting occluders (Franka, UR10, AnymalC)
                     → load with all joints fixed; no articulation in v0.1.
    dynamic_props  — drop-in replacements for primitive bodies (YCB, Blocks)
                     → load with RigidBodyAPI + CollisionAPI.

Usage (after step 5.6 lands):
    from asset_library import library
    url = library.sample_static_prop(rng)
    if url is not None:
        add_reference_to_stage(usd_path=url, prim_path="/World/asset_01")

This module is dead code until step 5.6 wires it into the scene loaders.

Standalone inspection:
    ${ISAAC_SIM}/python.sh src/data_gen/asset_library.py
"""
import logging

# When run as a script, SimulationApp must precede any omni import. As a
# regular import from collect_clip.py the SimulationApp is already up and
# this branch is skipped.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from isaacsim import SimulationApp
    _simulation_app = SimulationApp({"headless": True})
    # Override the persistent asset root setting that Kit loads from
    # ~/.local/share/ov/data/Kit/Isaac-Sim*/user.config.json (pinned to S3).
    # See collect_clip.py for the same pattern.
    import carb
    carb.settings.get_settings().set(
        "/persistent/isaac/asset_root/default", "file:///home/yura/isaac_assets"
    )

from dataclasses import dataclass, field
from typing import Optional

import omni.client
from isaacsim.storage.native import get_assets_root_path


logger = logging.getLogger(__name__)


# Candidates relative to get_assets_root_path() (i.e. the URL is
# f"{assets_root}{rel_path}"). Exact paths vary across Isaac versions —
# `_resolve` validates each one and prunes the list to whatever actually
# exists on the configured asset server.
CANDIDATE_STATIC_PROPS = [
    "/Isaac/Props/Pallet/pallet.usd",
    "/Isaac/Props/Rack/rack.usd",
    "/Isaac/Props/KLT_Bin/small_KLT.usd",
    "/Isaac/Props/Mounts/ur10_mount.usd",
    "/Isaac/Props/Mounts/Stand/stand_instanceable.usd",
]

CANDIDATE_STATIC_ROBOTS = [
    "/Isaac/Robots/Franka/franka.usd",
    "/Isaac/Robots/Franka/franka_instanceable.usd",
    "/Isaac/Robots/UR10/ur10.usd",
    "/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
    "/Isaac/Robots/AnymalC/anymal_c.usd",
    "/Isaac/Robots/ANYbotics/anymal_c.usd",
]

CANDIDATE_DYNAMIC_PROPS = [
    # YCB subset (Isaac ships these axis-aligned)
    "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    # Coloured wooden blocks
    "/Isaac/Props/Blocks/block_red.usd",
    "/Isaac/Props/Blocks/block_blue.usd",
    "/Isaac/Props/Blocks/block_green.usd",
    "/Isaac/Props/Blocks/block_yellow.usd",
    # Small instanceable items
    "/Isaac/Props/KLT_Bin/small_KLT.usd",
]


def _resolve(assets_root: Optional[str], candidates) -> list:
    """Validate each candidate path against the live asset server.

    Returns the subset that resolved successfully. Logs a warning per
    dropped path so a misconfigured Isaac install is visible at startup.
    """
    if assets_root is None:
        logger.warning(
            "asset_library: get_assets_root_path() returned None — Isaac asset "
            "server unreachable. Asset-based scenarios will be unavailable."
        )
        return []
    valid = []
    for rel in candidates:
        url = f"{assets_root}{rel}"
        result, _ = omni.client.stat(url)
        if result == omni.client.Result.OK:
            valid.append(url)
        else:
            logger.warning("asset_library: dropping %s (not found at %s)", rel, url)
    return valid


@dataclass(frozen=True)
class AssetLibrary:
    """Validated, categorized USD asset URLs."""

    assets_root: Optional[str] = None
    static_props: list = field(default_factory=list)
    static_robots: list = field(default_factory=list)
    dynamic_props: list = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.static_props or self.static_robots or self.dynamic_props)

    def sample_static_prop(self, rng) -> Optional[str]:
        return str(rng.choice(self.static_props)) if self.static_props else None

    def sample_static_robot(self, rng) -> Optional[str]:
        return str(rng.choice(self.static_robots)) if self.static_robots else None

    def sample_dynamic_prop(self, rng) -> Optional[str]:
        return str(rng.choice(self.dynamic_props)) if self.dynamic_props else None


def discover() -> AssetLibrary:
    """Eager discovery — run once at module import. See plan §9 question 2.

    `skip_check=True`: get_assets_root_path() otherwise requires *both*
    `/Isaac` and `/NVIDIA` to exist under the root. We only sync the `/Isaac`
    subtree locally, and every candidate path is verified individually below.
    """
    assets_root = get_assets_root_path(skip_check=True)
    lib = AssetLibrary(
        assets_root=assets_root,
        static_props=_resolve(assets_root, CANDIDATE_STATIC_PROPS),
        static_robots=_resolve(assets_root, CANDIDATE_STATIC_ROBOTS),
        dynamic_props=_resolve(assets_root, CANDIDATE_DYNAMIC_PROPS),
    )
    logger.info(
        "asset_library: discovered %d static_props, %d static_robots, %d dynamic_props",
        len(lib.static_props), len(lib.static_robots), len(lib.dynamic_props),
    )
    return lib


# Module-level singleton — discovered once at import time. Eager so that a
# missing or unreachable asset server surfaces at startup, not at first use.
library = discover()


if __name__ == "__main__":
    def _dump(name, urls):
        print(f"\n{name} ({len(urls)}):")
        for u in urls:
            print(f"  {u}")

    print(f"assets_root: {library.assets_root}")
    _dump("static_props", library.static_props)
    _dump("static_robots", library.static_robots)
    _dump("dynamic_props", library.dynamic_props)
    _simulation_app.close()
