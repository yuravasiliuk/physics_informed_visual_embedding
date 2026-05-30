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

import numpy as np
import omni.client
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom


logger = logging.getLogger(__name__)


# Candidates relative to get_assets_root_path() (i.e. the URL is
# f"{assets_root}{rel_path}"). Exact paths vary across Isaac versions —
# `_resolve` validates each one and prunes the list to whatever actually
# exists on the configured asset server.
#
# Tier 1 of `plans/isaac-asset-diversification.md`: expanded from the
# original 5 paths (3 of which resolved) to 16 paths across pallets, KLT
# bins, cabinets, tables/mounts, and industrial scenery, with per-path
# sampling weights so pallets stop visually dominating every clip. Each
# entry is `(rel_path, weight)` — higher weight = sampled more often.
# Pallets are split into 4 variants at 0.4 each (effective combined
# weight 1.6) instead of one entry at full weight; tables and KLT bins
# stay at 1.0; cabinets / forklifts / dollies sit between.
CANDIDATE_STATIC_PROPS = [
    # Pallet family — 4 variants, each weighted low so they don't dominate
    ("/Isaac/Props/Pallet/pallet.usd",                       0.4),
    ("/Isaac/Props/Pallet/o3dyn_pallet.usd",                 0.4),
    ("/Isaac/Props/Pallet/pallet_holder.usd",                0.4),
    ("/Isaac/Props/Pallet/pallet_holder_short.usd",          0.4),
    # KLT bin — useful as a low container body
    ("/Isaac/Props/KLT_Bin/small_KLT.usd",                   1.0),
    # Cabinet
    ("/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd", 0.8),
    # Tables and mounts
    ("/Isaac/Props/Mounts/table.usd",                        1.0),
    ("/Isaac/Props/Mounts/textured_table.usd",               1.0),
    ("/Isaac/Props/Mounts/thor_table.usd",                   1.0),
    ("/Isaac/Props/Mounts/ur10_mount.usd",                   0.7),
    ("/Isaac/Props/PackingTable/packing_table.usd",          1.0),
    # Industrial scenery
    ("/Isaac/Props/Dolly/dolly.usd",                         0.7),
    ("/Isaac/Props/Dolly/dolly_physics.usd",                 0.7),
    ("/Isaac/Props/Forklift/forklift.usd",                   0.6),
    ("/Isaac/Props/Sortbot_Housing/sortbot_housing.usd",     0.6),
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

    Accepts either bare relative-path strings (legacy form, uniform weight)
    or `(rel_path, weight)` tuples. Returns a list of `(url, weight)` pairs
    for paths that resolved. Logs a warning per dropped path.
    """
    if assets_root is None:
        logger.warning(
            "asset_library: get_assets_root_path() returned None — Isaac asset "
            "server unreachable. Asset-based scenarios will be unavailable."
        )
        return []
    valid = []
    for entry in candidates:
        if isinstance(entry, tuple):
            rel, weight = entry
        else:
            rel, weight = entry, 1.0
        url = f"{assets_root}{rel}"
        result, _ = omni.client.stat(url)
        if result == omni.client.Result.OK:
            valid.append((url, float(weight)))
        else:
            logger.warning("asset_library: dropping %s (not found at %s)", rel, url)
    return valid


_FOOTPRINT_PROBE_PATH = "/AssetFootprintProbe"


def _measure_footprint(stage, url: str) -> Optional[tuple]:
    """Reference `url` at a probe path, read its world XY extent, then unreference.

    Returns `(extent_x, extent_y)` in meters — the axis-aligned XY extent of the
    asset's default-purpose geometry, with the asset placed at its authored
    origin (no rotation, no scale override). Caller uses this to compute a
    circumscribed-circle radius for placement rejection in `build_cluttered`.

    Returns None if the asset fails to reference or has a degenerate bbox
    (zero extent in either axis). Includes both `default` and `render` purposes
    so assets that author their geometry under render purpose (some Isaac
    props do) still measure correctly.

    Guarded by try/finally — a failing measurement never leaves the probe
    prim on the stage to pollute later loads.
    """
    if stage.GetPrimAtPath(_FOOTPRINT_PROBE_PATH).IsValid():
        stage.RemovePrim(_FOOTPRINT_PROBE_PATH)
    try:
        add_reference_to_stage(usd_path=url, prim_path=_FOOTPRINT_PROBE_PATH)
        prim = stage.GetPrimAtPath(_FOOTPRINT_PROBE_PATH)
        if not prim.IsValid():
            return None
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        aabb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        size = aabb.GetSize()
        extent_x = float(size[0])
        extent_y = float(size[1])
        if extent_x <= 0.0 or extent_y <= 0.0:
            return None
        return (extent_x, extent_y)
    finally:
        if stage.GetPrimAtPath(_FOOTPRINT_PROBE_PATH).IsValid():
            stage.RemovePrim(_FOOTPRINT_PROBE_PATH)


def _measure_footprints(static_props_weighted: list) -> list:
    """Augment a `[(url, weight), ...]` list with measured footprints.

    Returns `[(url, weight, (extent_x, extent_y)), ...]`. Drops entries whose
    bbox measurement fails (logged) — better to lose one asset than crash the
    whole scenario at clip time.
    """
    stage = omni.usd.get_context().get_stage()
    out = []
    for url, weight in static_props_weighted:
        fp = _measure_footprint(stage, url)
        if fp is None:
            logger.warning("asset_library: dropping %s (footprint measurement failed)", url)
            continue
        out.append((url, weight, fp))
    return out


def _weighted_choice(rng, weighted_urls):
    """Sample one URL from a list of `(url, weight)` pairs."""
    if not weighted_urls:
        return None
    urls = [u for u, _ in weighted_urls]
    weights = np.array([w for _, w in weighted_urls], dtype=np.float64)
    probs = weights / weights.sum()
    return str(rng.choice(urls, p=probs))


@dataclass(frozen=True)
class AssetLibrary:
    """Validated, categorized USD asset URLs.

    `static_props` holds `(url, weight, footprint_xy)` triples — the footprint
    is the XY extent measured at discovery time, used by `build_cluttered`
    for radius-based rejection sampling so large props (dollies, forklifts)
    don't intersect each other. `static_robots` and `dynamic_props` still
    hold bare `(url, weight)` pairs; Tier 3 will extend them similarly.
    """

    assets_root: Optional[str] = None
    static_props: list = field(default_factory=list)   # [(url, weight, footprint_xy)]
    static_robots: list = field(default_factory=list)  # [(url, weight)]
    dynamic_props: list = field(default_factory=list)  # [(url, weight)]

    @property
    def is_empty(self) -> bool:
        return not (self.static_props or self.static_robots or self.dynamic_props)

    def static_prop_urls(self) -> list:
        return [u for u, _, _ in self.static_props]

    def static_robot_urls(self) -> list:
        return [u for u, _ in self.static_robots]

    def dynamic_prop_urls(self) -> list:
        return [u for u, _ in self.dynamic_props]

    def sample_static_prop(self, rng) -> Optional[tuple]:
        """Returns `(url, footprint_xy)` or None — footprint feeds rejection sampling."""
        if not self.static_props:
            return None
        urls = [u for u, _, _ in self.static_props]
        weights = np.array([w for _, w, _ in self.static_props], dtype=np.float64)
        footprints = [fp for _, _, fp in self.static_props]
        probs = weights / weights.sum()
        idx = int(rng.choice(len(urls), p=probs))
        return urls[idx], footprints[idx]

    def sample_static_robot(self, rng) -> Optional[str]:
        return _weighted_choice(rng, self.static_robots)

    def sample_dynamic_prop(self, rng) -> Optional[str]:
        return _weighted_choice(rng, self.dynamic_props)


def discover() -> AssetLibrary:
    """Eager discovery — run once at module import. See plan §9 question 2.

    `skip_check=True`: get_assets_root_path() otherwise requires *both*
    `/Isaac` and `/NVIDIA` to exist under the root. We only sync the `/Isaac`
    subtree locally, and every candidate path is verified individually below.

    Static props additionally get a one-time footprint measurement — each
    asset is briefly referenced into a probe prim, its world AABB read via
    `UsdGeom.BBoxCache`, then unreferenced. Adds ~15-30s to the first import
    after SimulationApp boot but is amortized across every clip in a multi-
    clip run.
    """
    assets_root = get_assets_root_path(skip_check=True)
    static_props_weighted = _resolve(assets_root, CANDIDATE_STATIC_PROPS)
    static_props = _measure_footprints(static_props_weighted)
    lib = AssetLibrary(
        assets_root=assets_root,
        static_props=static_props,
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
    def _dump(name, entries):
        print(f"\n{name} ({len(entries)}):")
        for entry in entries:
            if len(entry) == 3:
                u, w, fp = entry
                print(f"  [w={w:.2f}] footprint={fp[0]:.2f}x{fp[1]:.2f}m  {u}")
            elif len(entry) == 2:
                u, w = entry
                print(f"  [w={w:.2f}] {u}")
            else:
                print(f"  {entry}")

    print(f"assets_root: {library.assets_root}")
    _dump("static_props", library.static_props)
    _dump("static_robots", library.static_robots)
    _dump("dynamic_props", library.dynamic_props)
    _simulation_app.close()
