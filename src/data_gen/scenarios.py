"""Scenario builders for RigidProbe v0.1.

Step 5.2 of `plans/scene-diversification.md`. Each archetype is a function:

    build_<name>(world, rng, config) -> (bodies, body_specs, scene_specs, initial_velocities)

where:
    - `bodies`              : list of dynamic body handles, in spawn order
    - `body_specs`          : list of dicts for meta.json "bodies" (one per body)
    - `scene_specs`         : dict for meta.json "scene":
                              {"static_assets": [...], "scenario_specs": {...}}
    - `initial_velocities`  : list of (lin_vel_world, ang_vel_world) numpy
                              arrays aligned with `bodies`. Caller applies
                              these AFTER world.reset() — Isaac discards
                              velocities that were set before reset.

Static geometry (ramps, walls, platforms, bowls) is treated as scenery: it
appears in RGB and segmentation but is not added to `bodies` or
`body_specs`. Its parametric description goes into `scene_specs.static_assets`
so the scene is reproducible from meta.json alone.

The scenarios.py module is dead code until step 5.3 wires
`collect_clip.build_scene` to dispatch into these builders.
"""
import numpy as np

import asset_library
import asset_loader
import body_factory
import seg_labels
import static_geometry


# --- shared helpers ---------------------------------------------------------

def _sample_n_bodies(rng, config):
    n_lo, n_hi = config.n_bodies_range
    return int(rng.integers(n_lo, n_hi + 1))


def _sample_angular_velocity(rng, config):
    return rng.uniform(*config.init_ang_vel_range, size=3)


def _zero_linear_velocity():
    return np.zeros(3, dtype=np.float64)


def _placement_failed(scenario_name: str, n_bodies: int, config):
    return RuntimeError(
        f"{scenario_name}: could not place {n_bodies} bodies without overlap "
        f"within {config.pose_rejection_max_attempts} attempts per body. "
        "Re-seed or widen the scenario's spawn footprint."
    )


# --- drop -------------------------------------------------------------------

def build_drop(world, rng, config):
    """N bodies airborne over empty ground. Drops from rest with random spin.

    Reference scenario — the v0 baseline. Static geometry is empty (just the
    default ground plane added by the caller).
    """
    n = _sample_n_bodies(rng, config)

    bodies, body_specs = body_factory.build_random_bodies(world, rng, config, n)
    if bodies is None:
        raise _placement_failed("drop", n, config)

    initial_velocities = [
        (_zero_linear_velocity(), _sample_angular_velocity(rng, config))
        for _ in bodies
    ]
    scene_specs = {"static_assets": [], "scenario_specs": {}}
    return bodies, body_specs, scene_specs, initial_velocities


# --- roll -------------------------------------------------------------------

ROLL_RAMP_LENGTH = 0.6
ROLL_RAMP_WIDTH = 0.6
ROLL_ANGLE_RANGE_DEG = (15.0, 30.0)
ROLL_SPAWN_XY_RANGE = (-0.2, 0.2)
ROLL_SPAWN_HEIGHT_RANGE = (0.6, 1.0)


def build_roll(world, rng, config):
    """N bodies dropped above a solid wedge ramp. They land and roll/slide.

    The wedge's toe (low corner) sits at y ≈ -0.3 so the footprint covers
    y ∈ (-0.3, +0.3) and x ∈ (-0.3, +0.3). Bodies spawn in a smaller
    airborne box centered above the wedge.
    """
    angle_deg = float(rng.uniform(*ROLL_ANGLE_RANGE_DEG))
    ramp_base = np.array([0.0, -ROLL_RAMP_LENGTH * 0.5, 0.0])

    static_geometry.make_ramp(
        world, prim_path="/World/ramp", name="ramp",
        base_position=ramp_base, angle_deg=angle_deg,
        length=ROLL_RAMP_LENGTH, width=ROLL_RAMP_WIDTH,
    )

    n = _sample_n_bodies(rng, config)
    bodies, body_specs = body_factory.build_random_bodies(
        world, rng, config, n,
        xy_range=ROLL_SPAWN_XY_RANGE,
        height_range=ROLL_SPAWN_HEIGHT_RANGE,
    )
    if bodies is None:
        raise _placement_failed("roll", n, config)

    initial_velocities = [
        (_zero_linear_velocity(), _sample_angular_velocity(rng, config))
        for _ in bodies
    ]
    scene_specs = {
        "static_assets": [{
            "type": "ramp",
            "shape": "wedge",
            "prim_path": "/World/ramp",
            "base_position_world": ramp_base.tolist(),
            "angle_deg": angle_deg,
            "length": ROLL_RAMP_LENGTH,
            "width": ROLL_RAMP_WIDTH,
            "height": ROLL_RAMP_LENGTH * float(np.tan(np.deg2rad(angle_deg))),
        }],
        "scenario_specs": {"ramp_angle_deg": angle_deg},
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- throw ------------------------------------------------------------------

THROW_SPEED_RANGE = (0.5, 2.0)        # m/s, horizontal magnitude
THROW_UP_RANGE = (0.0, 0.5)           # m/s, +Z component


def _sample_throw_velocity(rng):
    """Random horizontal direction + small upward kick → arcing trajectory."""
    speed = rng.uniform(*THROW_SPEED_RANGE)
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    up = rng.uniform(*THROW_UP_RANGE)
    return np.array([
        speed * np.cos(azimuth),
        speed * np.sin(azimuth),
        up,
    ])


def build_throw(world, rng, config):
    """N bodies airborne with non-zero initial linear velocity.

    No static geometry — bodies fly through empty space and land on the
    ground at varying positions and times. This kills the "everything is
    pure gravity" shortcut that `drop` admits.
    """
    n = _sample_n_bodies(rng, config)

    bodies, body_specs = body_factory.build_random_bodies(world, rng, config, n)
    if bodies is None:
        raise _placement_failed("throw", n, config)

    initial_velocities = [
        (_sample_throw_velocity(rng), _sample_angular_velocity(rng, config))
        for _ in bodies
    ]
    scene_specs = {
        "static_assets": [],
        "scenario_specs": {
            "throw_speed_range": list(THROW_SPEED_RANGE),
            "throw_up_range": list(THROW_UP_RANGE),
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- drop_on_platform -------------------------------------------------------

PLATFORM_LENGTH = 0.4
PLATFORM_WIDTH = 0.4
PLATFORM_HEIGHT = 0.10
PLATFORM_SPAWN_XY_RANGE = (-0.25, 0.25)     # wider than platform half-width=0.2
PLATFORM_SPAWN_HEIGHT_RANGE = (0.5, 1.0)


def build_drop_on_platform(world, rng, config):
    """N bodies dropped onto a low platform. They impact and bounce/roll off edges.

    Spawn footprint extends past the platform's footprint so a fraction of
    bodies hit the edge or miss entirely — produces bounce + spin variety
    that a flat ground impact does not.
    """
    static_geometry.make_platform(
        world, prim_path="/World/platform", name="platform",
        base_position=np.zeros(3),
        length=PLATFORM_LENGTH, width=PLATFORM_WIDTH, height=PLATFORM_HEIGHT,
    )

    n = _sample_n_bodies(rng, config)
    bodies, body_specs = body_factory.build_random_bodies(
        world, rng, config, n,
        xy_range=PLATFORM_SPAWN_XY_RANGE,
        height_range=PLATFORM_SPAWN_HEIGHT_RANGE,
    )
    if bodies is None:
        raise _placement_failed("drop_on_platform", n, config)

    initial_velocities = [
        (_zero_linear_velocity(), _sample_angular_velocity(rng, config))
        for _ in bodies
    ]
    scene_specs = {
        "static_assets": [{
            "type": "platform",
            "prim_path": "/World/platform",
            "base_position_world": [0.0, 0.0, 0.0],
            "length": PLATFORM_LENGTH,
            "width": PLATFORM_WIDTH,
            "height": PLATFORM_HEIGHT,
        }],
        "scenario_specs": {},
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- settle -----------------------------------------------------------------

BOWL_INNER_LENGTH = 0.9
BOWL_INNER_WIDTH = 0.9
BOWL_HEIGHT = 0.25
BOWL_WALL_THICKNESS = 0.02
SETTLE_SPAWN_XY_RANGE = (-0.27, 0.27)        # inside bowl, ~0.05 m wall clearance for max-radius bodies
SETTLE_SPAWN_HEIGHT_RANGE = (0.5, 1.5)       # 1.0 m vertical headroom for rejection sampling


def build_settle(world, rng, config):
    """N bodies dropped into an open bowl. They settle and pile up.

    Bowl interior is 0.9 × 0.9 — sized so rejection-sampling can place up to
    config.n_bodies_range's upper bound (8) without exhausting attempts.
    Smaller interiors fail for runs that happen to sample many large cuboids.
    """
    static_geometry.make_bowl(
        world, prim_path="/World/bowl", name="bowl",
        base_position=np.zeros(3),
        inner_length=BOWL_INNER_LENGTH, inner_width=BOWL_INNER_WIDTH,
        height=BOWL_HEIGHT, wall_thickness=BOWL_WALL_THICKNESS,
    )

    n = _sample_n_bodies(rng, config)
    bodies, body_specs = body_factory.build_random_bodies(
        world, rng, config, n,
        xy_range=SETTLE_SPAWN_XY_RANGE,
        height_range=SETTLE_SPAWN_HEIGHT_RANGE,
    )
    if bodies is None:
        raise _placement_failed("settle", n, config)

    initial_velocities = [
        (_zero_linear_velocity(), _sample_angular_velocity(rng, config))
        for _ in bodies
    ]
    scene_specs = {
        "static_assets": [{
            "type": "bowl",
            "prim_path": "/World/bowl",
            "base_position_world": [0.0, 0.0, 0.0],
            "inner_length": BOWL_INNER_LENGTH,
            "inner_width": BOWL_INNER_WIDTH,
            "height": BOWL_HEIGHT,
            "wall_thickness": BOWL_WALL_THICKNESS,
        }],
        "scenario_specs": {},
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- push -------------------------------------------------------------------

PUSH_SPEED_RANGE = (0.3, 1.5)                # m/s, pure horizontal
PUSH_SPAWN_XY_RANGE = (-0.3, 0.3)
PUSH_SPAWN_HEIGHT_RANGE = (0.1, 0.2)         # near-ground; small drop before sliding


def _sample_push_velocity(rng):
    """Pure horizontal velocity — no +Z kick (that's what makes this not throw)."""
    speed = rng.uniform(*PUSH_SPEED_RANGE)
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([speed * np.cos(azimuth), speed * np.sin(azimuth), 0.0])


def build_push(world, rng, config):
    """N bodies near the ground with horizontal initial velocity. They slide and roll.

    No static geometry. Differs from `throw` because there's no upward
    component: bodies stay near the ground and exhibit friction-mediated
    sliding + rolling rather than parabolic flight.
    """
    n = _sample_n_bodies(rng, config)
    bodies, body_specs = body_factory.build_random_bodies(
        world, rng, config, n,
        xy_range=PUSH_SPAWN_XY_RANGE,
        height_range=PUSH_SPAWN_HEIGHT_RANGE,
    )
    if bodies is None:
        raise _placement_failed("push", n, config)

    initial_velocities = [
        (_sample_push_velocity(rng), _sample_angular_velocity(rng, config))
        for _ in bodies
    ]
    scene_specs = {
        "static_assets": [],
        "scenario_specs": {"push_speed_range": list(PUSH_SPEED_RANGE)},
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- cluttered --------------------------------------------------------------

CLUTTERED_N_STATIC_RANGE = (2, 5)                # inclusive
CLUTTERED_STATIC_XY_RANGE = (-0.5, 0.5)          # wider than the body footprint
CLUTTERED_STATIC_MIN_SEPARATION = 0.35           # m between static-asset centers (XY)
CLUTTERED_DYNAMIC_XY_RANGE = (-0.4, 0.4)
CLUTTERED_DYNAMIC_HEIGHT_RANGE = (1.0, 1.5)      # well above typical asset heights (~0.5 m)


def _sample_static_xy(rng, existing, config):
    """Rejection-sample one XY point at least MIN_SEPARATION from every existing one."""
    xy_lo, xy_hi = CLUTTERED_STATIC_XY_RANGE
    for _ in range(config.pose_rejection_max_attempts):
        x = float(rng.uniform(xy_lo, xy_hi))
        y = float(rng.uniform(xy_lo, xy_hi))
        if all(np.hypot(x - px, y - py) > CLUTTERED_STATIC_MIN_SEPARATION for px, py in existing):
            return x, y
    return None


def build_cluttered(world, rng, config):
    """Static Isaac props as scaffolding + N primitive bodies dropped above.

    Dynamic bodies stay primitives for v0.1 — mixing asset bodies into the
    drop set is a later iteration (per plan §3 the cluttered row permits
    "primitive or asset dynamic bodies", and primitive-only is the simpler
    half). Static assets get a wide XY footprint with minimum separation
    between centers so they don't visually overlap; dynamic bodies spawn
    high enough (1.0-1.5 m) that the typical ≤0.5 m asset heights don't
    intersect them at t=0.

    Raises if the asset library is empty — the scenario can't run without
    static props, so the caller should remove "cluttered" from
    config.scenarios when the asset root is unreachable.
    """
    library = asset_library.library
    if not library.static_props:
        raise RuntimeError(
            "cluttered: asset_library.static_props is empty — Isaac asset root "
            "unreachable or contains none of the expected props. Either remove "
            "'cluttered' from config.scenarios or fix the asset root."
        )

    n_static = int(rng.integers(CLUTTERED_N_STATIC_RANGE[0], CLUTTERED_N_STATIC_RANGE[1] + 1))
    static_xy: list = []
    static_specs = []
    for i in range(n_static):
        xy = _sample_static_xy(rng, static_xy, config)
        if xy is None:
            raise _placement_failed("cluttered (static)", n_static, config)
        static_xy.append(xy)

        url = library.sample_static_prop(rng)
        rotation_deg = float(rng.uniform(0.0, 360.0))
        prim_path = f"/World/static_{i:02d}"
        asset_loader.load_static_asset(
            world,
            usd_url=url,
            prim_path=prim_path,
            name=f"static_{i:02d}",
            position=(xy[0], xy[1], 0.0),
            semantic_label=seg_labels.SCENERY_LABEL,
            rotation_z_deg=rotation_deg,
        )
        static_specs.append({
            "type": "asset",
            "prim_path": prim_path,
            "asset_url": url,
            "position_world": [xy[0], xy[1], 0.0],
            "rotation_z_deg": rotation_deg,
        })

    n = _sample_n_bodies(rng, config)
    bodies, body_specs = body_factory.build_random_bodies(
        world, rng, config, n,
        xy_range=CLUTTERED_DYNAMIC_XY_RANGE,
        height_range=CLUTTERED_DYNAMIC_HEIGHT_RANGE,
    )
    if bodies is None:
        raise _placement_failed("cluttered", n, config)

    initial_velocities = [
        (_zero_linear_velocity(), _sample_angular_velocity(rng, config))
        for _ in bodies
    ]
    scene_specs = {
        "static_assets": static_specs,
        "scenario_specs": {
            "n_static_assets": n_static,
            "static_xy_range": list(CLUTTERED_STATIC_XY_RANGE),
            "dynamic_height_range": list(CLUTTERED_DYNAMIC_HEIGHT_RANGE),
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- registry ---------------------------------------------------------------

BUILDERS = {
    "drop": build_drop,
    "roll": build_roll,
    "throw": build_throw,
    "drop_on_platform": build_drop_on_platform,
    "settle": build_settle,
    "push": build_push,
    "cluttered": build_cluttered,
}


def build(scenario_name: str, world, rng, config):
    """Dispatch by name. Raises KeyError if the scenario is unknown."""
    return BUILDERS[scenario_name](world, rng, config)
