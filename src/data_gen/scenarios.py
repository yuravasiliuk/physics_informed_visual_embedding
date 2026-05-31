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
import logging
import math

import numpy as np

import asset_library
import asset_loader
import body_factory
import seg_labels
import static_geometry


logger = logging.getLogger(__name__)


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

THROW_SPEED_RANGE = (0.5, 12.0)        # m/s, horizontal magnitude
THROW_UP_RANGE = (0.0, 10.5)           # m/s, +Z component


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

PUSH_SPEED_RANGE = (1.5, 3.5)                # m/s, pure horizontal
PUSH_SPAWN_XY_RANGE = (-0.3, 0.3)
PUSH_SPAWN_HEIGHT_RANGE = (0.15, 0.25)       # half-extent ≲0.075 → no spawn-time
                                             # ground penetration even for max-scale cuboids
# Speed range is intentionally higher than `throw`: friction kills sliding
# motion as `v²/(2μg)`, so with friction ~0.3-0.8 and gravity, anything below
# ~1 m/s decelerates to a stop in <100 ms and the clip looks like a drop.
# 1.5-3.5 m/s yields stopping distances ~0.2-1.2 m — visible across the camera
# frame within the 1-second clip window.


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
CLUTTERED_STATIC_RADIUS_MARGIN = 0.05            # extra gap between circumscribed circles, m
CLUTTERED_DYNAMIC_XY_RANGE = (-0.4, 0.4)
CLUTTERED_DYNAMIC_HEIGHT_RANGE = (1.0, 1.5)      # well above typical asset heights (~0.5 m)


def _footprint_radius(footprint_xy: tuple) -> float:
    """Circumscribed-circle radius of a footprint axis-aligned XY extent.

    Rotation-invariant: an asset rotated by any angle still fits inside a
    circle of this radius around its center, so we don't have to rotate
    OBBs per attempt. Conservative (rejects some valid layouts where two
    long-thin assets could nest), but cheap and good-enough for v0.1.
    """
    w, d = footprint_xy
    return 0.5 * math.hypot(w, d)


def _sample_static_placement(rng, existing, new_radius, config):
    """Rejection-sample one XY point whose disk doesn't overlap any existing disk.

    `existing` is a list of `(cx, cy, radius)` already placed. Returns `(x, y)`
    or None if no attempt within `config.pose_rejection_max_attempts` cleared
    all existing disks (accounting for radii on both sides + a small margin).
    """
    xy_lo, xy_hi = CLUTTERED_STATIC_XY_RANGE
    for _ in range(config.pose_rejection_max_attempts):
        x = float(rng.uniform(xy_lo, xy_hi))
        y = float(rng.uniform(xy_lo, xy_hi))
        ok = all(
            math.hypot(x - cx, y - cy) > new_radius + cr + CLUTTERED_STATIC_RADIUS_MARGIN
            for cx, cy, cr in existing
        )
        if ok:
            return x, y
    return None


def build_cluttered(world, rng, config):
    """Static Isaac props as scaffolding + N primitive bodies dropped above.

    Dynamic bodies stay primitives for v0.1 — mixing asset bodies into the
    drop set is a later iteration (per plan §3 the cluttered row permits
    "primitive or asset dynamic bodies", and primitive-only is the simpler
    half). Static assets are placed using radius-based rejection sampling
    against per-asset circumscribed circles (footprints measured at library
    discovery time) so a forklift and a dolly never overlap; dynamic bodies
    spawn high enough (1.0-1.5 m) that the typical ≤0.5 m asset heights
    don't intersect them at t=0.

    Assets are sampled and immediately attempted; a sample that can't fit
    against already-placed disks is *dropped* rather than retried with a
    smaller asset. This biases the final count downward when the library is
    skewed toward large props — preferable to looping over assets searching
    for a fitting one, which would over-represent small assets in busy scenes.

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

    n_static_target = int(rng.integers(
        CLUTTERED_N_STATIC_RANGE[0], CLUTTERED_N_STATIC_RANGE[1] + 1,
    ))
    placed: list = []  # [(cx, cy, radius), ...]
    static_specs = []
    for i in range(n_static_target):
        url, footprint = library.sample_static_prop(rng)
        radius = _footprint_radius(footprint)
        xy = _sample_static_placement(rng, placed, radius, config)
        if xy is None:
            logger.info(
                "cluttered: gave up placing static asset %d/%d (radius=%.2fm) "
                "after %d attempts; accepting %d statics",
                i + 1, n_static_target, radius,
                config.pose_rejection_max_attempts, len(placed),
            )
            break
        placed.append((xy[0], xy[1], radius))

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
            "footprint_xy": [float(footprint[0]), float(footprint[1])],
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
            "n_static_assets_target": n_static_target,
            "n_static_assets_placed": len(placed),
            "static_xy_range": list(CLUTTERED_STATIC_XY_RANGE),
            "static_radius_margin": CLUTTERED_STATIC_RADIUS_MARGIN,
            "dynamic_height_range": list(CLUTTERED_DYNAMIC_HEIGHT_RANGE),
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- topple -----------------------------------------------------------------
# Observability scenario (physics-diversification.md W3 / W3a). A tall, small-
# footprint body is spawned leaning *just past* its critical tip angle, so
# GRAVITY topples it onto a face (with a small kick in the tip direction). The
# toppling is the dynamic signal; with W2's offset CoM, *which* face it settles
# on is decided by that offset — making CoM strongly observable (R6 fix).
#
# W3a: the previous version set only an angular-velocity kick on an upright
# body. That kick (1.0-2.5 rad/s) is at/below the tipping energy barrier, so the
# body rocked and settled back upright — the scenario looked static. Leaning
# past theta_crit guarantees a visible topple every clip without a large kick
# that would swamp the CoM-direction signal.

TOPPLE_N_RANGE = (1, 2)                      # inclusive — keep it clean/observable
TOPPLE_FOOTPRINT_RANGE = (0.05, 0.09)        # small base side / diameter, m
TOPPLE_HEIGHT_RANGE = (0.25, 0.45)           # tall, m — aspect ratio ~3-9:1
TOPPLE_KICK_ANGVEL_RANGE = (0.3, 1.0)        # rad/s, small assist in the tip direction (gravity does the work)
TOPPLE_LEAN_MARGIN_DEG = (3.0, 8.0)          # lean this far PAST theta_crit so it always tips
TOPPLE_TIP_JITTER_DEG = 15.0                 # ± jitter on a cuboid's tip-over direction
TOPPLE_XY_RANGE = (-0.30, 0.30)
TOPPLE_GROUND_CLEARANCE = 0.005              # spawn just clear of ground (no t=0 penetration)
TOPPLE_SHAPES = ("cuboid", "cylinder")       # only shapes with a flat base to stand on


def _sample_tall_dims(rng, shape):
    """Tall, small-footprint dims so the body is marginally stable and topples.

    Layout matches sample_shape_dims:
        cuboid   -> (sx, sy, sz)  small sx,sy; tall sz
        cylinder -> (r, h)        small r; tall h
    """
    if shape == "cuboid":
        sx = float(rng.uniform(*TOPPLE_FOOTPRINT_RANGE))
        sy = float(rng.uniform(*TOPPLE_FOOTPRINT_RANGE))
        sz = float(rng.uniform(*TOPPLE_HEIGHT_RANGE))
        return (sx, sy, sz)
    # cylinder: footprint range is a diameter, so radius is half of it
    r = 0.5 * float(rng.uniform(*TOPPLE_FOOTPRINT_RANGE))
    h = float(rng.uniform(*TOPPLE_HEIGHT_RANGE))
    return (r, h)


def _topple_spacing_radius(shape, dims):
    """XY spacing radius for non-overlap: base footprint + a small lean allowance.

    A body tilted by ~theta_crit+margin shifts horizontally by only
    ~half_h·sin(lean) — a few cm, NOT a full half-height. Using the half-height
    (the original here) made two tall bodies un-placeable in the spawn box; this
    footprint-based radius keeps them apart without over-spacing."""
    if shape == "cuboid":
        foot = 0.5 * float(np.hypot(dims[0], dims[1]))
        half_h = 0.5 * dims[2]
    else:
        foot = dims[0]
        half_h = 0.5 * dims[1]
    return foot + 0.3 * half_h


def _axis_angle_rotmat(axis, angle):
    """Rotation matrix for a right-handed rotation `angle` about unit-ish `axis`
    (Rodrigues). Used to place a leaning body so its lowest vertex sits on the
    ground."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _topple_lean_pose(rng, shape, dims, center_xy):
    """Lean a tall body just past its tip angle (W3a).

    Returns (position_world, orientation_wxyz, ang_vel_world, lean_deg, tip_az_rad).
    The body is tilted by theta_crit + margin about the horizontal axis ⟂ the
    chosen tip direction φ, then raised so its lowest tilted vertex rests at
    ground+clearance (no t=0 penetration). A small angular kick `mag·axis`
    continues the tip toward φ.
    """
    he = body_factory.half_extents(shape, dims)          # (hx, hy, hz)
    half_h = float(he[2])
    if shape == "cuboid":
        # Tip over the SHORTER base edge (smallest theta_crit) so it reliably
        # falls; bias φ along that axis with jitter.
        if he[0] <= he[1]:
            d = float(he[0]); base_phi = float(rng.choice([0.0, np.pi]))
        else:
            d = float(he[1]); base_phi = float(rng.choice([0.5 * np.pi, 1.5 * np.pi]))
        phi = base_phi + np.deg2rad(rng.uniform(-TOPPLE_TIP_JITTER_DEG, TOPPLE_TIP_JITTER_DEG))
    else:                                                # cylinder: axisymmetric
        d = float(he[0]); phi = float(rng.uniform(0.0, 2.0 * np.pi))

    theta_crit = float(np.arctan2(d, half_h))
    alpha = theta_crit + np.deg2rad(rng.uniform(*TOPPLE_LEAN_MARGIN_DEG))
    axis = np.array([-np.sin(phi), np.cos(phi), 0.0])    # horizontal, ⟂ tip dir
    quat = np.array([np.cos(alpha / 2.0),
                     np.sin(alpha / 2.0) * axis[0],
                     np.sin(alpha / 2.0) * axis[1],
                     np.sin(alpha / 2.0) * axis[2]])      # wxyz
    R = _axis_angle_rotmat(axis, alpha)
    # Lowest tilted point relative to center along z = -Σ_k |R[2,k]|·he_k.
    center_z = TOPPLE_GROUND_CLEARANCE + float(np.sum(np.abs(R[2, :]) * np.asarray(he)))
    position = np.array([center_xy[0], center_xy[1], center_z])
    ang_vel = float(rng.uniform(*TOPPLE_KICK_ANGVEL_RANGE)) * axis
    return position, quat, ang_vel, float(np.rad2deg(alpha)), float(phi % (2.0 * np.pi))


def _place_topple_centers(rng, radii, config):
    """Rejection-sample distinct XY base centers (z is set later by the lean
    pose, so we can't reuse sample_non_overlapping_positions which samples z).
    Returns a list of (2,) arrays, or None if any body can't be placed within
    pose_rejection_max_attempts."""
    xy_lo, xy_hi = TOPPLE_XY_RANGE
    centers = []
    for r_i in radii:
        for _ in range(config.pose_rejection_max_attempts):
            cand = rng.uniform([xy_lo, xy_lo], [xy_hi, xy_hi])
            if all(
                np.linalg.norm(cand - c) > 1.10 * (r_i + r_j)
                for c, r_j in zip(centers, radii)
            ):
                centers.append(cand)
                break
        else:
            return None
    return centers


def build_topple(world, rng, config):
    """1-2 tall bodies spawned leaning past their tip angle; gravity topples them (W3a)."""
    n = int(rng.integers(TOPPLE_N_RANGE[0], TOPPLE_N_RANGE[1] + 1))
    shapes = [rng.choice(TOPPLE_SHAPES).item() for _ in range(n)]
    dims_list = [_sample_tall_dims(rng, s) for s in shapes]
    radii = [_topple_spacing_radius(shapes[i], dims_list[i]) for i in range(n)]

    # Degrade gracefully rather than hard-failing: if N bodies won't fit in the
    # spawn box, drop one and retry (N=1 always places). Removes the placement
    # RuntimeError on unlucky two-large-body draws.
    centers = _place_topple_centers(rng, radii, config)
    while centers is None and n > 1:
        n -= 1
        shapes, dims_list, radii = shapes[:n], dims_list[:n], radii[:n]
        centers = _place_topple_centers(rng, radii, config)
    if centers is None:
        raise _placement_failed("topple", n, config)

    bodies, body_specs = [], []
    initial_velocities = []
    lean_degs, tip_azimuths = [], []
    for i in range(n):
        pos, quat, ang_vel, lean_deg, tip_az = _topple_lean_pose(
            rng, shapes[i], dims_list[i], centers[i],
        )
        body, spec = body_factory.spawn_body(
            world, rng, config, i, shapes[i], dims_list[i], pos, orientation=quat,
        )
        bodies.append(body)
        body_specs.append(spec)
        initial_velocities.append((_zero_linear_velocity(), ang_vel))
        lean_degs.append(lean_deg)
        tip_azimuths.append(tip_az)

    scene_specs = {
        "static_assets": [],
        "scenario_specs": {
            "kick_angvel_range": list(TOPPLE_KICK_ANGVEL_RANGE),
            "lean_margin_deg": list(TOPPLE_LEAN_MARGIN_DEG),
            "footprint_range": list(TOPPLE_FOOTPRINT_RANGE),
            "height_range": list(TOPPLE_HEIGHT_RANGE),
            "lean_deg": lean_degs,
            "tip_azimuth_rad": tip_azimuths,
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- collide ----------------------------------------------------------------
# Observability scenario (physics-diversification.md W3). Two bodies on a
# shared horizontal axis are aimed at each other so they impact near the
# origin. Momentum exchange makes the *mass ratio* observable — the only way
# mass (now decoupled from volume by W1) becomes a learnable target rather than
# a relabelling of visible size.

COLLIDE_HALF_SEP_RANGE = (0.30, 0.42)        # m, each body's distance from origin
COLLIDE_SPEED_RANGE = (1.5, 30.0)             # m/s, horizontal closing speed per body
COLLIDE_HEIGHT_RANGE = (0.12, 0.22)          # m, shared spawn height of both centers
COLLIDE_AIM_JITTER_DEG = 8.0                 # max ± aim jitter so it's not perfectly 1D


def _rot_z(vec, ang):
    """Rotate a 3-vector about +Z by `ang` radians."""
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = vec
    return np.array([c * x - s * y, s * x + c * y, z])


def build_collide(world, rng, config):
    """Two bodies launched toward each other along a shared axis; they impact.

    Both centers spawn at a common height so they actually meet rather than one
    sailing over the other. Velocities point inward (toward the origin) with a
    small per-body aim jitter, so most collisions are near-head-on but not a
    degenerate 1-D line every clip.
    """
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    u = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    half_sep = float(rng.uniform(*COLLIDE_HALF_SEP_RANGE))
    z0 = float(rng.uniform(*COLLIDE_HEIGHT_RANGE))

    shapes = [rng.choice(config.shape_choices).item() for _ in range(2)]
    dims_list = [body_factory.sample_shape_dims(rng, s, config.scale_range) for s in shapes]

    # Body 0 at +half_sep along u (moving -u), body 1 at -half_sep (moving +u).
    positions = [half_sep * u + np.array([0.0, 0.0, z0]),
                 -half_sep * u + np.array([0.0, 0.0, z0])]
    inward = [-u, u]

    bodies, body_specs = [], []
    initial_velocities = []
    jitter = np.deg2rad(COLLIDE_AIM_JITTER_DEG)
    for i in range(2):
        body, spec = body_factory.spawn_body(
            world, rng, config, i, shapes[i], dims_list[i], positions[i],
        )
        bodies.append(body)
        body_specs.append(spec)
        speed = float(rng.uniform(*COLLIDE_SPEED_RANGE))
        aim = float(rng.uniform(-jitter, jitter))
        lin_vel = speed * _rot_z(inward[i], aim)
        initial_velocities.append((lin_vel, _sample_angular_velocity(rng, config)))

    scene_specs = {
        "static_assets": [],
        "scenario_specs": {
            "axis_azimuth_rad": azimuth,
            "half_separation": half_sep,
            "spawn_height": z0,
            "speed_range": list(COLLIDE_SPEED_RANGE),
            "aim_jitter_deg": COLLIDE_AIM_JITTER_DEG,
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
    "topple": build_topple,
    "collide": build_collide,
}

# Coarse expected-observability hint per scenario (physics-diversification.md
# W3). Consumed later by the observability-weighted loss (R6) and dataset
# balancing. Injected into scene_specs.scenario_specs by `build` below so the
# tag lives in meta.json without each builder having to set it.
#   free_flight  — mostly ballistic, weak contact (mass/CoM weakly observable)
#   contact      — sustained ground/scenery contact (CoM observable via tumbling)
#   collision    — inter-body momentum exchange (mass ratio observable)
OBSERVABILITY_HINT = {
    "drop": "contact",
    "roll": "contact",
    "throw": "free_flight",
    "drop_on_platform": "contact",
    "settle": "contact",
    "push": "contact",
    "cluttered": "contact",
    "topple": "contact",
    "collide": "collision",
}


def build(scenario_name: str, world, rng, config):
    """Dispatch by name. Raises KeyError if the scenario is unknown."""
    bodies, body_specs, scene_specs, initial_velocities = BUILDERS[scenario_name](
        world, rng, config,
    )
    scene_specs.setdefault("scenario_specs", {})["observability_hint"] = (
        OBSERVABILITY_HINT.get(scenario_name, "unknown")
    )
    return bodies, body_specs, scene_specs, initial_velocities
