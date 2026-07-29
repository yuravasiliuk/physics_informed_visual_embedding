"""Pair-experiment scenario builders — additive registration (CLAUDE.md §Principles).

Two archetypes designed for the controlled-intervention pair corpus
(gate-b-probe.md §13), where the CoM dichotomy is a *sustained* dynamical
signature instead of the one-shot chaotic fork a plain drop gives:

    weighted_roll   Spheres pushed horizontally across flat ground. An offset
                    CoM modulates rolling speed at the rolling frequency (the
                    "loaded die" wobble) and curves the path — a periodic,
                    dose-graded signature present in every frame window.
                    Initial angular velocity is zero so the wobble is
                    attributable to the CoM alone; a uniform sphere is the
                    purest R8 object (no pixel cue whatsoever).

    collide_static  Bodies launched horizontally at a static wall. Pre-impact
                    flight is torque-free (gravity acts through the CoM), so
                    twins stay pixel-identical until contact; the wall's
                    normal impulse acts off-CoM and converts linear momentum
                    into spin — a clean single-contact fork at a predictable
                    frame, with no second dynamic body. This replaces `collide`
                    for pair corpora: build_collide hardcodes 2 bodies and
                    ignores config.n_bodies_range, so --n_objects can't hold.

Both builders draw the body count via scenarios._sample_n_bodies, so
collect_pairs.py --n_objects (which pins config.n_bodies_range) holds exactly.

Importing this module registers the builders into scenarios.BUILDERS and
scenarios.OBSERVABILITY_HINT at the bottom of the file — scenarios.py itself
stays untouched, and collect_clip.py (which never imports this module) is
byte-for-byte unaffected.
"""
import numpy as np

import body_factory
import scenarios
import static_geometry

# --- tunables --------------------------------------------------------------
# config.py-style constants, kept local so config.py stays untouched.

# v/r at these speeds and scale_range radii (0.025–0.075 m) gives a rolling
# rate of ~11–80 rad/s → multiple wobble periods inside a 64-frame / ~2.1 s
# clip, which is what makes the signature stationary across probe windows.
WEIGHTED_ROLL_SPEED_RANGE = (0.8, 2.0)        # m/s, horizontal push
WEIGHTED_ROLL_LANE_GAP = 1.5                  # lane spacing, × max diameter
WEIGHTED_ROLL_BACKSET = 0.25                  # m, spawn behind the origin so the
                                              # roll crosses the look-at region
WEIGHTED_ROLL_CLEARANCE = 0.005               # m above ground (no t=0 penetration)

COLLIDE_STATIC_DIST_RANGE = (0.28, 0.40)      # m, spawn distance from the wall
COLLIDE_STATIC_SPEED_RANGE = (1.5, 4.0)       # m/s toward the wall
COLLIDE_STATIC_HEIGHT_RANGE = (0.12, 0.22)    # m, spawn height of body centers
COLLIDE_STATIC_AIM_JITTER_DEG = 6.0           # ± deg; small enough to still hit
COLLIDE_STATIC_LANE_GAP = 1.4                 # lane spacing, × 2·bounding radius
COLLIDE_STATIC_WALL_HEIGHT = 0.5              # m
COLLIDE_STATIC_WALL_THICKNESS = 0.04          # m (CCD is on, but stay tunnel-safe)
COLLIDE_STATIC_WALL_MARGIN = 0.15             # m wall width past the outer lanes


def build_weighted_roll(world, rng, config):
    """N spheres pushed along one horizontal direction on flat ground.

    Spheres only: W5 composites are cuboid-bboxed and don't roll, and
    cylinders/capsules spawn axis-up and would slide — so composite_frac and
    shape_choices are intentionally ignored here. Bodies are placed
    line-abreast on lanes perpendicular to the push, spaced wide enough that
    they never interact: each clip is n independent single-body experiments.
    """
    s = scenarios._scene_scale(config)
    azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    u = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])   # roll direction
    lat = np.array([-u[1], u[0], 0.0])                      # lane axis (⊥ u)

    n = scenarios._sample_n_bodies(rng, config)
    dims_list = [
        body_factory.sample_shape_dims(rng, "sphere", config.scale_range)
        for _ in range(n)
    ]
    radii = [d[0] for d in dims_list]
    gap = WEIGHTED_ROLL_LANE_GAP * 2.0 * max(radii)
    backset = scenarios._scaled(WEIGHTED_ROLL_BACKSET, s)

    bodies, body_specs, initial_velocities = [], [], []
    speeds = []
    for i, (dims, r) in enumerate(zip(dims_list, radii)):
        lane = (i - (n - 1) / 2.0) * gap
        pos = (-backset * u + lane * lat
               + np.array([0.0, 0.0, r + WEIGHTED_ROLL_CLEARANCE]))
        body, spec = body_factory.spawn_dynamic_body(
            world, rng, config, i, "sphere", dims, pos, composite=False,
        )
        speed = float(rng.uniform(*scenarios._scaled(WEIGHTED_ROLL_SPEED_RANGE, s)))
        bodies.append(body)
        body_specs.append(spec)
        # Zero initial ω: rolling is induced by ground friction, so any speed
        # wobble / path curvature is the CoM's doing, not a sampled spin.
        initial_velocities.append((speed * u, np.zeros(3, dtype=np.float64)))
        speeds.append(speed)

    scene_specs = {
        "static_assets": [],
        "scenario_specs": {
            "roll_azimuth_rad": azimuth,
            "speeds": speeds,
            "lane_gap": gap,
            "speed_range": list(WEIGHTED_ROLL_SPEED_RANGE),
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


def build_collide_static(world, rng, config):
    """N bodies launched at a static wall; one contact event per body.

    Geometry is axis-aligned (make_wall doesn't rotate): the wall stands at
    the origin spanning X, bodies spawn at -Y and fly toward +Y. Viewpoint
    variety comes from the camera rig, not the scene. Lanes along X are
    spaced so bodies never touch each other — n independent impacts.
    """
    s = scenarios._scene_scale(config)
    n = scenarios._sample_n_bodies(rng, config)
    dist = float(rng.uniform(*scenarios._scaled(COLLIDE_STATIC_DIST_RANGE, s)))

    # Same composite/shape sampling as build_collide (W5: a composite is a
    # cuboid bbox; collect_pairs forces composite_frac=0 for pair corpora).
    composite_flags = [bool(rng.uniform() < config.composite_frac) for _ in range(n)]
    shapes = [
        "cuboid" if composite_flags[i] else rng.choice(config.shape_choices).item()
        for i in range(n)
    ]
    dims_list = [
        body_factory.sample_shape_dims(rng, sh, config.scale_range) for sh in shapes
    ]
    radii = [body_factory.bounding_radius(sh, d) for sh, d in zip(shapes, dims_list)]

    gap = COLLIDE_STATIC_LANE_GAP * 2.0 * max(radii)
    span = (n - 1) * gap + 2.0 * max(radii)
    wall_width = span + 2.0 * scenarios._scaled(COLLIDE_STATIC_WALL_MARGIN, s)
    wall_height = scenarios._scaled(COLLIDE_STATIC_WALL_HEIGHT, s)
    wall_thickness = scenarios._scaled(COLLIDE_STATIC_WALL_THICKNESS, s)
    wall_base = np.array([0.0, 0.0, 0.0])
    static_geometry.make_wall(
        world, prim_path="/World/wall", name="wall",
        base_position=wall_base,
        height=wall_height, width=wall_width, thickness=wall_thickness,
    )

    jitter = np.deg2rad(COLLIDE_STATIC_AIM_JITTER_DEG)
    bodies, body_specs, initial_velocities = [], [], []
    for i in range(n):
        lane = (i - (n - 1) / 2.0) * gap
        z = max(
            float(rng.uniform(*scenarios._scaled(COLLIDE_STATIC_HEIGHT_RANGE, s))),
            radii[i] + WEIGHTED_ROLL_CLEARANCE,  # big bodies: clear the ground
        )
        pos = np.array([lane, -dist, z])
        body, spec = body_factory.spawn_dynamic_body(
            world, rng, config, i, shapes[i], dims_list[i], pos,
            composite=composite_flags[i],
        )
        speed = float(rng.uniform(*scenarios._scaled(COLLIDE_STATIC_SPEED_RANGE, s)))
        aim = float(rng.uniform(-jitter, jitter))
        # Zero initial ω: with gravity acting through the CoM, pre-impact
        # flight is torque-free, so twins are identical until wall contact and
        # the post-impact spin is purely the off-CoM impulse's signature.
        lin_vel = speed * scenarios._rot_z(np.array([0.0, 1.0, 0.0]), aim)
        bodies.append(body)
        body_specs.append(spec)
        initial_velocities.append((lin_vel, np.zeros(3, dtype=np.float64)))

    scene_specs = {
        "static_assets": [{
            "type": "wall",
            "prim_path": "/World/wall",
            "base_position_world": wall_base.tolist(),
            "height": wall_height,
            "width": wall_width,
            "thickness": wall_thickness,
        }],
        "scenario_specs": {
            "wall_distance": dist,
            "lane_gap": gap,
            "speed_range": list(COLLIDE_STATIC_SPEED_RANGE),
            "aim_jitter_deg": COLLIDE_STATIC_AIM_JITTER_DEG,
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- registration ------------------------------------------------------------
# Mutates the scenarios registry at import time; scenarios.build() then
# dispatches to these by name like any built-in. Both are sustained-contact
# scenarios in the W3 observability taxonomy (the wall has infinite mass, so
# there is no inter-body mass ratio to observe — "contact", not "collision").
scenarios.BUILDERS["weighted_roll"] = build_weighted_roll
scenarios.BUILDERS["collide_static"] = build_collide_static
scenarios.OBSERVABILITY_HINT["weighted_roll"] = "contact"
scenarios.OBSERVABILITY_HINT["collide_static"] = "contact"
