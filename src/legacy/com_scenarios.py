"""CoM-observability scenarios — RETIRED 2026-07-28. NOT IMPORTABLE, read only.

Both builders existed for one purpose: make the **authored offset CoM** (W2) the
deciding variable in a body's macroscopic motion, so a probe could be asked to
decode it. The CoM target is abandoned (EXPERIMENTS.md Graveyard: not decodable
even from ground-truth trajectory features), and `body_factory.attach_center_of_mass`
was removed with it — so as of that removal these two scenarios no longer do what
their docstrings say. `balance` would decide purely on geometry, and
`eccentric_roll` would roll a *uniform* cylinder, i.e. plain rolling with no
wobble. Leaving them callable would have been worse than deleting them: the code
would run and quietly produce physics that contradicts its own documentation.

Kept here because the DESIGNS are the durable part — both are worked examples of
the same idea, which the surviving lines still use: put the quantity of interest
in a **narrow contested band** where it decides the outcome, instead of
overdetermining the outcome and hoping the quantity leaves a trace. That is
exactly the reframing `balance` applied to W3a's `topple`, and it is the reason
the event-anchored material targets worked (restitution R^2 -0.018 -> +0.217).

These reference `scenarios` helpers (`_sample_n_bodies`, `_lean_*`, `TOPPLE_*`)
that stay in the live module; this file is documentation of intent, not a
runnable unit. Per src/legacy/README.md, files here are read, not imported.
"""

# ---------------------------------------------------------------------------
# Config constants these used (removed from config.py in the same commit)
# ---------------------------------------------------------------------------
# --- balance ------------------------------------------------------------------
# W6 — CoM-observability-first (physics-diversification.md W6). The com_oracle
# (results/probe/com_oracle) showed CoM is NOT decodable even from GT
# trajectories on the v2 corpus: W3a's topple guarantees motion by
# OVERDETERMINING it — leaning 3-8° PAST critical toward a random azimuth
# decides the outcome before the offset can matter. `balance` inverts that:
# lean is sampled in a narrow band AROUND the GEOMETRIC critical angle, so
# whether the body tips or rocks back — and how fast — is decided by where its
# invisible (W2/W5, R8) CoM actually sits. The authored offset (≤0.3·half-
# extent) shifts the TRUE threshold by up to ~±5°, wider than the ±2.5° band,
# so the offset is the deciding variable for most draws.
BALANCE_N_RANGE = (1, 2)                     # inclusive
# Pilot retune (2026-07-05, pilot_w6, n=31 bodies): the ±2.5° band tipped 84% —
# the EMPIRICAL boundary sits below the geometric theta_crit (kick + the 5 mm
# settle-drop jolt push knife-edge bodies over), so the band is re-centred
# negative to target ~50% tip rate, and the kick is cut to near-zero. The
# design mechanism is confirmed: corr(tipped, offset-toward-tip) = +0.34,
# BEATING the geometric margin (+0.28) in the contested band (§W6).
BALANCE_LEAN_BAND_DEG = (-4.5, 0.5)          # sampled AROUND theta_crit, re-centred on the empirical boundary
BALANCE_KICK_ANGVEL_RANGE = (0.0, 0.05)      # rad/s, ~zero: nothing may swamp the offset
# Footprint / height / XY / clearance / tip-jitter / shapes reuse TOPPLE_*.

# --- eccentric_roll -------------------------------------------------------------
# W6 — CoM-observability-first, continuous variant. Rolling couples the offset
# to macroscopic motion the entire clip: an offset-CoM cylinder/sphere rolls
# like an eccentric wheel (speed oscillation at the rotation frequency, curved
# path for spheres) and settles CoM-low, so the REST ORIENTATION reads out the
# offset direction in the body frame. Primitives only (composites are cuboids —
# they don't roll); offset authored invisibly by spawn_body (R8).
ECCROLL_N_RANGE = (1, 3)                     # inclusive
ECCROLL_SHAPES = ("cylinder", "sphere")
ECCROLL_RADIUS_RANGE = (0.05, 0.10)          # m
ECCROLL_LENGTH_RANGE = (0.15, 0.30)          # m, cylinder length (axis horizontal)
# Pilot retunes (2026-07-05, pilot_w6 + _v2): rest fraction stayed ~14% even at
# 0.4-1.2 m/s × 128 frames — with angular damping deliberately zeroed (W4) and
# no PhysX rolling resistance, pure rolling is LOSSLESS, so most rollers never
# rest regardless of speed/length. Reframe (§W6): the readout is the terminal
# OSCILLATION AROUND the CoM-low pose (a frictionless pendulum), which the
# trajectory carries the whole clip; the few bodies that do rest confirm the
# channel (cos(offset, down) med +0.88..+0.97). Keep slow speeds (wobble
# resolution) + --frame_count 128 (more oscillation cycles on film).
ECCROLL_SPEED_RANGE = (0.4, 1.2)             # m/s — slow, so the wobble modulates motion visibly
ECCROLL_XY_RANGE = (-0.30, 0.30)
ECCROLL_GROUND_CLEARANCE = 0.005             # spawn just clear of ground


def build_balance(world, rng, config):
    """1-2 tall bodies leaning AT their geometric tip angle; the CoM decides (W6).

    DEPRECATED (2026-07-27): built solely to make the CoM offset observable, and
    CoM is abandoned (EXPERIMENTS.md Graveyard). Dropped from the rebuild — it is
    also weak for the replacement targets: a near-equilibrium scenario produces
    few high-energy impacts, and restitution needs `e·|vz_in| > g·Δt`. Never in
    `config.scenarios` (pilots invoked it explicitly), so nothing changes by
    default. Kept callable per CLAUDE.md §Retire-don't-delete; remove on the next
    substantial edit of this module if still unused.
    """
    scene_s = _scene_scale(config)
    footprint_range = _scaled(TOPPLE_FOOTPRINT_RANGE, scene_s)
    height_range = _scaled(TOPPLE_HEIGHT_RANGE, scene_s)
    xy_range = _scaled(TOPPLE_XY_RANGE, scene_s)

    n = int(rng.integers(BALANCE_N_RANGE[0], BALANCE_N_RANGE[1] + 1))
    # Composites balance too: their invisible density split (W5) moves the true
    # threshold exactly like the W2 authored offset. Forced to cuboid (flat base).
    composite_flags = [bool(rng.uniform() < config.composite_frac) for _ in range(n)]
    shapes = [
        "cuboid" if composite_flags[i] else rng.choice(TOPPLE_SHAPES).item()
        for i in range(n)
    ]
    dims_list = [_sample_tall_dims(rng, sh, footprint_range, height_range) for sh in shapes]
    radii = [_topple_spacing_radius(shapes[i], dims_list[i]) for i in range(n)]

    centers = _place_topple_centers(rng, radii, config, xy_range)
    while centers is None and n > 1:
        n -= 1
        shapes, dims_list, radii = shapes[:n], dims_list[:n], radii[:n]
        composite_flags = composite_flags[:n]
        centers = _place_topple_centers(rng, radii, config, xy_range)
    if centers is None:
        raise _placement_failed("balance", n, config)

    bodies, body_specs = [], []
    initial_velocities = []
    lean_degs, tip_azimuths, theta_crits = [], [], []
    for i in range(n):
        pos, quat, ang_vel, lean_deg, tip_az, theta_crit = _topple_lean_pose(
            rng, shapes[i], dims_list[i], centers[i],
            lean_band_deg=BALANCE_LEAN_BAND_DEG,
            kick_range=BALANCE_KICK_ANGVEL_RANGE,
        )
        body, spec = body_factory.spawn_dynamic_body(
            world, rng, config, i, shapes[i], dims_list[i], pos,
            orientation=quat, composite=composite_flags[i],
        )
        bodies.append(body)
        body_specs.append(spec)
        initial_velocities.append((_zero_linear_velocity(), ang_vel))
        lean_degs.append(lean_deg)
        tip_azimuths.append(tip_az)
        theta_crits.append(theta_crit)

    scene_specs = {
        "static_assets": [],
        "scenario_specs": {
            # lean − theta_crit is the geometric margin; the SIGN of the true
            # margin (with the CoM folded in) is what the clip reveals.
            "lean_band_deg": list(BALANCE_LEAN_BAND_DEG),
            "kick_angvel_range": list(BALANCE_KICK_ANGVEL_RANGE),
            "lean_deg": lean_degs,
            "theta_crit_geom_deg": theta_crits,
            "tip_azimuth_rad": tip_azimuths,
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- eccentric_roll -----------------------------------------------------------
# W6 CoM-observability-first scenario, continuous variant (physics-
# diversification.md W6). Rolling couples the offset to macroscopic motion for
# the WHOLE clip, not one transient: an offset-CoM cylinder/sphere rolls like
# an eccentric wheel — speed oscillates at the rotation frequency, spheres
# curve — and settles CoM-low, so the rest orientation reads out the offset
# direction in the body frame. Primitives only (composites are cuboids and
# don't roll); the offset is authored invisibly by spawn_body (R8: one uniform
# material, no appearance cue).

# Tunables: config.ECCROLL_*


def _eccroll_body(rng, s):
    """Sample (shape, dims, orientation_wxyz, roll_dir, spawn_z) for one roller.

    Cylinders are laid on their side: the local +Z axis is rotated to a random
    horizontal azimuth ψ (90° about the horizontal axis ⟂ ψ), and the roll
    direction is ⟂ the cylinder axis. Spheres take a random orientation (so the
    authored CoM direction is uniformly distributed in the world) and a random
    roll direction.
    """
    shape = rng.choice(ECCROLL_SHAPES).item()
    r = float(rng.uniform(*_scaled(ECCROLL_RADIUS_RANGE, s)))
    if shape == "cylinder":
        h = float(rng.uniform(*_scaled(ECCROLL_LENGTH_RANGE, s)))
        dims = (r, h)
        psi = float(rng.uniform(0.0, 2.0 * np.pi))
        u = np.array([-np.sin(psi), np.cos(psi), 0.0])   # rotation axis ⟂ ψ
        quat = np.array([np.cos(np.pi / 4.0),            # 90° about u → axis horizontal
                         np.sin(np.pi / 4.0) * u[0],
                         np.sin(np.pi / 4.0) * u[1],
                         np.sin(np.pi / 4.0) * u[2]])
        sign = float(rng.choice([-1.0, 1.0]))
        roll_dir = sign * np.array([-np.sin(psi), np.cos(psi), 0.0])
    else:                                                # sphere
        dims = (r,)
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        quat = np.array([np.cos(ang / 2.0), *(np.sin(ang / 2.0) * ax)])
        az = float(rng.uniform(0.0, 2.0 * np.pi))
        roll_dir = np.array([np.cos(az), np.sin(az), 0.0])
    return shape, dims, quat, roll_dir, r + ECCROLL_GROUND_CLEARANCE


def build_eccentric_roll(world, rng, config):
    """1-3 offset-CoM cylinders/spheres rolling on flat ground; the wobble is the label (W6).

    DEPRECATED (2026-07-27): the only R6-certified CoM scenario (+0.020), retired
    with the CoM target. Also the weakest fit for the replacement targets: PhysX
    rolling here is lossless, and a rolling contact carries NO friction-magnitude
    information — the same mechanism that flipped the sign of the μ estimator
    (material_oracle._slide_stats). Never in `config.scenarios`. Kept callable per
    CLAUDE.md §Retire-don't-delete.
    """
    s = _scene_scale(config)
    xy_range = _scaled(ECCROLL_XY_RANGE, s)

    n = int(rng.integers(ECCROLL_N_RANGE[0], ECCROLL_N_RANGE[1] + 1))
    sampled = [_eccroll_body(rng, s) for _ in range(n)]
    # Spacing radius: lying cylinder sweeps ~half its length; sphere its radius.
    radii = [max(dims) * (0.5 if shape == "cylinder" else 1.0) + 0.02
             for shape, dims, _, _, _ in sampled]

    centers = _place_topple_centers(rng, radii, config, xy_range)
    while centers is None and n > 1:
        n -= 1
        sampled, radii = sampled[:n], radii[:n]
        centers = _place_topple_centers(rng, radii, config, xy_range)
    if centers is None:
        raise _placement_failed("eccentric_roll", n, config)

    bodies, body_specs = [], []
    initial_velocities = []
    speeds, azimuths = [], []
    for i, (shape, dims, quat, roll_dir, z0) in enumerate(sampled):
        pos = np.array([centers[i][0], centers[i][1], z0])
        # Primitives only: composite=False by construction (rolling needs a
        # curved surface; the offset comes from the W2 authored CoM).
        body, spec = body_factory.spawn_dynamic_body(
            world, rng, config, i, shape, dims, pos,
            orientation=quat, composite=False,
        )
        bodies.append(body)
        body_specs.append(spec)
        speed = float(rng.uniform(*ECCROLL_SPEED_RANGE))
        r = dims[0]
        # Rolling-without-slipping start: ω = (v/r)·(d̂ × ẑ) gives v = ω × r_c
        # for the ground contact at −r·ẑ — no initial skid transient to mask
        # the eccentric wobble.
        omega = (speed / r) * np.cross(roll_dir, np.array([0.0, 0.0, 1.0]))
        initial_velocities.append((speed * roll_dir, omega))
        speeds.append(speed)
        azimuths.append(float(np.arctan2(roll_dir[1], roll_dir[0])))

    scene_specs = {
        "static_assets": [],
        "scenario_specs": {
            "speed_range": list(ECCROLL_SPEED_RANGE),
            "speeds": speeds,
            "roll_azimuth_rad": azimuths,
        },
    }
    return bodies, body_specs, scene_specs, initial_velocities


# --- collide ----------------------------------------------------------------
# Observability scenario (physics-diversification.md W3). Two bodies on a
# shared horizontal axis are aimed at each other so they impact near the
# origin. Momentum exchange makes the *mass ratio* observable — the only way
# mass (now decoupled from volume by W1) becomes a learnable target rather than
# a relabelling of visible size.

# Tunables: config.COLLIDE_*
