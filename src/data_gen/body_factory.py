"""Body-factory helpers shared by collect_clip.py and scenarios.py.

Extracted from collect_clip.py during step 5.2 of
`plans/scene-diversification.md` so scenario builders can call these helpers
without creating a circular import (collect_clip.py imports scenarios.py for
dispatch in step 5.3).

Pure functions: no side effects beyond the explicit Isaac Sim scene mutations
that the body-spawning helpers do. All randomness comes from the caller-
supplied `numpy.random.Generator`.
"""
import colorsys
import numpy as np

import omni.usd
from omni.isaac.core.materials import OmniPBR, PhysicsMaterial
from omni.isaac.core.objects import (
    DynamicCapsule,
    DynamicCuboid,
    DynamicCylinder,
    DynamicSphere,
)

from seg_labels import apply_class_label


def sample_color(rng, hue_range, sat_range, val_range):
    h = rng.uniform(*hue_range)
    s = rng.uniform(*sat_range)
    v = rng.uniform(*val_range)
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def sample_shape_dims(rng, shape, scale_range):
    """Sample per-shape dimensions. Returns a tuple of floats.

    Layout per shape:
        cuboid    -> (sx, sy, sz)
        sphere    -> (radius,)
        cylinder  -> (radius, height)
        capsule   -> (radius, height)
    """
    lo, hi = scale_range
    if shape == "cuboid":
        return tuple(rng.uniform(lo, hi, size=3).tolist())
    if shape == "sphere":
        return (float(rng.uniform(lo, hi) * 0.5),)
    if shape in ("cylinder", "capsule"):
        return (float(rng.uniform(lo, hi) * 0.5), float(rng.uniform(lo, hi)))
    raise ValueError(f"Unknown shape: {shape}")


def bounding_radius(shape, dims):
    """Bounding-sphere radius used for non-overlap rejection sampling."""
    if shape == "cuboid":
        return 0.5 * float(np.linalg.norm(dims))
    if shape == "sphere":
        return dims[0]
    if shape == "cylinder":
        return float(np.hypot(dims[0], 0.5 * dims[1]))
    if shape == "capsule":
        return dims[0] + 0.5 * dims[1]
    raise ValueError(f"Unknown shape: {shape}")


def sample_non_overlapping_positions(
    rng, radii, xy_range, height_range, max_attempts, margin=1.10,
):
    """Rejection-sample one xyz per radius so bounding spheres don't overlap.

    Returns a list of (3,) numpy arrays. Returns None if any body fails to
    place within `max_attempts` tries — caller should re-seed.
    """
    xy_lo, xy_hi = xy_range
    z_lo, z_hi = height_range

    positions = []
    for r_i in radii:
        for _ in range(max_attempts):
            pos = rng.uniform([xy_lo, xy_lo, z_lo], [xy_hi, xy_hi, z_hi])
            if all(
                np.linalg.norm(pos - p) > margin * (r_i + r_j)
                for p, r_j in zip(positions, radii)
            ):
                positions.append(pos)
                break
        else:
            return None
    return positions


def make_body(world, prim_path, name, shape, dims, position, color):
    """Spawn one dynamic primitive of the requested shape."""
    if shape == "cuboid":
        return world.scene.add(DynamicCuboid(
            prim_path=prim_path, name=name, position=position,
            scale=list(dims), color=color,
        ))
    if shape == "sphere":
        return world.scene.add(DynamicSphere(
            prim_path=prim_path, name=name, position=position,
            radius=dims[0], color=color,
        ))
    if shape == "cylinder":
        return world.scene.add(DynamicCylinder(
            prim_path=prim_path, name=name, position=position,
            radius=dims[0], height=dims[1], color=color,
        ))
    if shape == "capsule":
        return world.scene.add(DynamicCapsule(
            prim_path=prim_path, name=name, position=position,
            radius=dims[0], height=dims[1], color=color,
        ))
    raise ValueError(f"Unknown shape: {shape}")


def attach_physics_material(body, prim_path, static_friction, dynamic_friction, restitution):
    mat = PhysicsMaterial(
        prim_path=prim_path,
        static_friction=float(static_friction),
        dynamic_friction=float(dynamic_friction),
        restitution=float(restitution),
    )
    body.apply_physics_material(mat)


def attach_pbr_material(body, prim_path, color, roughness, metallic):
    """Attach an OmniPBR visual material — random roughness + occasional metallic.

    Step 5.10 (plan §5 Tier 3). Replaces the DynamicCuboid display-color path
    with a real PBR shader so the encoder sees specular highlights / metallic
    reflections rather than flat-shaded primitives.
    """
    mat = OmniPBR(prim_path=prim_path, color=np.asarray(color, dtype=np.float32))
    mat.set_reflection_roughness(float(roughness))
    mat.set_metallic_constant(float(metallic))
    body.apply_visual_material(mat)


def _sample_roughness_metallic(rng, config):
    """Sample (roughness, metallic) per plan §5 Tier 3.

    Roughness uniform in `config.roughness_range`; metallic is `0.0` most of
    the time and `rng.choice(config.metallic_values)` with probability
    `config.metallic_prob` — the "occasional metallic" the plan calls for.
    """
    roughness = float(rng.uniform(*config.roughness_range))
    if rng.uniform() < config.metallic_prob:
        metallic = float(rng.choice(config.metallic_values))
    else:
        metallic = 0.0
    return roughness, metallic


def build_random_bodies(world, rng, config, n_bodies, xy_range=None, height_range=None):
    """Sample shapes, dims, colors, positions, frictions; spawn `n_bodies`.

    `xy_range` and `height_range` default to `config.init_xy_range` and
    `config.init_height_range` respectively, but scenario builders can pass
    different ranges (e.g. spawn above a ramp's footprint).

    Returns (bodies, body_specs). Returns (None, None) if position rejection
    sampling fails — caller should re-seed.
    """
    xy_range = xy_range if xy_range is not None else config.init_xy_range
    height_range = height_range if height_range is not None else config.init_height_range

    shapes = [rng.choice(config.shape_choices).item() for _ in range(n_bodies)]
    dims_list = [sample_shape_dims(rng, s, config.scale_range) for s in shapes]
    radii = [bounding_radius(s, d) for s, d in zip(shapes, dims_list)]

    positions = sample_non_overlapping_positions(
        rng, radii, xy_range, height_range, config.pose_rejection_max_attempts,
    )
    if positions is None:
        return None, None

    stage = omni.usd.get_context().get_stage()
    bodies = []
    specs = []
    for i in range(n_bodies):
        name = f"body_{i:02d}"
        prim_path = f"/World/{name}"
        color = sample_color(
            rng, config.color_hue_range, config.color_sat_range, config.color_val_range,
        )
        body = make_body(
            world, prim_path=prim_path, name=name,
            shape=shapes[i], dims=dims_list[i],
            position=positions[i], color=color,
        )
        sf = float(rng.uniform(*config.static_friction_range))
        df = float(rng.uniform(*config.dynamic_friction_range))
        rest = float(rng.uniform(*config.restitution_range))
        attach_physics_material(
            body, prim_path=f"{prim_path}/physics_material",
            static_friction=sf, dynamic_friction=df, restitution=rest,
        )
        roughness, metallic = _sample_roughness_metallic(rng, config)
        attach_pbr_material(
            body, prim_path=f"{prim_path}/visual_material",
            color=color, roughness=roughness, metallic=metallic,
        )
        # Per-body class label — step 5.8 collapses prims under one class to
        # one segmentation ID, so each primitive body gets its own class
        # (matches asset bodies, which use `name` as their class label too).
        apply_class_label(stage.GetPrimAtPath(prim_path), name)
        bodies.append(body)
        specs.append({
            "id": i + 1,
            "name": name,
            "shape": shapes[i],
            "asset_url": None,
            "dims": list(map(float, dims_list[i])),
            "color_rgb": [float(c) for c in color],
            "roughness": roughness,
            "metallic": metallic,
            "static_friction": sf,
            "dynamic_friction": df,
            "restitution": rest,
            "init_position_world": [float(x) for x in positions[i]],
        })
    return bodies, specs
