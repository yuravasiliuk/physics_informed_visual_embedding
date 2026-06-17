"""Body-factory helpers shared by collect_clip.py and scenarios.py.

Extracted from collect_clip.py during step 5.2 of
`plans/data-gen/scene-diversification.md` so scenario builders can call these helpers
without creating a circular import (collect_clip.py imports scenarios.py for
dispatch in step 5.3).

Pure functions: no side effects beyond the explicit Isaac Sim scene mutations
that the body-spawning helpers do. All randomness comes from the caller-
supplied `numpy.random.Generator`.
"""
import colorsys
import os

import cv2
import numpy as np

import omni.usd
from omni.isaac.core.materials import OmniPBR, PhysicsMaterial
from omni.isaac.core.objects import (
    DynamicCapsule,
    DynamicCuboid,
    DynamicCylinder,
    DynamicSphere,
)
from omni.isaac.core.prims import RigidPrim
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

from config import GRID_TEXTURE_CACHE
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


def shape_volume(shape, dims):
    """Analytic volume (m^3) of a primitive.

    Matches PhysX's mass integration for box/sphere/cylinder/capsule, so
    `density * shape_volume` equals the mass PhysX resolves at world.reset()
    (cross-checked against state.json's per-frame `get_mass()`). dims layout
    mirrors sample_shape_dims:
        cuboid   -> (sx, sy, sz)   full side lengths
        sphere   -> (r,)
        cylinder -> (r, h)
        capsule  -> (r, h)         cylinder of length h + two hemisphere caps
    """
    if shape == "cuboid":
        return float(dims[0] * dims[1] * dims[2])
    if shape == "sphere":
        r = dims[0]
        return float(4.0 / 3.0 * np.pi * r ** 3)
    if shape == "cylinder":
        r, h = dims
        return float(np.pi * r ** 2 * h)
    if shape == "capsule":
        r, h = dims
        return float(np.pi * r ** 2 * h + 4.0 / 3.0 * np.pi * r ** 3)
    raise ValueError(f"Unknown shape: {shape}")


def attach_density(prim, density):
    """Author per-body density via UsdPhysics.MassAPI.

    PhysX computes mass = density * collision-shape-volume (and the inertia
    tensor) at world.reset(). No explicit mass/inertia attr is set, so density
    is the sole driver. Paired with attach_center_of_mass (W2), which overrides
    only the CoM; mass + inertia stay PhysX-computed and scale-correct.
    """
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr().Set(float(density))


def half_extents(shape, dims):
    """Per-axis half-extent of the collision volume (np.array of 3 floats).

    Used to bound a sampled CoM offset so it stays inside the body. Sphere
    uses its radius on all three axes; cylinder/capsule use radius on x,y and
    half the cylinder length on z (their local symmetry axis).
    """
    if shape == "cuboid":
        return np.array([0.5 * dims[0], 0.5 * dims[1], 0.5 * dims[2]])
    if shape == "sphere":
        r = dims[0]
        return np.array([r, r, r])
    if shape in ("cylinder", "capsule"):
        r, h = dims
        return np.array([r, r, 0.5 * h])
    raise ValueError(f"Unknown shape: {shape}")


def inertia_diagonal(shape, dims, mass):
    """Principal moments (Ixx, Iyy, Izz) of a uniform-density primitive about
    its centroid, in the body frame (np.array of 3 floats).

    The body axes are the principal axes for all four shapes, so this is the
    diagonal inertia with identity principal-axes orientation. NOT authored:
    we let PhysX compute inertia from density + geometry (scale-correct, and
    consistent with the offset CoM). Retained — Monte-Carlo-validated to <0.03%
    — as a reference / potential future label, not used at spawn time.

    cylinder/capsule symmetry axis = local Z (Isaac's default), matching the
    (r, h) dims layout. Capsule = cylinder(length h) + two hemispherical caps
    (radius r); component masses split by uniform density rho = mass / volume.
    """
    if shape == "cuboid":
        a, b, c = dims
        return mass / 12.0 * np.array([b*b + c*c, a*a + c*c, a*a + b*b])
    if shape == "sphere":
        r = dims[0]
        i = 0.4 * mass * r * r
        return np.array([i, i, i])
    if shape == "cylinder":
        r, h = dims
        izz = 0.5 * mass * r * r
        ixx = mass * (r*r / 4.0 + h*h / 12.0)
        return np.array([ixx, ixx, izz])
    if shape == "capsule":
        r, h = dims
        rho = mass / shape_volume("capsule", dims)
        m_cyl = rho * np.pi * r * r * h
        m_sph = rho * (4.0 / 3.0) * np.pi * r ** 3      # both hemispherical caps
        izz = 0.5 * m_cyl * r * r + 0.4 * m_sph * r * r
        ixx = (m_cyl * (r*r / 4.0 + h*h / 12.0)
               + m_sph * (0.4 * r*r + h*h / 4.0 + 0.375 * h * r))
        return np.array([ixx, ixx, izz])
    raise ValueError(f"Unknown shape: {shape}")


def prim_local_scale(prim):
    """Per-axis local scale of a prim from its USD xform (np.array of 3).

    Robust to however Isaac builds each shape: a cuboid is a unit Cube with
    xformOp:scale = dims, while sphere/cylinder/capsule set their geometry
    size directly (scale = 1). We read the composed local transform rather than
    assume, because the MassAPI CoM is authored in unscaled local space.
    """
    m = UsdGeom.Xformable(prim).GetLocalTransformation()
    s = Gf.Transform(m).GetScale()
    return np.array([s[0], s[1], s[2]], dtype=np.float64)


def attach_center_of_mass(prim, com_local, prim_scale):
    """Author an offset CoM via MassAPI, scale-corrected (W2 + W4 fix).

    `com_local` is the intended CoM offset in PHYSICAL body-local metres. USD's
    MassAPI centerOfMass is interpreted in the prim's UNSCALED local space, so
    we divide by the prim scale: physical CoM = attr * scale. Without this a
    scaled prim (a cuboid built with scale=dims) silently shrinks the offset by
    the scale factor and decouples it from the logged label — the bug the W4
    verify caught. Mass + inertia are left to PhysX (computed from density +
    geometry about this CoM); the offset CoM is the only mass property we pin.
    The authored value is cross-checked post-reset against read_local_com.
    """
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    com_attr = np.asarray(com_local, dtype=np.float64) / np.asarray(prim_scale, dtype=np.float64)
    mass_api.CreateCenterOfMassAttr().Set(
        Gf.Vec3f(float(com_attr[0]), float(com_attr[1]), float(com_attr[2]))
    )


def zero_damping(prim):
    """Remove PhysX's default velocity damping (W4 fix).

    PhysX applies a per-step exponential decay (omega /= 1 + angular_damping*dt,
    likewise for linear) — a numerical/authoring knob, NOT real aerodynamics.
    The default ~0.05 angular term silently bled ~1.7%/clip off spin (the W4
    spin_cuboid case caught it). At these scales (sub-metre, ~m/s, ~1 s) real
    air drag is negligible, so zero is both cleaner and more faithful: free
    flight is governed only by gravity + initial velocity, and bodies still
    settle via friction/restitution/contact. Keeping it controlled also means
    angular momentum is conserved through collisions (the `collide` mass signal)
    and we don't depend on a PhysX default that could change between versions.
    """
    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        rb = PhysxSchema.PhysxRigidBodyAPI(prim)
    else:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateLinearDampingAttr().Set(0.0)
    rb.CreateAngularDampingAttr().Set(0.0)


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


def enable_ccd(prim):
    """Enable PhysX continuous collision detection on a rigid body.

    Discrete collision only tests at substep endpoints, so a body moving more
    than ~its own half-extent per physics substep (the `collide` scenario's fast
    closing speeds) can tunnel through its target and leave the pair
    interpenetrating on the first rendered frame. CCD sweeps the shape along its
    trajectory so the contact is caught. PhysX gates body CCD on scene CCD, which
    collect_clip enables on the physics context.
    """
    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        rb = PhysxSchema.PhysxRigidBodyAPI(prim)
    else:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateEnableCCDAttr().Set(True)


def make_body(world, prim_path, name, shape, dims, position, color, orientation=None):
    """Spawn one dynamic primitive of the requested shape.

    `orientation` is an optional spawn orientation as a wxyz quaternion (Isaac's
    `Dynamic*` convention). Used by `topple` to spawn a body leaning just past
    its tip angle (physics-diversification.md W3a); None = axis-aligned.
    """

    if shape == "cuboid":
        return world.scene.add(DynamicCuboid(
            prim_path=prim_path, name=name, position=position, orientation=orientation,
            scale=list(dims), color=color,
        ))
    if shape == "sphere":
        return world.scene.add(DynamicSphere(
            prim_path=prim_path, name=name, position=position, orientation=orientation,
            radius=dims[0], color=color,
        ))
    if shape == "cylinder":
        return world.scene.add(DynamicCylinder(
            prim_path=prim_path, name=name, position=position, orientation=orientation,
            radius=dims[0], height=dims[1], color=color,
        ))
    if shape == "capsule":
        return world.scene.add(DynamicCapsule(
            prim_path=prim_path, name=name, position=position, orientation=orientation,
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


def ensure_grid_texture(path=GRID_TEXTURE_CACHE, size=1024, cells=8, line_frac=0.04):
    """Generate (once, then cache) a grid-line texture PNG and return its path.

    Light background with darker ruled lines (scene-diversification.md §10.4).
    Generated locally so the ground has a translation/parallax cue without the
    asset-server dependency of Isaac's shipped grid floor. The visible grid
    *density* is set per clip by OmniPBR texture_scale (UV tiling), so one cached
    texture serves all clips; only the tiling/tint vary.
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.full((size, size, 3), 210, dtype=np.uint8)      # light grey field
    step = size // cells
    half = max(1, int(step * line_frac))
    for k in range(0, size + 1, step):
        lo, hi = max(0, k - half), min(size, k + half)
        img[lo:hi, :] = 70                                    # horizontal rules
        img[:, lo:hi] = 70                                    # vertical rules
    cv2.imwrite(path, img)
    return path


def make_ground_grid_material(prim_path, rng, config, texture_path=None):
    """Build an OmniPBR grid material for the ground (returns the material).

    Bound to the GroundPlane via `GroundPlane(visual_material=...)`. The UV
    tiling (apparent line spacing) is randomized per clip from
    `config.ground_grid_tiling_range`; the cached grid PNG supplies the pattern.
    """
    texture_path = texture_path or ensure_grid_texture()
    tiling = float(rng.uniform(*config.ground_grid_tiling_range))
    return OmniPBR(
        prim_path=prim_path,
        texture_path=texture_path,
        texture_scale=np.array([tiling, tiling], dtype=np.float32),
    )


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


def spawn_body(world, rng, config, i, shape, dims, position, color=None, orientation=None):
    """Spawn one fully-decorated dynamic primitive; return (body, spec).

    The shared per-body construction used by `build_random_bodies` (which
    samples shape/dims/position) and by the bespoke scenario builders
    `topple`/`collide` (which choose them explicitly). Centralizing it keeps
    the W1 density path — and the physics/PBR/label/spec treatment — identical
    across both call sites, so a single edit can't decouple mass from volume in
    one place and not the other.

    `i` is the body index; name/prim_path/id are derived from it to match the
    existing convention. `color` is sampled if None. `orientation` is an
    optional wxyz spawn quaternion (used by `topple`'s lean, W3a). RNG draw
    order is color -> density -> friction(s) -> restitution -> roughness/metallic,
    matching the original inline loop exactly.
    """
    stage = omni.usd.get_context().get_stage()
    name = f"body_{i:02d}"
    prim_path = f"/World/{name}"
    if color is None:
        color = sample_color(
            rng, config.color_hue_range, config.color_sat_range, config.color_val_range,
        )
    body = make_body(
        world, prim_path=prim_path, name=name,
        shape=shape, dims=dims, position=position, color=color, orientation=orientation,
    )
    prim = stage.GetPrimAtPath(prim_path)
    # W1: decouple mass from visible volume via a per-body density (PhysX
    # resolves mass = density * volume at reset; we log the analytic value,
    # which matches, plus the measured get_mass() post-reset as the label).
    # W2: author an offset CoM so the CoM-offset label has real variance. The
    # offset is a per-axis fraction of the half-extent, bounded inside the
    # collision volume, scale-corrected in attach_center_of_mass, and verified
    # against read_local_com post-reset. Inertia is left to PhysX. Appearance
    # gives no cue to the CoM. (physics-diversification.md W1+W2 / #7,#9; R7.)
    density = float(rng.uniform(*config.density_range))
    volume = shape_volume(shape, dims)
    mass = density * volume
    com_local = (
        rng.uniform(-config.com_offset_frac, config.com_offset_frac, size=3)
        * half_extents(shape, dims)
    )
    attach_density(prim, density)
    attach_center_of_mass(prim, com_local, prim_local_scale(prim))
    zero_damping(prim)
    enable_ccd(prim)

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
    apply_class_label(prim, name)
    spec = {
        "id": i + 1,
        "name": name,
        "kind": "primitive",
        "shape": shape,
        "asset_url": None,
        "dims": list(map(float, dims)),
        "color_rgb": [float(c) for c in color],
        "roughness": roughness,
        "metallic": metallic,
        "static_friction": sf,
        "dynamic_friction": df,
        "restitution": rest,
        "density": density,
        "volume": volume,
        "mass": mass,
        "com_local_authored": [float(x) for x in com_local],
        "init_position_world": [float(x) for x in position],
        "init_orientation_world": (
            None if orientation is None else [float(x) for x in orientation]
        ),
    }
    return body, spec


def _bind_material(prim, material, purpose=None):
    """Bind a UsdShade.Material directly to `prim`. Used for composite bodies,
    where the rigid body is a parent Xform and the visible/collidable geometry
    are child Cubes: SingleRigidPrim exposes the dynamics interface but not
    GeometryPrim's `apply_*_material`, so we bind at the USD level. Binding is
    applied to each child gprim (NOT the parent — parent-Xform bindings don't
    propagate to the child Cubes for RTX rendering). The same material is bound
    to every child, so the composite still has one uniform appearance (no
    per-slab colour cue, R8). `purpose="physics"` binds a PhysicsMaterial.
    """
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose is None:
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants)
    else:
        binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, purpose)


def _partition_cuboid(rng, dims, n_parts, min_frac):
    """Split a cuboid (sx, sy, sz) into `n_parts` contiguous slabs along one
    random axis. Returns (parts, split_axis) where `parts` is a list of
    (slab_dims, offset_local): each slab keeps the full cross-section on the two
    non-split axes and a fraction of the box on the split axis; `offset_local`
    is the slab centre relative to the box centre.

    The slabs tile the box exactly — no overlap, no gap — so the union IS the
    cuboid bbox and Σ slab volume == box volume. This is what keeps the analytic
    composite mass/CoM closed-form (overlap would double-count mass). Slab
    lengths are a Dirichlet draw clamped to `min_frac` of the axis so no slab is
    a degenerate sliver.
    """
    dims = np.asarray(dims, dtype=np.float64)
    axis = int(rng.integers(0, 3))
    fracs = rng.dirichlet(np.ones(n_parts))
    fracs = np.maximum(fracs, min_frac)
    fracs = fracs / fracs.sum()                         # renormalise to tile exactly
    lengths = fracs * dims[axis]
    edges = np.concatenate([[0.0], np.cumsum(lengths)])
    centres = 0.5 * (edges[:-1] + edges[1:]) - 0.5 * dims[axis]   # rel. box centre
    parts = []
    for k in range(n_parts):
        slab_dims = dims.copy()
        slab_dims[axis] = lengths[k]
        offset = np.zeros(3)
        offset[axis] = centres[k]
        parts.append((slab_dims, offset))
    return parts, axis


def spawn_composite_body(world, rng, config, i, dims, position, color=None, orientation=None):
    """Spawn one composite rigid body; return (body, spec) (W5).

    A composite is a uniform-colour cuboid of overall size `dims`, internally
    split into 2-3 contiguous density slabs (`_partition_cuboid`). Each slab is a
    child Cube collider carrying its OWN density (`attach_density`); PhysX
    integrates the union into the body's mass / CoM / inertia at world.reset().
    No mass/CoM is authored on the body — unlike the W2 single-primitive path,
    the offset CoM *emerges* from the real mass distribution. Because the slabs
    share one material, the density asymmetry is invisible: the CoM is
    recoverable only from geometry + dynamics, never pixels (R8).

    The spec advertises `shape="cuboid"`, `dims=bbox` so every existing
    placement / lean / preroll path treats a composite exactly like a cuboid.
    Labels are closed-form: `mass = Σ ρ_k V_k`, `com_local = Σ ρ_k V_k t_k / mass`
    (logged as `com_local_analytic`); the post-reset `read_local_com` /
    `get_mass` cross-check in collect_clip must agree (asserted by the W5 verify
    case). Inertia stays PhysX-computed and unverified (no clean closed form).

    RNG draw order: color -> n_parts -> split(axis, lengths) -> per-slab density
    -> friction(s) -> restitution -> roughness/metallic.
    """
    stage = omni.usd.get_context().get_stage()
    name = f"body_{i:02d}"
    prim_path = f"/World/{name}"
    if color is None:
        color = sample_color(
            rng, config.color_hue_range, config.color_sat_range, config.color_val_range,
        )

    # Parent Xform = the single rigid body; child Cubes are its colliders.
    UsdGeom.Xform.Define(stage, prim_path)
    parent_prim = stage.GetPrimAtPath(prim_path)

    n_parts = int(rng.integers(
        config.composite_n_parts_range[0], config.composite_n_parts_range[1] + 1,
    ))
    parts_geom, split_axis = _partition_cuboid(
        rng, dims, n_parts, config.composite_split_min_frac,
    )

    part_specs = []
    child_prims = []
    total_mass = 0.0
    com_num = np.zeros(3, dtype=np.float64)
    for k, (slab_dims, offset) in enumerate(parts_geom):
        part_path = f"{prim_path}/part_{k:02d}"
        cube = UsdGeom.Cube.Define(stage, part_path)
        cube.GetSizeAttr().Set(1.0)                     # unit cube -> side == scale
        xc = UsdGeom.XformCommonAPI(cube.GetPrim())
        xc.SetScale(Gf.Vec3f(float(slab_dims[0]), float(slab_dims[1]), float(slab_dims[2])))
        xc.SetTranslate(Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2])))
        cube_prim = cube.GetPrim()
        child_prims.append(cube_prim)
        UsdPhysics.CollisionAPI.Apply(cube_prim)
        density = float(rng.uniform(*config.density_range))
        attach_density(cube_prim, density)
        volume = float(slab_dims[0] * slab_dims[1] * slab_dims[2])
        mass_k = density * volume
        total_mass += mass_k
        com_num += mass_k * offset
        part_specs.append({
            "shape": "cuboid",
            "dims": [float(x) for x in slab_dims],
            "offset_local": [float(x) for x in offset],
            "density": density,
            "volume": volume,
            "mass": mass_k,
        })
    com_local_analytic = com_num / total_mass

    # Make the parent a rigid body (PhysX computes mass/CoM/inertia from the
    # child colliders + their densities). Apply the API explicitly before
    # wrapping — the proven asset_loader.load_dynamic_asset pattern.
    UsdPhysics.RigidBodyAPI.Apply(parent_prim)
    zero_damping(parent_prim)
    enable_ccd(parent_prim)


    # One shared visual + physics material, bound to EACH child gprim. Binding on
    # the parent Xform does NOT propagate to the child Cubes for RTX rendering
    # (a smoke run showed composites rendering default grey, not their sampled
    # colour — which would itself be an R8 cue: "grey box == composite"). Binding
    # per child gives them the SAME colour (still no per-slab cue) and renders it
    # correctly; physics material on the collider is also where PhysX reads it.
    sf = float(rng.uniform(*config.static_friction_range))
    df = float(rng.uniform(*config.dynamic_friction_range))
    rest = float(rng.uniform(*config.restitution_range))
    roughness, metallic = _sample_roughness_metallic(rng, config)
    pbr = OmniPBR(prim_path=f"{prim_path}/visual_material", color=np.asarray(color, dtype=np.float32))
    pbr.set_reflection_roughness(roughness)
    pbr.set_metallic_constant(metallic)
    phys_mat = PhysicsMaterial(
        prim_path=f"{prim_path}/physics_material",
        static_friction=sf, dynamic_friction=df, restitution=rest,
    )
    for cube_prim in child_prims:
        _bind_material(cube_prim, pbr.material)
        _bind_material(cube_prim, phys_mat.material, purpose="physics")
    apply_class_label(parent_prim, name)

    body = world.scene.add(RigidPrim(
        prim_path=prim_path, name=name, position=position, orientation=orientation,
    ))

    spec = {
        "id": i + 1,
        "name": name,
        "kind": "composite",
        "shape": "cuboid",
        "asset_url": None,
        "dims": [float(x) for x in dims],
        "color_rgb": [float(c) for c in color],
        "roughness": roughness,
        "metallic": metallic,
        "static_friction": sf,
        "dynamic_friction": df,
        "restitution": rest,
        "split_axis": split_axis,
        "n_parts": n_parts,
        "parts": part_specs,
        "mass": total_mass,
        "com_local_analytic": [float(x) for x in com_local_analytic],
        "init_position_world": [float(x) for x in position],
        "init_orientation_world": (
            None if orientation is None else [float(x) for x in orientation]
        ),
    }
    return body, spec


def spawn_dynamic_body(world, rng, config, i, shape, dims, position,
                       color=None, orientation=None, composite=False):
    """Dispatch one dynamic body to the primitive (W2) or composite (W5) path.

    Shared by every scenario so the composite-vs-primitive branch lives in one
    place. Composites require a cuboid `dims` (the bbox); callers force
    `shape="cuboid"` when `composite=True`.
    """
    if composite:
        return spawn_composite_body(
            world, rng, config, i, dims, position, color=color, orientation=orientation,
        )
    return spawn_body(
        world, rng, config, i, shape, dims, position, color=color, orientation=orientation,
    )


def build_random_bodies(world, rng, config, n_bodies, xy_range=None, height_range=None):
    """Sample shapes, dims, colors, positions, frictions; spawn `n_bodies`.

    `xy_range` and `height_range` default to `config.init_xy_range` and
    `config.init_height_range` respectively, but scenario builders can pass
    different ranges (e.g. spawn above a ramp's footprint).

    Each body is independently a composite with probability `config.composite_frac`
    (W5); a composite is forced to a cuboid bbox so its bounding radius /
    non-overlap placement is identical to a primitive cuboid's. The rest of the
    pipeline can't tell the two apart (both advertise shape="cuboid").

    Returns (bodies, body_specs). Returns (None, None) if position rejection
    sampling fails — caller should re-seed.
    """
    xy_range = xy_range if xy_range is not None else config.init_xy_range
    height_range = height_range if height_range is not None else config.init_height_range

    composite_flags = [bool(rng.uniform() < config.composite_frac) for _ in range(n_bodies)]
    shapes = [
        "cuboid" if composite_flags[i] else rng.choice(config.shape_choices).item()
        for i in range(n_bodies)
    ]
    dims_list = [sample_shape_dims(rng, s, config.scale_range) for s in shapes]
    radii = [bounding_radius(s, d) for s, d in zip(shapes, dims_list)]

    positions = sample_non_overlapping_positions(
        rng, radii, xy_range, height_range, config.pose_rejection_max_attempts,
    )
    if positions is None:
        return None, None

    bodies = []
    specs = []
    for i in range(n_bodies):
        body, spec = spawn_dynamic_body(
            world, rng, config, i, shapes[i], dims_list[i], positions[i],
            composite=composite_flags[i],
        )
        bodies.append(body)
        specs.append(spec)
    return bodies, specs
