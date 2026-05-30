"""Isaac Sim asset loaders.

Step 5.6 of `plans/scene-diversification.md`. Two helpers — one for static
colliders (no dynamics, scene scaffolding) and one for dynamic rigid bodies
(drop-in replacement for primitive bodies in body_factory).

Both apply the appropriate `UsdPhysics` APIs and tag the referenced prim
with a class semantic label. Step 5.8 will configure the segmentation
annotator to group prims by class label so multi-prim assets (e.g. a Franka
arm's ~12 links) collapse to one instance ID — matching the per-body
identity the probe needs.

Mass override on the dynamic loader is exposed but defaults to "trust the
asset's authored mass" per plan §9 question 3.

The asset_loader module is dead code until step 5.9 (`cluttered` scenario)
wires it into a scenario builder.
"""
from typing import Optional

import numpy as np

import omni.usd
from omni.isaac.core.prims import GeometryPrim, RigidPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from seg_labels import apply_class_label


_ROT_OP_TO_COMMON_ORDER = {
    UsdGeom.XformOp.TypeRotateXYZ: UsdGeom.XformCommonAPI.RotationOrderXYZ,
    UsdGeom.XformOp.TypeRotateXZY: UsdGeom.XformCommonAPI.RotationOrderXZY,
    UsdGeom.XformOp.TypeRotateYXZ: UsdGeom.XformCommonAPI.RotationOrderYXZ,
    UsdGeom.XformOp.TypeRotateYZX: UsdGeom.XformCommonAPI.RotationOrderYZX,
    UsdGeom.XformOp.TypeRotateZXY: UsdGeom.XformCommonAPI.RotationOrderZXY,
    UsdGeom.XformOp.TypeRotateZYX: UsdGeom.XformCommonAPI.RotationOrderZYX,
}


def _detect_rotation_order(prim):
    """Return the rotation order already authored on `prim`, or XYZ by default.

    Some Isaac assets author `xformOp:rotateZYX` on the root reference (e.g.
    `dolly_physics.usd`, several Props/Mounts); calling `SetRotate(..., XYZ)`
    on those prims raises `XformCommonAPI`'s rotation-order-mismatch error.
    Since our rotation is Z-only either order yields the same result — we
    just have to match whatever the asset already uses.
    """
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        order = _ROT_OP_TO_COMMON_ORDER.get(op.GetOpType())
        if order is not None:
            return order
    return UsdGeom.XformCommonAPI.RotationOrderXYZ


def _place(stage, prim_path: str, position, rotation_z_deg: float):
    """Author Translate + RotateZ on the reference prim via XformCommonAPI.

    XformCommonAPI is idempotent in the way AddTranslateOp/AddRotateZOp are
    not — repeated calls update the same op rather than appending duplicates.
    The referenced asset's internal xforms compose underneath ours.
    """
    prim = stage.GetPrimAtPath(prim_path)
    rot_order = _detect_rotation_order(prim)
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(
        float(position[0]), float(position[1]), float(position[2]),
    ))
    xform.SetRotate(
        Gf.Vec3f(0.0, 0.0, float(rotation_z_deg)),
        rot_order,
    )
    return prim


def _disable_authored_dynamics(root_prim):
    """Walk root_prim's subtree and disable any authored physics dynamics.

    Some Isaac assets ship with `RigidBodyAPI` baked in — Dolly/dolly_physics.usd
    has 4 rigid-body wheels with castor joints, Forklift/forklift.usd has
    hydraulic joints, etc. Without this pass, referencing such an asset as
    "static scenery" still leaves PhysX simulating its parts, and contact
    forces from dynamic bodies make the cart roll across the floor.

    Strategy: for every descendant rigid body, set `physics:rigidBodyEnabled
    = False`; for every authored joint, set `physics:jointEnabled = False`.
    Disabling rigid bodies alone leaves joints orphaned (both endpoints become
    static and PhysX logs `cannot create a joint between static bodies` for
    each), so we silence them too. Visual + collision geometry stay intact;
    PhysX just stops integrating and simulating constraints.

    An articulation whose constituent links and joints are all disabled has
    nothing to simulate, so we don't need to touch `ArticulationRootAPI`
    separately (it's also a marker API with no `enabled` attribute on the
    base UsdPhysics schema).
    """
    for desc in Usd.PrimRange(root_prim):
        if desc.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(desc).CreateRigidBodyEnabledAttr(False)
        if desc.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(desc).CreateJointEnabledAttr(False)


def load_static_asset(
    world,
    usd_url: str,
    prim_path: str,
    name: str,
    position,
    semantic_label: str,
    rotation_z_deg: float = 0.0,
):
    """Reference a USD as a kinematic collider — no rigid body dynamics.

    `position` is the world-frame location of the referenced asset's root
    xform; `rotation_z_deg` is azimuthal rotation (Z-up).

    Disables any `RigidBodyAPI` / `ArticulationRootAPI` authored inside the
    referenced USD so assets like `dolly_physics.usd` (4 castor wheels) and
    `forklift.usd` (hydraulic joints) don't roll under contact forces.

    Returns a `GeometryPrim` handle. Static assets are scenery: the caller
    does not add them to `bodies`/`body_specs`, only to `scene_specs`.
    """
    add_reference_to_stage(usd_path=usd_url, prim_path=prim_path)
    stage = omni.usd.get_context().get_stage()
    prim = _place(stage, prim_path, position, rotation_z_deg)
    _disable_authored_dynamics(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    apply_class_label(prim, semantic_label)
    return world.scene.add(GeometryPrim(prim_path=prim_path, name=name))


def load_dynamic_asset(
    world,
    usd_url: str,
    prim_path: str,
    name: str,
    position,
    semantic_label: str,
    rotation_z_deg: float = 0.0,
    mass: Optional[float] = None,
):
    """Reference a USD as a dynamic rigid body — drop-in for primitive bodies.

    Applies `RigidBodyAPI` + `CollisionAPI` + `MassAPI` to the reference prim.
    If `mass` is None, the asset's authored mass is preserved (plan §9 q3:
    trust shipped USD masses for v0.1). Pass an explicit mass to override.

    Returns a `RigidPrim` handle compatible with the body interface
    log_state_frame expects: get_world_pose, get_linear_velocity,
    get_angular_velocity, set_linear_velocity, set_angular_velocity,
    get_mass.
    """
    add_reference_to_stage(usd_path=usd_url, prim_path=prim_path)
    stage = omni.usd.get_context().get_stage()
    prim = _place(stage, prim_path, position, rotation_z_deg)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    if mass is not None:
        mass_api.CreateMassAttr().Set(float(mass))
    apply_class_label(prim, semantic_label)
    return world.scene.add(RigidPrim(prim_path=prim_path, name=name))
