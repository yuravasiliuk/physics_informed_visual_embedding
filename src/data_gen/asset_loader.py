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
from pxr import Gf, UsdGeom, UsdPhysics

from seg_labels import apply_class_label


def _place(stage, prim_path: str, position, rotation_z_deg: float):
    """Author Translate + RotateZ on the reference prim via XformCommonAPI.

    XformCommonAPI is idempotent in the way AddTranslateOp/AddRotateZOp are
    not — repeated calls update the same op rather than appending duplicates.
    The referenced asset's internal xforms compose underneath ours.
    """
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(
        float(position[0]), float(position[1]), float(position[2]),
    ))
    xform.SetRotate(
        Gf.Vec3f(0.0, 0.0, float(rotation_z_deg)),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    return prim


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

    Returns a `GeometryPrim` handle. Static assets are scenery: the caller
    does not add them to `bodies`/`body_specs`, only to `scene_specs`.
    """
    add_reference_to_stage(usd_path=usd_url, prim_path=prim_path)
    stage = omni.usd.get_context().get_stage()
    prim = _place(stage, prim_path, position, rotation_z_deg)
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
