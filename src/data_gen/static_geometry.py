"""Static scene scaffolding (kinematic colliders).

Step 5.1 of `plans/data-gen/scene-diversification.md`. Pure-primitive helpers for ramps,
walls, platforms, and open boxes — used by scenario builders to give bodies
something to interact with beyond the ground plane.

Helpers return either a FixedCuboid handle, a GeometryPrim handle (for the
mesh wedge), or a list of handles (for the multi-piece box). Static prims
are not added to per-body state records: they don't move, so the encoder
learns them as "scene" rather than "object." They participate in instance
segmentation only as scenery.

Coordinate convention: Isaac Sim is Z-up. The default ground plane sits at
z=0. All `base_position` arguments refer to the spot where the geometry
meets the ground (toe of wedge, base of wall, center of platform's lower
face, center of box's floor).
"""
import numpy as np

import omni.usd
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.prims import GeometryPrim
from pxr import Gf, UsdGeom, UsdPhysics

from seg_labels import SCENERY_LABEL, apply_class_label


DEFAULT_COLOR = np.array([0.55, 0.55, 0.58], dtype=np.float32)  # neutral grey


def _tag_scenery(stage, *prim_paths: str) -> None:
    """Attach `class=scenery` to each prim path. Step 5.8 will collapse all
    prims sharing this class into a single segmentation ID."""
    for p in prim_paths:
        apply_class_label(stage.GetPrimAtPath(p), SCENERY_LABEL)


def make_ramp(
    world,
    prim_path: str,
    name: str,
    base_position,
    angle_deg: float,
    length: float = 0.6,
    width: float = 0.4,
    color=DEFAULT_COLOR,
):
    """Solid wedge (right-triangular prism) resting on the ground.

    Cross-section in the YZ plane is a right triangle with the right-angle
    at the +Y end:
        toe   (low corner)  at +Y = 0,       +Z = 0
        back               at +Y = length,   +Z = 0
        top                at +Y = length,   +Z = length * tan(angle_deg)
    The hypotenuse (toe → top) is the rolling surface; bodies dropped above
    it slide/roll toward -Y. Extruded by `width` along ±X.

    `base_position` is the world position of the toe (low corner) at ground
    level. A solid wedge — not a thin rotated plate — so the geometry is
    physically self-supporting; a thin plate tilted in the air would topple
    in the real world even though FixedCuboid colliders are immune to
    gravity, and the unphysical look would teach the encoder a bad prior.
    """
    stage = omni.usd.get_context().get_stage()
    angle_rad = np.deg2rad(angle_deg)
    height = length * np.tan(angle_rad)
    base = np.asarray(base_position, dtype=np.float64)
    half_W = 0.5 * width
    c = np.asarray(color, dtype=np.float64)

    def _vec(offset):
        p = base + np.asarray(offset, dtype=np.float64)
        return Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))

    points = [
        _vec([-half_W, 0.0,    0.0   ]),   # 0: -X toe
        _vec([-half_W, length, 0.0   ]),   # 1: -X back
        _vec([-half_W, length, height]),   # 2: -X top
        _vec([+half_W, 0.0,    0.0   ]),   # 3: +X toe
        _vec([+half_W, length, 0.0   ]),   # 4: +X back
        _vec([+half_W, length, height]),   # 5: +X top
    ]
    face_vertex_counts = [3, 3, 4, 4, 4]
    face_vertex_indices = [
        0, 2, 1,        # -X end (triangle)
        3, 4, 5,        # +X end (triangle)
        0, 1, 4, 3,     # bottom (z=0)
        1, 2, 5, 4,     # back (y=length, vertical)
        0, 3, 5, 2,     # hypotenuse (inclined surface)
    ]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))])

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("convexHull")

    _tag_scenery(stage, prim_path)
    return world.scene.add(GeometryPrim(prim_path=prim_path, name=name))


def make_wall(
    world,
    prim_path: str,
    name: str,
    base_position,
    height: float = 0.4,
    width: float = 0.6,
    thickness: float = 0.02,
    color=DEFAULT_COLOR,
):
    """Vertical wall standing on the ground.

    `base_position` is the world position where the wall meets the ground
    (center of the bottom edge). The wall extends upward by `height`, along
    +X by `width`.
    """
    base = np.asarray(base_position, dtype=np.float64)
    center = base + np.array([0.0, 0.0, 0.5 * height])
    handle = world.scene.add(FixedCuboid(
        prim_path=prim_path, name=name,
        position=center,
        scale=[width, thickness, height],
        color=np.asarray(color, dtype=np.float32),
    ))
    _tag_scenery(omni.usd.get_context().get_stage(), prim_path)
    return handle


def make_platform(
    world,
    prim_path: str,
    name: str,
    base_position,
    length: float = 0.4,
    width: float = 0.4,
    height: float = 0.08,
    color=DEFAULT_COLOR,
):
    """Short pedestal — a low cuboid bodies can fall onto.

    `base_position` is the world position of the center of the platform's
    bottom face. The top face sits at z = base_position.z + height.
    """
    base = np.asarray(base_position, dtype=np.float64)
    center = base + np.array([0.0, 0.0, 0.5 * height])
    handle = world.scene.add(FixedCuboid(
        prim_path=prim_path, name=name,
        position=center,
        scale=[width, length, height],
        color=np.asarray(color, dtype=np.float32),
    ))
    _tag_scenery(omni.usd.get_context().get_stage(), prim_path)
    return handle


def make_bowl(
    world,
    prim_path: str,
    name: str,
    base_position,
    inner_length: float = 0.4,
    inner_width: float = 0.4,
    height: float = 0.15,
    wall_thickness: float = 0.02,
    color=DEFAULT_COLOR,
):
    """Open rectangular container — floor + four upright walls.

    A bowl-shaped collider built from five FixedCuboids. The interior cavity
    has dimensions (inner_width, inner_length, height); bodies dropped inside
    settle on the floor and are constrained by the walls.

    Returns a list of five handles: [floor, wall_+x, wall_-x, wall_+y, wall_-y].
    The caller is responsible for parenting these under a shared Xform if a
    single group identity is needed in segmentation.
    """
    base = np.asarray(base_position, dtype=np.float64)
    c = np.asarray(color, dtype=np.float32)
    floor_thickness = wall_thickness
    outer_length = inner_length + 2 * wall_thickness
    outer_width = inner_width + 2 * wall_thickness

    floor = world.scene.add(FixedCuboid(
        prim_path=f"{prim_path}/floor", name=f"{name}_floor",
        position=base + np.array([0.0, 0.0, 0.5 * floor_thickness]),
        scale=[outer_width, outer_length, floor_thickness],
        color=c,
    ))
    wall_z_center = floor_thickness + 0.5 * height
    wall_px = world.scene.add(FixedCuboid(
        prim_path=f"{prim_path}/wall_px", name=f"{name}_wall_px",
        position=base + np.array([0.5 * (inner_width + wall_thickness), 0.0, wall_z_center]),
        scale=[wall_thickness, outer_length, height],
        color=c,
    ))
    wall_nx = world.scene.add(FixedCuboid(
        prim_path=f"{prim_path}/wall_nx", name=f"{name}_wall_nx",
        position=base + np.array([-0.5 * (inner_width + wall_thickness), 0.0, wall_z_center]),
        scale=[wall_thickness, outer_length, height],
        color=c,
    ))
    wall_py = world.scene.add(FixedCuboid(
        prim_path=f"{prim_path}/wall_py", name=f"{name}_wall_py",
        position=base + np.array([0.0, 0.5 * (inner_length + wall_thickness), wall_z_center]),
        scale=[inner_width, wall_thickness, height],
        color=c,
    ))
    wall_ny = world.scene.add(FixedCuboid(
        prim_path=f"{prim_path}/wall_ny", name=f"{name}_wall_ny",
        position=base + np.array([0.0, -0.5 * (inner_length + wall_thickness), wall_z_center]),
        scale=[inner_width, wall_thickness, height],
        color=c,
    ))
    _tag_scenery(
        omni.usd.get_context().get_stage(),
        f"{prim_path}/floor",
        f"{prim_path}/wall_px", f"{prim_path}/wall_nx",
        f"{prim_path}/wall_py", f"{prim_path}/wall_ny",
    )
    return [floor, wall_px, wall_nx, wall_py, wall_ny]
