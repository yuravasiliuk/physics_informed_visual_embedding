"""Read the camera prim's ACTUAL transform back from USD and compare it to ours.

Diagnoses the dynamic-camera posing bug: the clip renders sideways motion and a
wrong orientation, while every offline check passes. That combination is the
signature of a **posing** fault rather than a path-math fault —
`set_camera_pose` passes explicit world `position=`, so translation must land
where we say; if the ORIENTATION is wrong, a correct world-space forward move
projects into the image as lateral drift. One fault, two symptoms.

Why neither the labels nor the video can settle it: `camera.json` and the prim
pose are derived from the SAME `R_camera_world`, so the log looks perfect no
matter what USD did with it. The only ground truth is the stage.

What this does, in order:
  1. Creates a camera through the PRODUCTION path (`make_annotators`, which uses
     `rep.create.camera(look_at=...)`) and reads the prim's world transform.
     That path is validated end-to-end by verify_physics_labels, so it is the
     known-good reference for what our `R_camera_world` should mean.
  2. Dumps `xformOpOrder` and every op value — if `create.camera(look_at=)`
     wrote `xformOp:orient` and `modify.pose(rotation=)` appends
     `xformOp:rotateXYZ`, the final transform COMPOSES the two instead of
     replacing, which is the leading hypothesis.
  3. Poses explicitly with `set_camera_pose` and re-reads. Steps 1 and 3 must
     agree; whichever way they differ is the bug.
  4. Applies the same rotation TWICE — if the prim rotates twice as far, the op
     is composing rather than replacing, which names the fix outright.
  5. Searches the Euler-convention space: poses with each candidate angle
     encoding and reports which one actually lands on the target matrix. This
     turns the diagnosis into the fix.

No physics, no bodies, tiny render products — seconds, not minutes.

    ${ISAAC_SIM}/python.sh src/data_gen/diagnose_camera_pose.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import math          # noqa: E402
import os            # noqa: E402
import sys           # noqa: E402

ISAAC_ASSETS_DIR = os.path.expanduser("~/isaac_assets")
import carb          # noqa: E402
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/default", f"file://{ISAAC_ASSETS_DIR}"
)

import numpy as np                     # noqa: E402
import omni.replicator.core as rep     # noqa: E402
import omni.usd                        # noqa: E402
from pxr import Usd, UsdGeom           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_paths import (                                     # noqa: E402
    REPLICATOR_CAMERA_CORRECTION, look_at_basis, replicator_rotation_xyz_deg,
    usd_rotation_xyz_deg,
)
from clip_capture import make_annotators, set_camera_pose      # noqa: E402

POS = np.array([2.4, 0.0, 1.7])
LOOK = np.array([0.0, 0.0, 0.1])
RES = (64, 64)          # tiny: we never look at the pixels, only at the transform


def camera_prim(stage):
    """The rig camera prim — excluding Kit's own viewport cameras."""
    cams = [p for p in stage.Traverse()
            if p.IsA(UsdGeom.Camera) and not p.GetName().startswith("OmniverseKit_")]
    if not cams:
        raise SystemExit("no camera prim found on the stage")
    if len(cams) > 1:
        print(f"NOTE: {len(cams)} camera prims; using {cams[0].GetPath()}")
    return cams[0]


def prim_basis(prim, time=Usd.TimeCode.Default()):
    """(R_camera_world, position) as USD actually computed it.

    GfMatrix4d is row-major with the row-vector convention, so the upper-left
    3x3 read as ROWS gives the camera's local axes in world coordinates:
    row0=+X(right), row1=+Y(up), row2=+Z(backward, since USD cameras look down
    -Z). Our `R_camera_world` has exactly those rows — so this is a direct,
    convention-free comparison, not a reinterpretation.
    """
    M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    R = np.array([[float(M[i][j]) for j in range(3)] for i in range(3)])
    p = np.array([float(M[3][j]) for j in range(3)])
    norms = np.linalg.norm(R, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        print(f"NOTE: prim rows are not unit ({norms.round(4).tolist()}) — scale present")
        R = R / norms[:, None]
    return R, p


def child_correction(prim):
    """The child Camera prim's own LOCAL rotation — the frame adapter `C`.

    `rep.modify.pose` writes to the parent `Camera_Xform`; this prim sits under
    it with a fixed rotateXYZ=(90,0,90), and USD composes child-then-parent, so
    what actually renders is `E_parent @ C` in column convention. Read rather
    than hardcoded so a Replicator change shows up as a changed matrix here
    instead of as silently wrong pixels.

    TRANSPOSED on read, and that is not cosmetic. `GfMatrix4d` is row-vector
    convention: `v_parent = v_child . L`. `prim_basis` above reads the rows as-is
    because there the rows ARE the answer (camera axes in world =
    `R_camera_world` by definition). Here we want a child->parent rotation in
    COLUMN convention, which is `L_rows.T`. Reading rows here instead yields
    C-transpose — itself a valid-looking rotation matrix, which is why the first
    version of this helper reported a false mismatch against the hardcoded
    constant and fed a transposed C to the candidate search.
    """
    L = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    rows = np.array([[float(L[i][j]) for j in range(3)] for i in range(3)])
    rows = rows / np.linalg.norm(rows, axis=1)[:, None]
    C = rows.T                      # row-vector -> column-vector convention
    print(f"   child local rotation C (camera-prim frame adapter):\n"
          f"{np.array2string(C, precision=6, suppress_small=True)}")
    print(f"   matches the hardcoded REPLICATOR_CAMERA_CORRECTION: "
          f"{np.allclose(C, REPLICATOR_CAMERA_CORRECTION, atol=1e-6)}")
    return C


def dump_xform_ops(prim):
    order = prim.GetAttribute("xformOpOrder")
    names = list(order.Get()) if order and order.Get() else []
    print(f"   xformOpOrder = {names}")
    for n in names:
        a = prim.GetAttribute(n)
        print(f"      {n:28s} = {a.Get() if a else '<missing>'}")
    parent = prim.GetParent()
    if parent and parent.IsValid():
        # The ops above belong to the CHILD `Camera` prim (the fixed frame
        # adapter). `rep.modify.pose` writes to the PARENT `Camera_Xform` — that
        # is the one that changes per frame, so it is the one worth watching.
        po = parent.GetAttribute("xformOpOrder")
        pn = list(po.Get()) if po and po.Get() else []
        print(f"   parent {parent.GetPath()} xformOpOrder = {pn}")
        for n in pn:
            a = parent.GetAttribute(n)
            print(f"      {n:28s} = {a.Get() if a else '<missing>'}")


def euler_xyz_deg(M):
    """(rx,ry,rz) deg with R = Rz@Ry@Rx — USD rotateXYZ (X first), column vectors."""
    ry = math.asin(float(np.clip(-M[2, 0], -1.0, 1.0)))
    if abs(M[2, 2]) < 1e-9 and abs(M[2, 1]) < 1e-9:
        rx, rz = 0.0, math.atan2(-M[0, 1], M[1, 1])
    else:
        rx, rz = math.atan2(M[2, 1], M[2, 2]), math.atan2(M[1, 0], M[0, 0])
    return tuple(math.degrees(a) for a in (rx, ry, rz))


def euler_zyx_deg(M):
    """(rx,ry,rz) deg with R = Rx@Ry@Rz — the opposite composition order."""
    ry = math.asin(float(np.clip(M[0, 2], -1.0, 1.0)))
    if abs(M[0, 0]) < 1e-9 and abs(M[0, 1]) < 1e-9:
        rx, rz = math.atan2(M[2, 1], M[1, 1]), 0.0
    else:
        rx, rz = math.atan2(-M[1, 2], M[2, 2]), math.atan2(-M[0, 1], M[0, 0])
    return tuple(math.degrees(a) for a in (rx, ry, rz))


def pose_and_read(annots, stage, prim, position, rotation_deg):
    """Apply an explicit rotation the way set_camera_pose does, then read back."""
    with annots["_camera"]:
        rep.modify.pose(position=tuple(float(x) for x in position),
                        rotation=tuple(float(a) for a in rotation_deg))
    rep.orchestrator.step()
    return prim_basis(prim)


def err(A, B):
    return float(np.abs(np.asarray(A) - np.asarray(B)).max())


def main():
    R_ours, _ = look_at_basis(POS, LOOK, roll_rad=0.0)     # world -> camera
    M_ours = R_ours.T                                       # camera -> world
    print("target R_camera_world (rows = right, up, backward):")
    print(np.array2string(R_ours, precision=6, suppress_small=True))

    annots = make_annotators(POS, LOOK, RES, rp_name="rp_diag")
    rep.orchestrator.step()
    stage = omni.usd.get_context().get_stage()
    prim = camera_prim(stage)
    print(f"\ncamera prim: {prim.GetPath()}")

    # --- 1. the known-good reference: what look_at actually produced ---------
    R_lookat, p_lookat = prim_basis(prim)
    print("\n[1] rep.create.camera(look_at=...)  — the VALIDATED static path")
    print(np.array2string(R_lookat, precision=6, suppress_small=True))
    e = err(R_lookat, R_ours)
    print(f"    max|prim - ours| = {e:.3e}   position {p_lookat.round(4).tolist()}")
    print(f"    => our R_camera_world convention is {'CORRECT' if e < 1e-4 else 'WRONG'}")
    dump_xform_ops(prim)

    # --- 2. the suspect: explicit rotation via set_camera_pose --------------
    set_camera_pose(annots, POS, R_ours)
    rep.orchestrator.step()
    R_expl, p_expl = prim_basis(prim)
    print("\n[2] set_camera_pose(R) — the DYNAMIC path")
    print(np.array2string(R_expl, precision=6, suppress_small=True))
    e2 = err(R_expl, R_ours)
    print(f"    max|prim - ours| = {e2:.3e}   position {p_expl.round(4).tolist()}")
    print(f"    sent rotateXYZ = {tuple(round(a, 4) for a in replicator_rotation_xyz_deg(R_ours))}"
          f"   (raw/uncorrected would be "
          f"{tuple(round(a, 4) for a in usd_rotation_xyz_deg(R_ours))})")
    print(f"    => explicit posing is {'CORRECT' if e2 < 1e-4 else '*** THE BUG ***'}")
    dump_xform_ops(prim)

    # --- 3. does the op replace or compose? --------------------------------
    #  Re-applying an identical rotation must be idempotent. If the prim moves
    #  again, modify.pose is stacking onto whatever look_at already wrote.
    set_camera_pose(annots, POS, R_ours)
    rep.orchestrator.step()
    R_twice, _ = prim_basis(prim)
    drift = err(R_twice, R_expl)
    print(f"\n[3] applying the SAME rotation twice: max drift {drift:.3e}")
    print(f"    => modify.pose {'REPLACES (idempotent)' if drift < 1e-9 else '*** COMPOSES — that is the fault ***'}")

    # --- 4. which encoding actually lands on the target? --------------------
    print("\n[4] convention search — pose with each candidate, read back, compare")
    # C is read from the CHILD prim rather than hardcoded, so this stays honest
    # if a future Replicator changes the adapter.
    C = child_correction(prim)
    cands = {
        "xyz(camera->world)  [old, broken]": euler_xyz_deg(M_ours),
        "★ xyz(C @ R_cw)  [the fix]":        replicator_rotation_xyz_deg(R_ours, C),
        "xyz(world->camera)":                euler_xyz_deg(R_ours),
        "zyx(camera->world)":                euler_zyx_deg(M_ours),
        "xyz(M @ C)":                        euler_xyz_deg(M_ours @ C),
        "xyz(cam->world) reversed":          tuple(reversed(euler_xyz_deg(M_ours))),
    }
    best, best_e = None, np.inf
    for name, ang in cands.items():
        R_try, _ = pose_and_read(annots, stage, prim, POS, ang)
        e_try = err(R_try, R_ours)
        flag = "  <== MATCH" if e_try < 1e-4 else ""
        print(f"    {name:32s} {tuple(round(a,3) for a in ang)!s:36s} err {e_try:.3e}{flag}")
        if e_try < best_e:
            best, best_e = name, e_try
    print(f"\n    best: {best} (err {best_e:.3e})")
    if best_e >= 1e-4:
        print("    NO candidate matches — the fault is not the Euler encoding.\n"
              "    Most likely modify.pose is composing with the look_at orientation\n"
              "    (see [3]); the fix is to author the prim's xform directly rather\n"
              "    than layering a second rotation op on top of it.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    raise SystemExit(code)
