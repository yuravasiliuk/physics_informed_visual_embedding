"""Shared helpers for clip capture and save pipelines.

Extracted from `collect_clip.py` so the production driver and any
verification / debugging script can share the same frame-capture,
camera-frame conversion, state-logging, and clip-save code without one
module depending on the other.

A live `SimulationApp` must be constructed by the caller before this
module is imported: the omni / isaac imports below only resolve inside
an initialized Kit session.
"""
import json
import os

import cv2
import numpy as np

import omni.replicator.core as rep

from config import ClipConfig
# Depth (de)serialization lives in its own Isaac-free module so codecs can be
# A/B-tested offline. quantize_depth_uint16 is re-exported here (single source of
# truth) so existing `from clip_capture import quantize_depth_uint16` keeps working.
from depth_map_factory import DepthMapFactory, quantize_depth_uint16  # noqa: F401
# Pure-numpy trajectory layer (no omni deps) — dynamic-camera.md §6.
from camera_paths import egomotion_labels, look_at_basis, usd_rotation_xyz_deg

# Camera intrinsics — the single source for both the live capture cameras
# (make_annotators) and the framing-distance solver (framing_distance). Keep
# them in lock-step. tan(HFOV/2) = (horizontal_aperture/2) / focal_length.
_CAM_FOCAL_MM = 18.0
_CAM_HORIZ_APERTURE_MM = 20.955
_CAM_HALF_FOV_TAN = (_CAM_HORIZ_APERTURE_MM / 2.0) / _CAM_FOCAL_MM


def make_annotators(camera_position, look_at, resolution, rp_name=None):
    """Create one camera + render product and attach RGB/depth/normals/seg annotators.

    Segmentation uses `semantic_segmentation` (not `instance_segmentation_fast`)
    so prims sharing a `class` label collapse to one ID — step 5.8 of
    `plans/data-gen/scene-diversification.md`. Each primitive/asset body carries its
    own class (its body name) and all scenery shares `class=scenery`, so the
    annotator yields exactly the per-body-plus-scenery grouping the probe
    needs. `colorize=False` keeps the output as raw class IDs.

    `rp_name` names the render product — required to be unique when several
    cameras coexist (the multi-view rig, scene-diversification.md §10). Each
    call returns an independent annotator dict bound to its own render product,
    so `capture(annots[c])` reads only camera c's data.
    """
    camera = rep.create.camera(
        position=tuple(camera_position),
        look_at=tuple(look_at),
        focal_length=_CAM_FOCAL_MM,
        horizontal_aperture=_CAM_HORIZ_APERTURE_MM,
        clipping_range=(0.1, 100.0),
        f_stop=0.0,                  # 0 disables DoF; >0 enables defocus blur
        focus_distance=400.0,
    )

    if rp_name is None:
        rp = rep.create.render_product(camera, tuple(resolution))
    else:
        rp = rep.create.render_product(camera, tuple(resolution), name=rp_name)
    annots = {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
        "depth": rep.AnnotatorRegistry.get_annotator("distance_to_camera"),
        "normals": rep.AnnotatorRegistry.get_annotator("normals"),
        "seg": rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False},
        ),
    }
    for a in annots.values():
        a.attach(rp)
    # Keep the camera handle so the dynamic-camera path can re-pose it every
    # frame (dynamic-camera.md §6). `capture()` iterates known sensor keys, so
    # this extra entry is inert for the static path.
    annots["_camera"] = camera
    return annots


def set_camera_pose(annots, camera_position, R_camera_world):
    """Re-pose an existing rig camera for the current frame (dynamic camera).

    Takes the ROTATION MATRIX, not a look-at, for two reasons:
      * a look-at always rebuilds the zero-roll basis, which is precisely the
        degeneracy the roll term exists to break; and
      * a free 6-DoF path (camera_motion.FreeCameraPath) has an orientation that
        is not expressible as "aim at a point" at all.
    The Euler angles come from the SAME `R_camera_world` that gets logged, and
    both verifiers gate the round trip — if the two ever disagree, the pixels and
    the labels silently describe different cameras.

    Call BEFORE `world.step(render=True)` for that frame, so frame t is rendered
    from pose t.
    """
    with annots["_camera"]:
        rep.modify.pose(position=tuple(float(x) for x in camera_position),
                        rotation=usd_rotation_xyz_deg(R_camera_world))
    return R_camera_world


def framing_distance(cloud_radius, config):
    """Camera distance so a body cloud of radius `cloud_radius` fills
    `config.frame_fill_frac` of the half-frame, clamped to
    `config.camera_radius_clamp` (camera-framing-fix.md §4)."""
    lo, hi = config.camera_radius_clamp
    d = cloud_radius / max(config.frame_fill_frac * _CAM_HALF_FOV_TAN, 1e-6)
    return float(np.clip(d, lo, hi))


def rig_angle_centers(config, az0):
    """Per-camera (azimuth_rad, elevation_deg) base angles for the rig: azimuths
    evenly spaced around the full circle (globally rotated by `az0` so the rig
    isn't axis-aligned every rollout) and elevations spread across
    `camera_elevation_range`. Jitter is applied later in `sample_camera_pose`."""
    n = int(config.n_cameras)
    el_lo, el_hi = config.camera_elevation_range
    base_elev = np.linspace(el_lo, el_hi, n) if n > 1 else np.array([(el_lo + el_hi) / 2.0])
    return [(az0 + 2.0 * np.pi * k / n, float(base_elev[k])) for k in range(n)]


def sample_camera_pose(rng, config, centroid, cloud_radius, az_center, el_center_deg):
    """One jittered (camera_pos_world, look_at_world) anchored on the body-cloud
    `centroid` and framed by `cloud_radius`. Azimuth/elevation jitter the given
    band centers; elevation is clamped to keep the camera off the floor and clear
    of the near-top-down NaN (scene-diversification.md §10.2). Used both for the
    initial rig proposal and for per-camera resampling during visibility gating.
    """
    az_j = np.deg2rad(config.camera_azimuth_jitter_deg)
    el_j = config.camera_elevation_jitter_deg
    el_clamp_lo, el_clamp_hi = config.camera_elevation_clamp_deg
    az = az_center + rng.uniform(-az_j, az_j)
    el_deg = np.clip(el_center_deg + rng.uniform(-el_j, el_j), el_clamp_lo, el_clamp_hi)
    el = np.deg2rad(el_deg)
    d = framing_distance(cloud_radius, config)
    look = np.asarray(centroid, dtype=np.float64) + rng.uniform(
        -config.camera_lookat_jitter, config.camera_lookat_jitter, size=3,
    )
    offset = d * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    return look + offset, look


def _body_seg_ids(seg_info):
    """Seg IDs whose class label is a dynamic body (`body_NN`), read from a
    semantic_segmentation annotator's `info` block. `idToLabels` may be a dict
    or a JSON string depending on the Kit version; handle both."""
    if not seg_info:
        return []
    mapping = seg_info.get("idToLabels", seg_info) if isinstance(seg_info, dict) else seg_info
    if isinstance(mapping, str):
        mapping = json.loads(mapping)
    ids = []
    for k, v in mapping.items():
        cls = v.get("class") if isinstance(v, dict) else v
        if isinstance(cls, str) and cls.startswith("body_"):
            ids.append(int(k))
    return ids


def body_seg_coverage(seg_array, seg_info):
    """Fraction of pixels in `seg_array` that belong to dynamic bodies."""
    ids = _body_seg_ids(seg_info)
    if not ids:
        return 0.0
    return float(np.isin(np.asarray(seg_array), ids).mean())


def probe_body_coverage(camera_pos, look_at, resolution, probe_idx=0):
    """Render ONE frame from a candidate viewpoint and return the fraction of
    pixels covered by dynamic bodies (camera-framing-fix.md §5, Stage 2).

    Creates and tears down its own seg-only render product so it leaves no
    residue in the real capture rig. The caller is responsible for posing the
    bodies (e.g. teleporting them to the max-spread snapshot) before calling.
    """
    cam = rep.create.camera(
        position=tuple(camera_pos),
        look_at=tuple(look_at),
        focal_length=_CAM_FOCAL_MM,
        horizontal_aperture=_CAM_HORIZ_APERTURE_MM,
        clipping_range=(0.1, 100.0),
        f_stop=0.0,
        focus_distance=400.0,
    )
    rp = rep.create.render_product(cam, tuple(resolution), name=f"rp_probe_{probe_idx:04d}")
    seg = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation", init_params={"colorize": False},
    )
    seg.attach(rp)
    rep.orchestrator.step()  # render the probe frame
    data = seg.get_data()
    arr = data["data"] if isinstance(data, dict) else data
    info = data["info"] if isinstance(data, dict) else None
    cov = body_seg_coverage(arr, info)
    seg.detach()
    try:
        rp.destroy()
    except Exception:
        pass  # best-effort cleanup; a stale probe RP is harmless if reset clears it
    return cov


def make_rig_annotators(rig, resolution):
    """One independent annotator dict per rig camera (parallel to `rig`)."""
    return [
        make_annotators(pos, look, resolution, rp_name=f"rp_cam_{i:02d}")
        for i, (pos, look) in enumerate(rig)
    ]


def capture(annots):
    """Pull one frame from every annotator and return it as plain arrays."""
    rgb = annots["rgb"].get_data()           # (H, W, 4) uint8
    depth = annots["depth"].get_data()       # (H, W) float32
    normals = annots["normals"].get_data()   # (H, W, 4) float32 in [-1, 1]
    seg = annots["seg"].get_data()           # {"data": uint32 (H,W), "info": {...}} or ndarray

    seg_array = seg["data"] if isinstance(seg, dict) else seg
    seg_info = seg["info"] if isinstance(seg, dict) else None

    return {
        "rgb": rgb[..., :3].astype(np.uint8),
        "depth": depth.astype(np.float16),
        "normals": normals[..., :3].astype(np.float16),
        "seg": seg_array.astype(np.uint32),
        "seg_info": seg_info,
    }


def world_to_camera_frame(camera_pos, look_at, world_up=(0.0, 0.0, 1.0)):
    """Return (R_camera_world, camera_pos_world) for a USD/OpenGL camera.

    USD camera convention: +X right, +Y up, -Z forward (camera looks down -Z).
    R_camera_world rotates a *vector* expressed in world coords into camera coords.
    A *point* transforms as p_cam = R_camera_world @ (p_world - camera_pos_world).

    Zero-roll legacy entry point, kept verbatim for backward compatibility. The
    roll-capable superset is `camera_paths.look_at_basis(..., roll_rad=)`, and
    `verify_camera_paths.py` gates that the two agree bit-wise at roll=0 — so the
    static path is provably unchanged by the dynamic-camera work.
    """
    # Delegates to the roll-capable implementation so there is ONE basis in the
    # codebase (CLAUDE.md §Avoid code duplication). At roll_rad=0 it is the same
    # arithmetic this function used to inline, and verify_camera_paths.py check
    # #1 gates that equivalence against a verbatim copy of the old formula — so
    # the static path is provably byte-for-byte unchanged. The degenerate-camera
    # guard (critique #12) lives there now and raises the same ValueError.
    return look_at_basis(camera_pos, look_at, roll_rad=0.0, world_up=world_up)


def com_world_to_camera(com_world, R_camera_world, camera_pos_world):
    """Convert a CoM 3D point from world frame to camera frame."""
    return R_camera_world @ (np.asarray(com_world, dtype=np.float64) - camera_pos_world)


def quat_wxyz_mul(q1, q2):
    """Hamilton product of two wxyz quaternions. Returns (4,) float64."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def rotmat_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to a wxyz quaternion. Shepperd's method."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_wxyz_to_rotmat(q):
    """Convert a wxyz quaternion to a 3x3 rotation matrix (inverse of rotmat_to_quat_wxyz)."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def read_local_com(body):
    """Body-local CoM offset (3,) via SingleRigidPrim.get_com().

    Reads PhysX's rigid-body view rather than the USD MassAPI attribute:
    PhysX computes a CoM from the collision mesh for assets that ship
    without an authored `centerOfMass`, and the view exposes that computed
    value even though it never round-trips back to USD. A MassAPI-direct
    read would return zeros in that case and mis-report CoM as the prim
    origin — the exact bug step 5.7 is fixing.

    Must be called after `world.reset()` (when the physics view is created).
    Since W2 every primitive carries an authored offset CoM (scale-corrected in
    body_factory.attach_center_of_mass), so this returns that offset — the
    physical body-local CoM PhysX integrates — not zeros. It must equal the
    logged `com_local_authored` (asserted by the offset_com verify case); a
    uniform body with no authored offset would still read zeros.
    """
    com_pos, _ = body.get_com()
    return np.asarray(com_pos, dtype=np.float64).reshape(3)


def log_state_frame(bodies, local_coms, frame_idx, records, R_camera_world, camera_pos_world, q_camera_world):
    """Append one record per body for the current simulation frame.

    `local_coms[i]` is the body-local CoM offset (read once post-reset via
    read_local_com). World CoM = prim_origin + R(quat_body_world) @ com_local —
    `get_world_pose` returns the prim origin, not the CoM (settled by the W4
    offset_com verify: reconstructed CoM xy is constant under spin, no
    double-count). Since W2 com_local is the authored offset for every primitive
    (it only collapses to the prim origin for a body with no authored offset).

    All vectors and orientations are stored in **camera frame**. World frame is
    recoverable from `R_camera_world` in meta.json if downstream code needs it.
    """
    for body, com_local in zip(bodies, local_coms):
        pos_world, quat_body_world = body.get_world_pose()
        pos_world = np.asarray(pos_world, dtype=np.float64)
        quat_body_world = np.asarray(quat_body_world, dtype=np.float64)
        R_body_world = quat_wxyz_to_rotmat(quat_body_world)
        com_world = pos_world + R_body_world @ com_local
        lin_vel_world = np.asarray(body.get_linear_velocity())
        ang_vel_world = np.asarray(body.get_angular_velocity())

        records.append({
            "frame": frame_idx,
            "body": body.name,
            "com_camera": com_world_to_camera(com_world, R_camera_world, camera_pos_world).tolist(),
            "lin_vel_camera": (R_camera_world @ lin_vel_world).tolist(),
            "ang_vel_camera": (R_camera_world @ ang_vel_world).tolist(),
            "quat_camera_wxyz": quat_wxyz_mul(q_camera_world, quat_body_world).tolist(),
            "mass": float(body.get_mass()),
        })


def log_camera_frame(path, frame_idx, records, R_camera_world, camera_pos_world):
    """Append the camera's own per-frame record (dynamic-camera.md §5.1).

    "The camera is just another body": `lin_vel_camera` / `ang_vel_camera` use
    the SAME names, units and frame convention as the per-body records — the
    instantaneous camera frame — so existing target builders apply unchanged.

    Absolute pose is deliberately NOT a training label (world origin and yaw
    about gravity are gauge, unobservable from pixels). The extrinsics are
    stored because they are the *transform*, from which any well-posed relative
    target (window-anchored `R_rel`/`p_rel`, `g_camera`) is derived at
    target-build time.

    `path` is a CameraPath; for a static camera it is the zero-coefficient case
    and the velocities come out exactly 0 (gated in verify_camera_paths.py).
    """
    v_cam, w_cam = egomotion_labels(path, frame_idx)
    records.append({
        "frame": frame_idx,
        "R_camera_world": np.asarray(R_camera_world, dtype=np.float64).tolist(),
        "camera_pos_world": [float(x) for x in camera_pos_world],
        "q_camera_world": rotmat_to_quat_wxyz(R_camera_world).tolist(),
        # gravity direction in camera frame — derived, but logged because every
        # camera-frame physics judgement needs it and it is 3 floats.
        "g_camera": (np.asarray(R_camera_world, dtype=np.float64)
                     @ np.array([0.0, 0.0, -1.0])).tolist(),
        "lin_vel_camera": [float(x) for x in v_cam],
        "ang_vel_camera": [float(x) for x in w_cam],
    })


def build_metadata(config: ClipConfig, body_specs, scene_specs, scenario_name, R_camera_world, seed: int,
                   camera_position_world=None, camera_look_at_world=None, multiview=None,
                   camera_path=None):
    """Assemble one clip's meta.json.

    `camera_position_world` / `camera_look_at_world` override the config defaults
    so each fanned-out multi-view clip records its OWN camera pose (the rig pose,
    not the single-camera fallback). `multiview` is the optional provenance block
    (scene-diversification.md §8/§10) linking the n_cameras sibling clips that
    share one physics rollout. Both default to the legacy single-camera values.
    """
    cam_pos = config.camera_position if camera_position_world is None else camera_position_world
    cam_look = config.camera_look_at if camera_look_at_world is None else camera_look_at_world
    meta = {
        "schema_version": "rigid-probe/v0-draft",
        "frame_count": config.frame_count,
        "resolution": list(config.resolution),                  # render/capture res
        "save_resolution": list(config.save_resolution),        # all saved sensor channels at this res
        "render_fps": 1.0 / config.render_dt,
        "physics_hz": 1.0 / config.physics_dt,
        "seed": seed,
        "scenario": scenario_name,
        "scene": scene_specs,
        "bodies": body_specs,
        "camera_position_world": [float(x) for x in cam_pos],
        "camera_look_at_world": [float(x) for x in cam_look],
        "R_camera_world": R_camera_world.tolist(),
    }
    if multiview is not None:
        meta["multiview"] = multiview
    if camera_path is not None:
        # Provenance + the schema bump that tells loaders the extrinsics are
        # PER FRAME (camera.json), not the single meta-level matrix
        # (dynamic-camera.md §5.1). meta["R_camera_world"] stays = frame 0 so
        # static-camera consumers keep working unchanged.
        meta["camera_path"] = camera_path.to_dict()
        meta["camera_is_static"] = bool(camera_path.is_static())
        if not camera_path.is_static():
            meta["schema_version"] = "rigid-probe/v1-dyncam"
            meta["extrinsics_per_frame"] = "camera.json"
    return meta


def _resize_rgb(rgb, size):
    """Downsample an (H, W, 3) uint8 frame to (h, w). size is (width, height)."""
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _resize_depth(depth, size):
    """Downsample a depth map to (h, w) by nearest neighbor. size is (width, height).

    Nearest, NOT INTER_AREA: Isaac's distance_to_camera is exact geometric depth
    (noise-free), so averaging buys no denoising on surface interiors and instead
    invents non-physical "flying pixel" depths at occlusion edges (blending a 2 m
    foreground with a 5 m background into a 3.5 m surface that exists nowhere).
    Nearest keeps every output value a real depth and co-registers with the seg
    map (also nearest), so a pixel's seg ID and depth describe the same surface.
    """
    return cv2.resize(depth.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)


def _resize_normals(normals, size):
    """Downsample an (H, W, 3) normal map to (h, w) and renormalize to unit length.

    Averaging unit vectors (INTER_AREA) shortens them, so we renormalize after
    resize. As with depth, normals averaged across an edge are geometrically
    meaningless there; fine for a downsampled target. size is (width, height).
    """
    n = cv2.resize(normals.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.where(norm < 1e-8, 1.0, norm)


def _resize_seg(seg, size):
    """Nearest-neighbor downsample of a label map to (h, w). size is (width, height).

    Nearest (not INTER_AREA): seg values are categorical class IDs — averaging
    two IDs yields a third, wrong label. `seg` must already be uint8/uint16 (cv2
    has no uint32 resize path). Caveat: at 4x downscale a body covering only a
    few source pixels can disappear entirely.
    """
    return cv2.resize(seg, size, interpolation=cv2.INTER_NEAREST)


def quantize_normals_uint8(normals_stack):
    """Map normals in [-1, 1] to uint8 via (n + 1) / 2 * 255 (fixed, rangeless).

    ~0.3 deg worst-case angular error. Invert with: n = q.astype(float32) / 255 * 2 - 1.
    """
    return np.round((np.clip(normals_stack, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)


def save_clip(out_dir: str, frames, state_records, meta, save_resolution=(256, 256),
              depth_format="npz", camera_records=None, contact_records=None):
    """Write the clip: rgb mp4 + quantized/compressed depth & normals + seg + JSON.

    All four sensor channels are downsampled to `save_resolution` (width,
    height) before saving. rgb/normals use INTER_AREA (continuous, antialiased);
    depth/seg use INTER_NEAREST (noise-free geometry / categorical IDs, so every
    value stays physical and co-registered). depth -> uint16 (per-clip range),
    normals -> uint8 (fixed [-1,1] mapping, in normals.npz), seg -> uint8 (or
    uint16 if >255 classes), in seg.npy.

    `depth_format` selects the DepthMapFactory backend ("npz" default keeps the
    legacy depth.npz schema; "ffv1"/"hevc12" store a video — see
    depth_map_factory.py). Always writes a self-describing depth_meta.json sidecar.
    """
    os.makedirs(out_dir, exist_ok=True)
    size = (int(save_resolution[0]), int(save_resolution[1]))  # cv2 wants (width, height)

    rgb_stack = np.stack([_resize_rgb(f["rgb"], size) for f in frames])          # (T, h, w, 3) uint8
    depth_stack = np.stack([_resize_depth(f["depth"], size) for f in frames])    # (T, h, w) float32
    normals_stack = np.stack([_resize_normals(f["normals"], size) for f in frames])  # (T, h, w, 3) f32

    # Seg: downcast from the annotator's uint32 to the smallest int that holds
    # all class IDs, then nearest-neighbor resize (must cast before resize — cv2
    # has no uint32 path). ≤~10 classes here, so uint8 is expected.
    max_id = int(max(int(f["seg"].max()) for f in frames))
    assert max_id < 65536, f"seg has {max_id + 1} classes; exceeds uint16"
    seg_dtype = np.uint8 if max_id < 256 else np.uint16
    seg_stack = np.stack(
        [_resize_seg(f["seg"].astype(seg_dtype), size) for f in frames]
    )                                                                            # (T, h, w)

    height, width = rgb_stack.shape[1:3]
    video_file = os.path.join(out_dir, "rgb.mp4")
    writer = cv2.VideoWriter(
        video_file, cv2.VideoWriter_fourcc(*"mp4v"),
        float(meta["render_fps"]), (width, height),
    )
    for frame in rgb_stack:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Depth via the factory: quantize (uint16 per-clip range) + store via the
    # chosen backend + write the depth_meta.json sidecar. "npz" reproduces the
    # legacy depth.npz schema exactly; "ffv1"/"hevc12" store a video instead.
    DepthMapFactory(backend=depth_format,
                    fps=float(meta["render_fps"])).encode(out_dir, depth_stack)
    normals_q = quantize_normals_uint8(normals_stack)
    np.savez_compressed(os.path.join(out_dir, "normals.npz"), normals=normals_q)
    np.savez_compressed(os.path.join(out_dir, "seg.npz"), seg=seg_stack)

    seg_info = frames[0].get("seg_info")
    if seg_info is not None:
        with open(os.path.join(out_dir, "seg_id_to_class.json"), "w") as f:
            json.dump(seg_info, f, indent=2, default=str)

    with open(os.path.join(out_dir, "state.json"), "w") as f:
        json.dump(state_records, f, indent=2)

    # Per-frame camera track (dynamic camera). Written whenever present — a
    # static clip gets one too, with zero egomotion, so consumers have ONE code
    # path instead of branching on camera kind (dynamic-camera.md §5.1).
    if camera_records is not None:
        with open(os.path.join(out_dir, "camera.json"), "w") as f:
            json.dump(camera_records, f, indent=2)

    # Contact events (contact_log.py) — the event TIMESTAMPS that make
    # event-anchored targets constructible, plus solver-exact impulses that are
    # immune to the render-rate sampling floor.
    if contact_records is not None:
        with open(os.path.join(out_dir, "contacts.json"), "w") as f:
            json.dump(contact_records, f, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
