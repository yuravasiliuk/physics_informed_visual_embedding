import json
import os
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BodyTrack:
    id: int
    name: str
    pos_w: np.ndarray      # (T,3) world CoM
    quat_w: np.ndarray     # (T,4) wxyz body orientation in world
    lin_vel_w: np.ndarray  # (T,3)
    ang_vel_w: np.ndarray  # (T,3)
    visible: np.ndarray    # (T,)
    intrinsics: dict       # raw spec fields


@dataclass
class RawClip:
    fps: float
    times: np.ndarray      # (T,) seconds
    cam_R_cw: np.ndarray   # (T,3,3) R_camera_world (world vec -> camera)
    cam_pos_w: np.ndarray  # (T,3)
    K: np.ndarray          # (3,3) or None
    g: np.ndarray          # (3,) gravity in world
    bodies: list


_SPEC = ("mass", "density", "restitution", "com_local_authored", "dims", "shape")


def _wxyz2R(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def _R2wxyz(R):
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def load_clip(clip_dir, g=(0.0, 0.0, -9.81)):
    meta = json.load(open(os.path.join(clip_dir, "meta.json")))
    state = json.load(open(os.path.join(clip_dir, "state.json")))
    R_cw = np.array(meta["R_camera_world"], float)
    cam_pos = np.array(meta["camera_position_world"], float)
    fps = float(meta["render_fps"])
    specs = {s["name"]: s for s in meta["bodies"]}

    by_body = {}
    for r in state:
        by_body.setdefault(r["body"], []).append(r)

    bodies = []
    for name, recs in by_body.items():
        recs = sorted(recs, key=lambda r: r["frame"])
        com_c = np.array([r["com_camera"] for r in recs], float)
        lv_c = np.array([r["lin_vel_camera"] for r in recs], float)
        av_c = np.array([r["ang_vel_camera"] for r in recs], float)
        qc = np.array([r["quat_camera_wxyz"] for r in recs], float)
        pos_w = com_c @ R_cw + cam_pos
        lin_w = lv_c @ R_cw
        ang_w = av_c @ R_cw
        quat_w = np.array([_R2wxyz(R_cw.T @ _wxyz2R(q)) for q in qc])
        spec = specs.get(name, {})
        bodies.append(BodyTrack(spec.get("id", -1), name, pos_w, quat_w, lin_w, ang_w,
                                np.ones(len(recs)), {k: spec.get(k) for k in _SPEC}))

    T = max(len(b.pos_w) for b in bodies)
    times = np.arange(T) / fps
    return RawClip(fps, times,
                   np.broadcast_to(R_cw, (T, 3, 3)).copy(),
                   np.broadcast_to(cam_pos, (T, 3)).copy(),
                   None, np.array(g, float), bodies)
