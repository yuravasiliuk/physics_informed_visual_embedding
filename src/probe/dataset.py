"""Clip-dir → per-body probe samples (gate-b-probe.md §5, data contract P0.4).

Consumes the data-gen output contract written by clip_capture.save_clip:
    rgb.mp4                 (T, h, w, 3) save_resolution video
    seg.npz["seg"]          (T, h, w) uint8/16 semantic class IDs
    seg_id_to_class.json    annotator info; idToLabels: id -> {"class": label}
    state.json              flat per-(frame, body) records, camera frame
    meta.json               bodies[], R_camera_world, multiview provenance

One sample = one (clip, dynamic body): the body's per-frame seg masks at the
sampled frames, plus camera-frame targets read at the *center* sampled frame
(state read-out, not forecasting — gate-b-probe.md §2). Depth/normals unused.

Targets (gate-b-probe.md §3, all camera frame):
    v          lin_vel_camera                       — control target
    w          ang_vel_camera
    com_offset R(quat_camera_wxyz) @ com_local      — CoM − centroid, NOT
               absolute CoM (R7). com_local is the body-local offset from the
               prim origin == geometric centroid for both body kinds
               (primitives are origin-centred; W5 composites are an
               origin-centred cuboid shell with internal density splits), so
               rotating it into camera frame IS the offset target, and the
               zero vector is exactly the centroid baseline.

No torch here — pure numpy/cv2/json so `fit` can run without a GPU stack.
"""

import json
import os

import cv2
import numpy as np

# Body-kind → which meta field holds the exact body-local CoM offset
# (body_factory: primitives author it, composites derive it analytically).
_COM_FIELD = {"primitive": "com_local_authored", "composite": "com_local_analytic"}


def quat_wxyz_to_rotmat(q):
    """wxyz quaternion → 3x3 rotation matrix. Copied from
    data_gen/clip_capture.py rather than imported: that module imports omni at
    top level and the probe must never pull in Isaac (gate-b-probe.md §7)."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def bbox_diagonal(shape, dims):
    """3D bounding-box diagonal of a body, for the CoM pass bar
    "L2 < ½ median bbox diagonal" (gate-b-probe.md §3). dims layouts follow
    body_factory: cuboid (a,b,c) full extents; sphere (r,); cyl/capsule (r,h)."""
    if shape == "cuboid":
        ext = np.asarray(dims, dtype=np.float64)
    elif shape == "sphere":
        r = float(dims[0])
        ext = np.array([2 * r, 2 * r, 2 * r])
    elif shape in ("cylinder", "capsule"):
        r, h = float(dims[0]), float(dims[1])
        # Capsule total length = cylinder h + two r caps.
        full_h = h + 2 * r if shape == "capsule" else h
        ext = np.array([2 * r, 2 * r, full_h])
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return float(np.linalg.norm(ext))


def _seg_id_by_class(seg_info_path):
    """class label -> seg ID, mirroring clip_capture._body_seg_ids parsing
    (idToLabels may be a dict or a JSON string depending on Kit version)."""
    with open(seg_info_path) as f:
        info = json.load(f)
    mapping = info.get("idToLabels", info) if isinstance(info, dict) else info
    if isinstance(mapping, str):
        mapping = json.loads(mapping)
    out = {}
    for k, v in mapping.items():
        cls = v.get("class") if isinstance(v, dict) else v
        if isinstance(cls, str):
            out[cls] = int(k)
    return out


def _read_sampled_frames(video_path, frame_count, sample_idx):
    """Sequentially decode rgb.mp4, keeping only `sample_idx` frames (RGB).
    Sequential read instead of frame seeking: mp4v inter-frame seeks are
    unreliable in cv2 and the clips are short (frame_count ≈ 64)."""
    cap = cv2.VideoCapture(video_path)
    wanted = set(int(i) for i in sample_idx)
    frames = {}
    for i in range(int(frame_count)):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    missing = wanted - set(frames)
    if missing:
        raise IOError(f"{video_path}: could not decode frames {sorted(missing)}")
    return np.stack([frames[int(i)] for i in sample_idx])


def find_clip_dirs(data_root):
    """All directories under data_root containing a meta.json (clip dirs are
    nested per collect_clip's layout; just walk)."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(data_root):
        if "meta.json" in filenames:
            out.append(dirpath)
    return sorted(out)


def load_clip_samples(clip_dir, n_frames=16):
    """One clip dir → list of per-body sample dicts.

    Bodies whose seg mask is empty at every sampled frame are skipped (never
    visible → nothing to pool; the per-scenario reporting in §5 covers the
    observability question, not invisibility).

    Returns (frames, samples):
        frames  (n_frames, h, w, 3) uint8 RGB — shared by all samples
        samples [{name, kind, scenario, group, masks(n_frames,h,w) bool,
                  v(3), w(3), com_offset(3), mass, bbox_diag}]
    """
    with open(os.path.join(clip_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(clip_dir, "state.json")) as f:
        state = json.load(f)

    frame_count = int(meta["frame_count"])
    sample_idx = np.unique(np.linspace(0, frame_count - 1, n_frames).round().astype(int))
    center = int(sample_idx[len(sample_idx) // 2])

    frames = _read_sampled_frames(os.path.join(clip_dir, "rgb.mp4"), frame_count, sample_idx)
    seg = np.load(os.path.join(clip_dir, "seg.npz"))["seg"][sample_idx]
    class_to_id = _seg_id_by_class(os.path.join(clip_dir, "seg_id_to_class.json"))

    # Group key for the leakage-proof split: sibling multiview clips share one
    # physics rollout and must land on the same side (§5 — group-split on
    # rollout seed). Single-camera clips fall back to the clip seed.
    mv = meta.get("multiview")
    group = int(mv["rollout_seed"]) if mv else int(meta["seed"])
    camera_index = int(mv["camera_index"]) if mv else 0
    # Paired-intervention provenance (collect_clip --pair): twins share
    # (rollout seed, camera index, body name) and differ only in `variant`
    # ("base" vs "com0"/"omegaX"). Empty string = ordinary unpaired clip.
    variant = meta.get("pair", {}).get("variant", "")

    rec_by_body = {(r["frame"], r["body"]): r for r in state}

    samples = []
    for spec in meta["bodies"]:
        name = spec["name"]
        rec = rec_by_body.get((center, name))
        seg_id = class_to_id.get(name)
        if rec is None or seg_id is None:
            continue
        masks = seg == seg_id
        if not masks.any():
            continue  # body never visible in the sampled frames
        com_local = np.asarray(spec[_COM_FIELD[spec["kind"]]], dtype=np.float64)
        # Body-frame offset → camera frame at the center frame.
        # quat_camera_wxyz = q_camera_world ⊗ q_body_world (clip_capture
        # .log_state_frame), so R(quat) maps body-frame vectors to camera frame.
        offset_cam = quat_wxyz_to_rotmat(rec["quat_camera_wxyz"]) @ com_local
        samples.append({
            "name": name,
            "kind": spec["kind"],
            "scenario": meta["scenario"],
            "group": group,
            "camera_index": camera_index,
            "variant": variant,
            "masks": masks,
            "v": np.asarray(rec["lin_vel_camera"], dtype=np.float64),
            "w": np.asarray(rec["ang_vel_camera"], dtype=np.float64),
            "com_offset": offset_cam,
            "mass": float(rec["mass"]),
            "bbox_diag": bbox_diagonal(spec["shape"], spec["dims"]),
        })
    return frames, samples
