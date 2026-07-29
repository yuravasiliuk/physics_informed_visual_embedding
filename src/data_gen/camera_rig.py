"""CameraRig — one owner for everything camera-side (dynamic-camera.md §6).

Before this module, camera logic was smeared across four files with no owner:

    camera_paths.py   subject-anchored parametrization + basis + USD Euler
    camera_motion.py  free 6-DoF model + cross-model family dispatch
    clip_capture.py   annotator creation, prim posing, per-frame logging
    collect_clip.py   rig proposal + gating, path building, path gating

so `simulate_clip` received two parallel lists (`cam_annots`, `cam_extrinsics`)
whose element type was either a `(R, pos, q)` tuple OR a path object, and
duck-typed its way through the difference (`hasattr(e, "extrinsics")`,
`any_dynamic`, `cam_paths[c] is None`). A rig owns all of it, and the static
case becomes the zero-coefficient case *in code*, not just in the docs.

FRAME CONVENTION — read before using `extrinsics`. The API is two tiers and the
split is deliberate:

  * **transform tier** — `extrinsics()` returns `(R_camera_world,
    camera_pos_world, q_camera_world)` in the USD stage world frame, where
    `p_cam = R_camera_world @ (p_world - camera_pos_world)`. World zero is the
    stage origin: ground plane at z=0, spawn boxes centred on it. It is **GAUGE**
    — world origin (3 DoF) and yaw about gravity (1 DoF) are exactly the
    unobservable directions, the ones that leave `g_cam` invariant. Translate the
    scene 10 m and every label is bit-identical. So this tier is for posing the
    prim and projecting body state; it is NOT a supervision target.
  * **label tier** — `egomotion()` and `gravity_camera()` are frame-relative and
    well-posed. These are what a head may be trained on.

`camera.json` still stores the world transform, deliberately: it is the
transform from which window-anchored relative targets are derived at
target-build time. Provenance, never a target.
"""

import sys
import time

import numpy as np

from camera_motion import orbit_params, sample_any_path
from camera_paths import CameraPath, egomotion_labels
from clip_capture import (
    log_camera_frame,
    make_rig_annotators,
    probe_body_coverage,
    rig_angle_centers,
    rotmat_to_quat_wxyz,
    sample_camera_pose,
    set_camera_pose,
)


def _progress(msg):
    """One terse stderr line per gating step.

    Gating is the most expensive silent stretch in the pipeline: EVERY coverage
    probe is a full scene render, and a dynamic rig pays the along-path gate on
    top of the t=0 gate. With `n_cameras=7` and `camera_resample_attempts=8`
    that is up to 63 + 28 renders between "rollout starts" and the first frame,
    all previously with no output — which is indistinguishable from a hang, and
    was in fact reported as one (2026-07-28). Cost is a print per camera.

    STDOUT, not stderr: Kit routes anything on stderr through
    `omni.kit.app._impl` and labels it `[Error]`, so progress lines came back
    looking like failures. Every other print in this pipeline uses stdout.
    """
    print(msg, file=sys.stdout, flush=True)


class CameraRig:
    """Surviving cameras for one rollout: poses, paths, annotators, records.

    Construct with `solve()` (production: propose, gate, path, gate again) or
    `from_paths()` / `fixed()` (verification: an explicit camera, no gating).
    `len(rig)` is the number of cameras that survived; it may be 0.
    """

    def __init__(self, poses, paths, annotators, config):
        self.poses = list(poses)          # [(camera_pos_world, look_at_world)]
        self.paths = list(paths)          # CameraPath | FreeCameraPath
        self._annots = list(annotators)
        self.config = config
        self._records = [[] for _ in self.paths]
        # A rig is dynamic if ANY camera moves. Static rigs skip per-frame
        # posing entirely, which keeps the legacy render path byte-for-byte:
        # a static camera stays posed by `rep.create.camera(look_at=)` and is
        # never re-posed through the explicit-rotation route.
        self._dynamic = any(not p.is_static() for p in self.paths)

    # --- construction -------------------------------------------------------

    @classmethod
    def solve(cls, rng, config, centroid, cloud_radius, *, teleport=None):
        """Propose a rig, gate it, give each camera a path, gate again.

        RNG DISCIPLINE (dynamic-camera.md §4) — the two phases stay SEPARATE and
        in this order: every camera's pose is sampled first, then every camera's
        path. Interleaving them per camera would reorder the stream and change
        the corpus for a given seed. Camera draws come after all
        physics-affecting draws, so switching camera family leaves the rollout
        bit-identical — the precondition for egomotion twins.

        `teleport` is called before each coverage probe (production passes a
        closure that re-poses the bodies at the pre-roll snapshot). It exists so
        this module never touches bodies.
        """
        # Phase 1 — propose + gate the t=0 pose (camera-framing-fix.md §5).
        t0 = time.time()
        az0 = rng.uniform(0.0, 2.0 * np.pi)
        poses, probe_idx = [], 0
        for cam_i, (az_c, el_c) in enumerate(rig_angle_centers(config, az0)):
            for _attempt in range(config.camera_resample_attempts + 1):
                if teleport is not None:
                    teleport()
                pos, look = sample_camera_pose(
                    rng, config, centroid, cloud_radius, az_c, el_c)
                cov = probe_body_coverage(pos, look, config.resolution,
                                          probe_idx=probe_idx)
                probe_idx += 1
                if cov >= config.min_body_coverage:
                    poses.append((pos, look))
                    break
            _progress(f"  [rig] pose gate cam {cam_i}: {probe_idx} renders, "
                      f"{len(poses)} kept, {time.time() - t0:.0f}s")
        if not poses:
            _progress(f"  [rig] EMPTY after pose gate ({probe_idx} renders)")
            return cls([], [], [], config)

        # Phase 2 — one path per surviving camera, then the along-path gate.
        paths = [cls._sample_path(rng, config, pos, look) for pos, look in poses]
        n_pose = len(poses)
        poses, paths, n_gate = cls._gate_paths(poses, paths, config)
        _progress(f"  [rig] {len(poses)}/{config.n_cameras} cameras survive | "
                  f"{probe_idx} pose + {n_gate} path renders | "
                  f"{time.time() - t0:.0f}s | dropped {n_pose - len(poses)} on path")
        if not poses:
            return cls([], [], [], config)

        # Annotators LAST: path gating can drop a camera and the annotator list
        # must stay parallel to `poses`.
        return cls(poses, paths, make_rig_annotators(poses, config.resolution), config)

    @classmethod
    def from_paths(cls, paths, config):
        """Rig around explicit paths, no gating — verification and diagnostics."""
        poses = [(p.pose(0)[0], p.pose(0)[1]) for p in paths]
        return cls(poses, paths, make_rig_annotators(poses, config.resolution), config)

    @classmethod
    def fixed(cls, camera_position, look_at, config):
        """Single camera pinned at one pose — the legacy static rig.

        Built as a zero-coefficient `CameraPath` rather than a special case, so
        `extrinsics`/`pose_for_frame` behave identically to a moving rig. At
        roll=0 its basis reproduces `clip_capture.world_to_camera_frame`
        bit-wise (gated in verify_camera_paths.py).
        """
        az, el_deg, d = orbit_params(camera_position, look_at)
        path = CameraPath(kind="static", az0=az, el0_deg=el_deg, d0=d,
                          look0=tuple(float(x) for x in look_at),
                          dt=float(config.render_dt),
                          el_clamp_deg=tuple(config.camera_elevation_clamp_deg),
                          d_clamp=tuple(config.camera_radius_clamp))
        return cls.from_paths([path], config)

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _sample_path(rng, config, pos, look):
        families = tuple(getattr(config, "camera_path_families", ("static",)))
        weights = getattr(config, "camera_path_weights", None)
        w = None
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w = w / w.sum()
        kind = str(rng.choice(families, p=w))
        return sample_any_path(
            rng, kind, pos, look, config.frame_count, config.render_dt,
            el_clamp_deg=config.camera_elevation_clamp_deg,
            d_clamp=config.camera_radius_clamp,
            roll_always=bool(getattr(config, "camera_roll_always", False)),
            track_tau=float(getattr(config, "camera_track_tau", 0.0)),
            speed_scale=float(getattr(config, "camera_speed_scale", 1.0)),
        )

    @staticmethod
    def _gate_paths(poses, paths, config, snapshot_probe_start=1000):
        """Drop cameras whose PATH leaves the action, not just whose t=0 pose does.

        A moving camera can start framed and drift into a wall, the floor, or
        empty space, so `camera_path_gate_samples` poses spread over the clip are
        re-probed and the camera is dropped if ANY falls below
        `min_body_coverage` — quality > count, same philosophy as the t=0 gate.

        Static paths are skipped: every pose is identical, so the t=0 gate
        already settled it and re-probing would burn render products.

        PRESERVED QUIRK: unlike the t=0 gate this does NOT re-teleport the bodies
        between probes, so they drift by a frame per probe (the probe's render
        step integrates physics). Kept as-is deliberately — changing it would
        change which cameras survive, i.e. the corpus, which is a data decision
        and not a refactor's to make.

        COST: `n` full renders per MOVING camera, on top of phase 1's. This is
        the single biggest reason a dynamic batch is slower than a static one —
        a static rig short-circuits below and pays none of it. Returns the probe
        count so the caller can report it.
        """
        n = int(getattr(config, "camera_path_gate_samples", 4))
        if n <= 1:
            return poses, paths, 0
        keep_poses, keep_paths, probe_idx = [], [], snapshot_probe_start
        for (pos0, look0), path in zip(poses, paths):
            if path.is_static():
                keep_poses.append((pos0, look0))
                keep_paths.append(path)
                continue
            ts = np.linspace(0, config.frame_count - 1, n + 1)[1:].astype(int)
            ok = True
            for t in ts:
                pos, look, _ = path.pose(int(t))
                cov = probe_body_coverage(pos, look, config.resolution,
                                          probe_idx=probe_idx)
                probe_idx += 1
                if cov < config.min_body_coverage:
                    ok = False
                    break
            if ok:
                keep_poses.append((pos0, look0))
                keep_paths.append(path)
        return keep_poses, keep_paths, probe_idx - snapshot_probe_start

    # --- surface ------------------------------------------------------------

    def __len__(self):
        return len(self.paths)

    def is_empty(self):
        return not self.paths

    def is_dynamic(self):
        return self._dynamic

    def annotators(self, c):
        return self._annots[c]

    def path(self, c):
        return self.paths[c]

    # transform tier — world frame, GAUGE. See the module docstring.
    def extrinsics(self, c, frame_idx):
        R_cw, cam_pos = self.paths[c].extrinsics(frame_idx)
        return R_cw, cam_pos, rotmat_to_quat_wxyz(R_cw)

    def pose_for_frame(self, frame_idx):
        """Re-pose every moving camera for frame `frame_idx`. No-op if static.

        Call BEFORE `world.step(render=True)` so frame t renders from pose t.
        The explicit rotation (not a look-at) is mandatory or roll is silently
        dropped, and it must carry the Replicator child-prim correction — both
        handled inside `clip_capture.set_camera_pose`.
        """
        if not self._dynamic:
            return
        for c, path in enumerate(self.paths):
            R_cw, cam_pos = path.extrinsics(frame_idx)
            set_camera_pose(self._annots[c], cam_pos, R_cw)

    def log_frame(self, c, frame_idx, R_camera_world, camera_pos_world):
        """Append camera `c`'s own per-frame record. No-op for a static rig.

        Static clips write no camera.json — their extrinsics are constant and
        already in meta.json, so a per-frame file would be pure duplication.
        """
        if not self._dynamic:
            return
        log_camera_frame(self.paths[c], frame_idx, self._records[c],
                         R_camera_world, camera_pos_world)

    def camera_records(self, c):
        return self._records[c]

    def all_camera_records(self):
        return self._records

    # label tier — frame-relative and well-posed; safe to supervise.
    def egomotion(self, c, frame_idx):
        return egomotion_labels(self.paths[c], frame_idx)

    def gravity_camera(self, c, frame_idx):
        return self.paths[c].gravity_camera(frame_idx)

    def provenance(self, c):
        """meta.json `camera_path` block for camera `c`."""
        return self.paths[c].to_dict()
