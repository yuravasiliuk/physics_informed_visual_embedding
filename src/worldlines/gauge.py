from dataclasses import dataclass
import numpy as np
from .io import BodyTrack, RawClip


@dataclass
class Gauge:
    R: np.ndarray  # world -> gauge rotation
    t: np.ndarray  # gauge origin in world
    L: float       # length unit
    T: float       # time unit

    def point(self, p):
        return ((np.atleast_2d(p) - self.t) @ self.R.T) / self.L

    def inv_point(self, pg):
        return (np.atleast_2d(pg) * self.L) @ self.R + self.t

    def vec(self, v):
        return (np.atleast_2d(v) @ self.R.T) * (self.T / self.L)

    def rate(self, w):
        return (np.atleast_2d(w) @ self.R.T) * self.T


def _align_to_down(g):
    gn = np.asarray(g, float) / np.linalg.norm(g)
    down = np.array([0.0, 0.0, -1.0])
    v = np.cross(gn, down)
    s = np.linalg.norm(v)
    c = float(gn @ down)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def build_gauge(raw, scheme="gravity"):
    R = _align_to_down(raw.g)
    centroid = np.mean([b.pos_w[0] for b in raw.bodies], axis=0)
    dur = float(raw.times[-1] - raw.times[0]) or 1.0
    if scheme == "metric":
        return Gauge(R, np.zeros(3), 1.0, 1.0)
    if scheme == "gravity":
        return Gauge(R, centroid, float(np.linalg.norm(raw.g)) * dur ** 2, dur)
    if scheme == "scene":
        pts = np.array([b.pos_w[0] for b in raw.bodies])
        L = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
        return Gauge(R, centroid, L, dur)
    raise ValueError(scheme)


def to_gauge(raw, gauge):
    bodies = [BodyTrack(b.id, b.name, gauge.point(b.pos_w), b.quat_w,
                        gauge.vec(b.lin_vel_w), gauge.rate(b.ang_vel_w),
                        b.visible, b.intrinsics)
              for b in raw.bodies]
    R_cw_g = np.einsum('tij,kj->tik', raw.cam_R_cw, gauge.R)  # R_cw @ R_g^T
    return RawClip(raw.fps, raw.times, R_cw_g, gauge.point(raw.cam_pos_w),
                   raw.K, raw.g, bodies)
