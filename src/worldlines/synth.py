import numpy as np
from scipy.spatial.transform import Rotation
from .io import RawClip, BodyTrack


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array(up, float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1).T  # R_camera_world


def make_synth_clip(n_bodies=3, n_frames=24, fps=24.0, moving_camera=True, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array([0.0, 0.0, -9.81])
    t = np.arange(n_frames) / fps

    bodies = []
    for i in range(n_bodies):
        p0 = rng.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.0]) + np.array([0, 0, rng.uniform(1, 3)])
        v0 = rng.uniform(-2, 2, 3); v0[2] = rng.uniform(1, 4)
        pos = p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, g)
        omega = rng.uniform(-3, 3, 3)
        quat = Rotation.from_rotvec(np.outer(t, omega)).as_quat()[:, [3, 0, 1, 2]]
        lin = v0 + np.outer(t, g)
        ang = np.tile(omega, (n_frames, 1))
        dims = rng.uniform(0.2, 0.6, 3)
        intr = {"mass": float(rng.uniform(0.5, 3)), "density": float(rng.uniform(200, 2000)),
                "restitution": float(rng.uniform(0, 0.8)),
                "com_local_authored": rng.uniform(-0.1, 0.1, 3).tolist(),
                "dims": dims.tolist(), "shape": "cuboid"}
        bodies.append(BodyTrack(i + 1, f"body_{i}", pos, quat, lin, ang, np.ones(n_frames), intr))

    cam0 = np.array([6.0, 0.0, 2.0])
    cam_pos = cam0 + np.outer(t, rng.uniform(-0.5, 0.5, 3)) if moving_camera else np.tile(cam0, (n_frames, 1))
    R_cw = np.array([_look_at(cam_pos[k], np.zeros(3)) for k in range(n_frames)])
    K = np.array([[500.0, 0, 128], [0, 500.0, 128], [0, 0, 1.0]])
    return RawClip(fps, t, R_cw, cam_pos, K, g, bodies)
