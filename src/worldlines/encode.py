from dataclasses import dataclass, field
import numpy as np
from .basis import fit_basis, fit_rotations, mats_to_quats
from .gauge import build_gauge, to_gauge
from .io import load_clip


@dataclass
class EncodeConfig:
    scheme: str = "gravity"      # metric | gravity | scene
    degree: int = 4
    drop_appearance: bool = True


@dataclass
class WorldLine:
    id: int
    name: str
    intrinsics: dict
    pos_coeffs: np.ndarray
    q0: np.ndarray
    rot_coeffs: np.ndarray
    visible: np.ndarray
    pos_err: float
    rot_err: float


@dataclass
class EgoLine:
    pos_coeffs: np.ndarray
    q0: np.ndarray
    rot_coeffs: np.ndarray
    static: bool


@dataclass
class WorldLines:
    gauge: object
    ego: EgoLine
    lines: list
    meta: dict = field(default_factory=dict)


def _norm_tau(times):
    tau = times - times[0]
    return tau / tau[-1] if tau[-1] > 0 else tau


def _intrinsics(spec, mean_mass):
    mass = spec.get("mass") or 0.0
    dims = spec.get("dims") or [1.0, 1.0, 1.0]
    radius = 0.5 * float(np.linalg.norm(dims))
    com = np.array(spec.get("com_local_authored") or [0, 0, 0], float)
    return {
        "mass_ratio": float(mass / mean_mass) if mean_mass > 0 else 1.0,
        "com_offset_over_size": float(np.linalg.norm(com) / radius) if radius > 0 else 0.0,
        "restitution": float(spec.get("restitution") or 0.0),
        "density": float(spec.get("density") or 0.0),
    }


def encode_clip(src, config=EncodeConfig()):
    raw = load_clip(src) if isinstance(src, str) else src
    gauge = build_gauge(raw, config.scheme)
    gr = to_gauge(raw, gauge)
    tau = _norm_tau(raw.times)
    mean_mass = float(np.mean([b.intrinsics.get("mass") or 0.0 for b in raw.bodies]))

    lines = []
    for b in gr.bodies:
        pc, pe = fit_basis(tau, b.pos_w, config.degree)
        q0, rc, re = fit_rotations(tau, b.quat_w, config.degree)
        lines.append(WorldLine(b.id, b.name, _intrinsics(b.intrinsics, mean_mass),
                               pc, q0, rc, b.visible, pe, re))

    static = bool(np.allclose(raw.cam_R_cw, raw.cam_R_cw[0])
                  and np.allclose(raw.cam_pos_w, raw.cam_pos_w[0]))
    epc, _ = fit_basis(tau, gr.cam_pos_w, config.degree)
    eq0, erc, _ = fit_rotations(tau, mats_to_quats(gr.cam_R_cw), config.degree)
    ego = EgoLine(epc, eq0, erc, static)

    return WorldLines(gauge, ego, lines,
                      {"fps": raw.fps, "n_frames": len(tau), "scheme": config.scheme,
                       "degree": config.degree})
