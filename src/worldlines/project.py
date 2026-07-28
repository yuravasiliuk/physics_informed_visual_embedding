import numpy as np
from .basis import reconstruct, reconstruct_rotations
from .encode import _norm_tau


def world_to_cam(p_w, R_cw, cam_pos):
    return np.einsum('tij,tj->ti', R_cw, p_w - cam_pos)


def cam_to_pixel(p_c, K):
    u = p_c @ K.T
    return u[:, :2] / u[:, 2:3]


def reconstruct_world(wl, line, tau):
    return wl.gauge.inv_point(reconstruct(tau, line.pos_coeffs))


def reproject_error(wl, raw):
    tau = _norm_tau(raw.times)
    by_name = {b.name: b for b in raw.bodies}
    errs = {}
    for line in wl.lines:
        pw = reconstruct_world(wl, line, tau)
        pc = world_to_cam(pw, raw.cam_R_cw, raw.cam_pos_w)
        oc = world_to_cam(by_name[line.name].pos_w, raw.cam_R_cw, raw.cam_pos_w)
        errs[line.name] = float(np.sqrt(np.mean((pc - oc) ** 2)))
    return errs
