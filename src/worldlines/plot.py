"""Decode world-lines → spatiotemporal plots (step 2 of the GT roundtrip baseline).

Two complementary views of the same decoded `(T,3)` trajectories:
  * a 3D world-frame view (the "room") with each object's decoded path, the GT
    samples overlaid, and the camera/ego path — time shown by marker fade;
  * Minkowski-style x(t)/y(t)/z(t) panels — the literal world-line diagram,
    decoded curve vs GT dots, where the fit quality (and where contact strains
    it) is read directly.

Pure decode: positions come from `reconstruct_world` (coeffs → gauge → metric
world), so the plot is generated *from the compact representation*, not the raw
log — which is the point of the baseline.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: save to file, no display
import matplotlib.pyplot as plt

from .basis import reconstruct
from .encode import _norm_tau
from .project import reconstruct_world


def decode_ego_world(wl, tau):
    """Ego camera position in metric world frame from the ego-line coeffs."""
    return wl.gauge.inv_point(reconstruct(tau, wl.ego.pos_coeffs))


def _ensure_dir(path):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def _draw_3d(ax, wl, raw, n=240, legend=True):
    """Draw the decoded world-lines + GT + ego into a 3D axis (shared by the
    combined figure, the standalone PNG, and the spin GIF)."""
    tau_d = np.linspace(0.0, 1.0, n)
    by_name = {b.name: b for b in raw.bodies}
    cmap = plt.get_cmap("tab10")
    for k, line in enumerate(wl.lines):
        col = cmap(k % 10)
        dec = reconstruct_world(wl, line, tau_d)
        gt = by_name[line.name].pos_w
        ax.plot(dec[:, 0], dec[:, 1], dec[:, 2], "-", color=col, lw=2,
                label=f"{line.name} (pos_err {line.pos_err:.1e})")
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color=col, s=14,
                   edgecolors="k", linewidths=0.3, depthshade=False)
    ego = decode_ego_world(wl, tau_d)
    if not wl.ego.static:
        ax.plot(ego[:, 0], ego[:, 1], ego[:, 2], "--", color="0.4", lw=1.5,
                label="ego (camera)")
    ax.scatter(*raw.cam_pos_w[0], color="0.2", marker="^", s=60, label="camera@t0")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if legend:
        ax.legend(loc="upper left", fontsize=7)


def plot_worldlines(wl, raw, out_path, n=240):
    """Encode-then-decode visualization. `wl` = encoded WorldLines, `raw` = the
    source RawClip (for GT overlay). Writes a PNG to out_path; returns it."""
    tau_d = np.linspace(0.0, 1.0, n)         # dense: world-lines are resolution-free
    tau_gt = _norm_tau(raw.times)
    by_name = {b.name: b for b in raw.bodies}
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    labels = ["x(t)", "y(t)", "z(t)"]

    _draw_3d(ax3d, wl, raw, n=n)
    for k, line in enumerate(wl.lines):
        col = cmap(k % 10)
        dec = reconstruct_world(wl, line, tau_d)
        gt = by_name[line.name].pos_w
        for a in range(3):
            axes[a].plot(tau_d, dec[:, a], "-", color=col, lw=1.8)
            axes[a].scatter(tau_gt, gt[:, a], color=col, s=12,
                            edgecolors="k", linewidths=0.3, zorder=3)

    ax3d.set_title(f"world-lines (decoded) — {wl.meta['scheme']} gauge, "
                   f"deg {wl.meta['degree']}, {wl.meta['n_frames']} frames")
    for a in range(3):
        axes[a].set_ylabel(labels[a]); axes[a].grid(alpha=0.3)
    axes[0].set_title("Minkowski panels: decoded curve vs GT dots")
    axes[2].set_xlabel("τ (normalized clip time)")

    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=110)
    plt.close(fig)
    return out_path


def save_3d_png(wl, raw, out_path, n=240, elev=22, azim=-60, figsize=(9, 8), dpi=130):
    """Standalone 3D world-line view at a chosen viewing angle (hi-res, no panels)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _draw_3d(ax, wl, raw, n=n)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"world-lines (decoded) — {wl.meta['scheme']} gauge, deg {wl.meta['degree']}")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=dpi)
    plt.close(fig)
    return out_path


def spin_3d_gif(wl, raw, out_path, n=240, frames=60, elev=22, fps=20, dpi=90):
    """Rotating 360° GIF of the 3D view (imageio) — the closest to interactive
    without a display server: inspect the trajectories from every azimuth."""
    import imageio.v2 as imageio
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    _draw_3d(ax, wl, raw, n=n, legend=False)
    ax.set_title(f"world-lines — {wl.meta['scheme']} gauge, deg {wl.meta['degree']}")
    imgs = []
    for azim in np.linspace(0, 360, frames, endpoint=False):
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        imgs.append(buf[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(_ensure_dir(out_path), imgs, fps=fps, loop=0)
    return out_path
