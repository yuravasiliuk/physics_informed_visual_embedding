"""GT encode→decode→plot roundtrip on a REAL sim clip (minimal world-lines baseline).

No encoder, no learning: read the sim's world-frame state, fit world-line +
ego-line coefficients, decode them back, and plot. Establishes that the compact
representation faithfully reproduces the trajectories *before* any model predicts
the coefficients. The printed errors are the fidelity floor; the plot is the
spatiotemporal readout.

    python -m worldlines.roundtrip --clip ~/data/sim_physics/clip_00000 \\
        --degree 4 --out out/worldlines/clip_00000.png
"""

import argparse
import os

import numpy as np

from .encode import encode_clip, EncodeConfig
from .io import load_clip
from .plot import plot_worldlines, save_3d_png, spin_3d_gif
from .project import reproject_error


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="a sim clip dir (meta.json + state.json)")
    p.add_argument("--scheme", default="gravity", choices=["metric", "gravity", "scene"])
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--out", default=None, help="PNG path (default: out/worldlines/<clip>.png)")
    p.add_argument("--png3d", action="store_true",
                   help="also save a standalone hi-res 3D view (<out>_3d.png)")
    p.add_argument("--elev", type=float, default=22.0, help="3D elevation angle")
    p.add_argument("--azim", type=float, default=-60.0, help="3D azimuth angle")
    p.add_argument("--spin", action="store_true",
                   help="also save a rotating 360° GIF (<out>_spin.gif)")
    args = p.parse_args()

    clip = os.path.expanduser(args.clip)
    raw = load_clip(clip)
    wl = encode_clip(raw, EncodeConfig(scheme=args.scheme, degree=args.degree))
    rep = reproject_error(wl, raw)

    print(f"clip={os.path.basename(clip)} bodies={len(wl.lines)} "
          f"frames={wl.meta['n_frames']} scheme={args.scheme} degree={args.degree} "
          f"ego.static={wl.ego.static}")
    for ln in wl.lines:
        print(f"  {ln.name:10s} pos_err={ln.pos_err:.2e} rot_err={ln.rot_err:.2e} "
              f"reproj(cam)={rep[ln.name]:.2e} | mass_ratio={ln.intrinsics['mass_ratio']:.2f} "
              f"com/size={ln.intrinsics['com_offset_over_size']:.3f} "
              f"e={ln.intrinsics['restitution']:.2f}")
    print(f"  MEAN pos_err={np.mean([l.pos_err for l in wl.lines]):.2e} "
          f"rot_err={np.mean([l.rot_err for l in wl.lines]):.2e} "
          f"reproj={np.mean(list(rep.values())):.2e}")

    out = args.out or os.path.join("out/worldlines", os.path.basename(clip) + ".png")
    print(f"plot -> {plot_worldlines(wl, raw, out)}")
    stem = os.path.splitext(out)[0]
    if args.png3d:
        print(f"3d   -> {save_3d_png(wl, raw, stem + '_3d.png', elev=args.elev, azim=args.azim)}")
    if args.spin:
        print(f"spin -> {spin_3d_gif(wl, raw, stem + '_spin.gif', elev=args.elev)}")


if __name__ == "__main__":
    main()
