"""Degree-vs-fidelity sweep over the whole corpus — where is the tradeoff?

For each clip, fit world-lines at a range of Chebyshev degrees and measure how
well the *decoded* trajectory reproduces the GT, under interpretable "good"
criteria (the point being: "sufficient" depends on the threshold):

  * position: metric-world RMSE / object bbox-diagonal  (fraction of object size)
  * rotation: mean geodesic angle error                 (radians)

Reports the error DISTRIBUTION vs degree (median + p90 + max — the bounce clips
live in the tail, the mean hides them), the least degree meeting each threshold
at median and p90, and a per-scenario split (free-flight vs contact-rich). The
contact split is also the empirical test of the complex-analysis prediction:
analytic (free-flight) clips should converge geometrically to ~machine-zero at
low degree; clips with a contact corner (a real-axis singularity) should crawl
~1/N and never fully clear. CPU only; no GPU, no model.

    python -m worldlines.degree_sweep --data ~/data/sim_physics --max_degree 20
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

from .basis import reconstruct_rotations
from .encode import encode_clip, EncodeConfig, _norm_tau
from .io import load_clip
from .project import reconstruct_world


def _geodesic_angle(qa_wxyz, qb_wxyz):
    """Mean orientation error (radians) between two quat tracks (wxyz)."""
    dot = np.abs(np.sum(qa_wxyz * qb_wxyz, axis=1)).clip(0.0, 1.0)
    return float(np.mean(2.0 * np.arccos(dot)))


def sweep_clip(clip_dir, degrees, scheme="gravity"):
    """Per-body (pos_frac, rot_rad) at each degree for one clip. Returns
    (scenario, list-of-rows) where each row = (degree, pos_frac, rot_rad)."""
    raw = load_clip(clip_dir)
    tau = _norm_tau(raw.times)
    by_name = {b.name: b for b in raw.bodies}
    meta = __import__("json").load(open(os.path.join(clip_dir, "meta.json")))
    scenario = meta.get("scenario", "?")

    rows = []
    for deg in degrees:
        wl = encode_clip(raw, EncodeConfig(scheme=scheme, degree=deg))
        for line in wl.lines:
            b = by_name[line.name]
            dec = reconstruct_world(wl, line, tau)           # metric world (n,3)
            world_rmse = float(np.sqrt(np.mean(np.sum((dec - b.pos_w) ** 2, axis=1))))
            size = float(np.linalg.norm(b.intrinsics.get("dims") or [1, 1, 1]))
            pos_frac = world_rmse / size if size > 0 else np.nan
            q_dec = reconstruct_rotations(tau, line.q0, line.rot_coeffs)
            rot_rad = _geodesic_angle(q_dec, b.quat_w)
            rows.append((deg, pos_frac, rot_rad))
    return scenario, rows


def _knee(degrees, p_by_deg, thresh):
    """Least degree whose percentile error is below thresh (None if never)."""
    for d in degrees:
        if p_by_deg[d] < thresh:
            return d
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/sim_physics")
    p.add_argument("--max_degree", type=int, default=20)
    p.add_argument("--min_degree", type=int, default=2)
    p.add_argument("--scheme", default="gravity", choices=["metric", "gravity", "scene"])
    p.add_argument("--out", default="out/worldlines/degree_sweep.png")
    p.add_argument("--max_clips", type=int, default=None)
    args = p.parse_args()

    data = os.path.expanduser(args.data)
    clips = sorted(d for d in glob.glob(os.path.join(data, "*"))
                   if os.path.exists(os.path.join(d, "meta.json")))
    if args.max_clips:
        clips = clips[:args.max_clips]
    degrees = list(range(args.min_degree, args.max_degree + 1))

    # pos[deg] = list of pos_frac across all bodies; same for rot; plus per scenario.
    pos = defaultdict(list); rot = defaultdict(list)
    pos_by_scen = defaultdict(lambda: defaultdict(list))
    n_bodies = 0
    for cd in clips:
        scen, rows = sweep_clip(cd, degrees, args.scheme)
        for deg, pf, rr in rows:
            pos[deg].append(pf); rot[deg].append(rr)
            pos_by_scen[scen][deg].append(pf)
        n_bodies = max(n_bodies, len(rows) // len(degrees))
    print(f"{len(clips)} clips swept, scheme={args.scheme}, degrees {degrees[0]}..{degrees[-1]}")

    def pct(d, q, table=pos):
        return float(np.percentile(table[d], q))

    # --- table ---
    print("\ndeg | pos/size: med    p90    max   | rot(rad): med    p90    max")
    for d in degrees:
        print(f"{d:3d} | {pct(d,50):.4f} {pct(d,90):.4f} {pct(d,100):.4f} "
              f"| {pct(d,50,rot):.4f} {pct(d,90,rot):.4f} {pct(d,100,rot):.4f}")

    # --- knees ---
    print("\nleast degree meeting threshold:")
    for q, qn in ((50, "median"), (90, "p90")):
        pos_q = {d: pct(d, q) for d in degrees}
        rot_q = {d: pct(d, q, rot) for d in degrees}
        for th in (0.02, 0.05, 0.10):
            print(f"  pos/size<{th:.2f} @ {qn}: deg {_knee(degrees, pos_q, th)}")
        for th in (0.02, 0.05, 0.10):
            print(f"  rot<{th:.2f}rad @ {qn}: deg {_knee(degrees, rot_q, th)}")

    # --- per-scenario position median (the free-flight vs contact split) ---
    print("\npos/size median by scenario (convergence regime):")
    for scen in sorted(pos_by_scen):
        line = " ".join(f"{float(np.median(pos_by_scen[scen][d])):.3f}" for d in degrees)
        print(f"  {scen:18s} | {line}")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, table, ylab in ((a1, pos, "pos RMSE / object size"), (a2, rot, "rot geodesic (rad)")):
        med = [float(np.percentile(table[d], 50)) for d in degrees]
        p90 = [float(np.percentile(table[d], 90)) for d in degrees]
        ax.plot(degrees, med, "o-", label="median")
        ax.plot(degrees, p90, "s--", label="p90")
        ax.set_yscale("log"); ax.set_xlabel("Chebyshev degree"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both"); ax.legend()
    a1.set_title("position fidelity vs degree"); a2.set_title("rotation fidelity vs degree")
    fig.suptitle(f"world-line degree sweep — {len(clips)} clips, {args.scheme} gauge")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"\nplot -> {args.out}")


if __name__ == "__main__":
    main()
