"""CoM observability oracle — is the offset recoverable from GT DYNAMICS at all?

The R9 reference model, now load-bearing. After arm 2b the CoM null has
exhausted every probe-side excuse (linear, normalized, scenario-split,
register-collapse ruled out, MLP, multi-layer concat — EXPERIMENTS.md ACTIVE
node). Three hypotheses remain and this experiment splits them WITHOUT any
encoder or GPU-extract work:

    feed ground-truth trajectories — not pixels, not features — to the same
    head/folds/metrics, and ask whether CoM is decodable from the *physics*
    of these clips.

Input per body (camera frame, the 32 extract-sampled frames): CoM position
relative to its first sample (3), an orientation channel (below), linear
velocity (3), angular velocity (3). The orientation channel DEFAULTS
(2026-07-13) to the gravity direction expressed in the body frame — a unit
3-vector per frame → 12 × frames features. The raw camera-frame quaternion
mixes three nuisances into its 4 numbers (camera azimuth, the mesh's
arbitrary local axes, the q≡−q double cover); g_body is invariant to all
three and keeps exactly the tilt-vs-gravity DOFs that the contact/W6
signatures live in (eccentric-roll pendulum phase, balance lean).
--use_quaternion restores the original hemisphere-continuous quaternion
(4 → 13 × frames) for comparison with pre-2026-07-13 runs. None of
this contains the label: in free flight the CoM path is a parabola and the
tumble is inertia-driven — offset-blind; only CONTACT (pivot geometry during
topple/rock/collide) writes the offset into this signal. That is R6 stated
as an experiment.

Reading (pre-registered):
  * oracle lift clearly > 0  → the corpus DOES make CoM observable; the
    pipeline (clip-pooling → probe) loses it → per-token/temporal probe
    granularity is the justified next spend.
  * oracle lift ≈ 0          → the corpus does NOT make CoM dynamically
    visible (R6 violated in practice: no scenario puts bodies on an
    offset-decided separatrix — W3a topple overdetermines its outcome via
    the past-critical lean) → fix DATA (W6 balance / eccentric_roll) before
    any further encoder work; the encoder null was uninformative.
  * per-scenario slices: whatever signal exists should sit in the
    contact-rich scenarios; a flat slice there localizes the design gap.

PILOT MODE (the W6 acceptance-gate loop): omit --features/--targets and the
rows are built straight from the corpus (meta.json + state.json + the seg-mask
bookkeeping of dataset.load_clip_samples) — no GPU extract needed. Generate a
~20-rollout pilot of a new scenario, run this, scale only if the lift lights up.

Honest caveat: this input lacks the silhouette channel (centroid-vs-CoM
wobble is visible in video during free tumbling but not in a CoM-anchored
state log), so the oracle slightly UNDERSTATES video observability in free
flight; contact-driven observability — the R6 mechanism — is fully covered.

    python src/probe/com_oracle.py --data ~/data/probe_corpus \\
        --features results/probe/probe_corpus_feats.npz \\
        --targets results/probe/targets_v2.npz --result_dir results/probe/com_oracle
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe import metrics
from probe.arm_2b import fit_mlp
from probe.build_targets_v2 import _sample_idx
from probe.replicate_gate_b import group_kfold
from probe.run_probe import MIN_TOKEN_WEIGHT


_SHAPES = ("cuboid", "sphere", "cylinder", "capsule")

# Isaac world is z-up (clip_capture stores camera-frame labels; world stays
# recoverable via meta.json R_camera_world) → gravity points along world −z.
_G_WORLD = np.array([0.0, 0.0, -1.0])


def _quat_rotate(q, v):
    """Rotate vectors v (…,3) by unit quaternions q (…,4, wxyz), row-wise:
    v' = v + 2w(u×v) + 2u×(u×v)."""
    w, u = q[..., :1], q[..., 1:]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def _gravity_in_body(R_camera_world, q_body_to_camera):
    """Per-frame gravity direction in the BODY frame → unit (F, 3).

    quat_camera_wxyz maps body→camera (clip_capture.log_state_frame), so its
    conjugate maps camera→body; gravity in camera frame is R_camera_world @ −ẑ.
    Invariant to camera azimuth, to body spin about world-vertical and to the
    q≡−q sign (R(q) = R(−q)) — no hemisphere fix needed. What survives is the
    2-DOF tilt the W6 scenarios write the CoM offset into
    (physics-diversification.md §W6)."""
    q = np.asarray(q_body_to_camera, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)   # guard PhysX unit drift
    g_cam = np.asarray(R_camera_world, dtype=np.float64) @ _G_WORLD
    q_conj = q * np.array([1.0, -1.0, -1.0, -1.0])
    return _quat_rotate(q_conj, np.broadcast_to(g_cam, q[..., :3].shape))


def _half_extents(shape, dims):
    """Body half-extents from the spec dims. Mirrors dataset.bbox_diagonal's
    layout knowledge (copied pattern, not imported Isaac code)."""
    if shape == "cuboid":
        return 0.5 * np.asarray(dims, dtype=np.float64)
    if shape == "sphere":
        r = float(dims[0])
        return np.array([r, r, r])
    r, h = float(dims[0]), float(dims[1])
    full_h = h + 2 * r if shape == "capsule" else h
    return np.array([r, r, 0.5 * full_h])


def _traj_features(root, clips, names, frames_n, use_quaternion=False):
    """((n, per_f·F) trajectory, (n, 7) geometry) features, row-aligned.
    per_f = 12 with the default gravity-in-body orientation, 13 with
    use_quaternion=True (the pre-2026-07-13 input — see module docstring).

    Geometry = half-extents + shape one-hot — silhouette-observable quantities
    (a video model sees them too), NOT label leakage. Balance NEEDS them: the
    outcome is "tipped, given the lean vs THIS body's critical angle", and
    theta_crit is a function of the dims the bare trajectory never exposes
    (the pilot's balance-null was partly this oracle blindness)."""
    per_f = 13 if use_quaternion else 12
    rows_by_clip = {}
    for i, c in enumerate(clips):
        rows_by_clip.setdefault(str(c), []).append(i)
    F = None
    out = None
    geom = np.full((len(clips), 3 + len(_SHAPES)), np.nan, dtype=np.float64)
    for c, idxs in rows_by_clip.items():
        with open(os.path.join(root, c, "meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join(root, c, "state.json")) as f:
            state = json.load(f)
        spec_by_name = {b["name"]: b for b in meta["bodies"]}
        sidx = _sample_idx(int(meta["frame_count"]), frames_n)
        if F is None:
            F = len(sidx)
            out = np.full((len(clips), per_f * F), np.nan, dtype=np.float64)
        keep_f = set(sidx.tolist())
        recs = {}
        for r in state:
            if r["frame"] in keep_f:
                recs.setdefault(r["body"], []).append(r)
        for i in idxs:
            rs = recs.get(str(names[i]))
            if not rs or len(rs) != F:
                continue
            com = np.array([r["com_camera"] for r in rs], float)
            com -= com[0]                      # translation-invariant
            q = np.array([r["quat_camera_wxyz"] for r in rs], float)
            if use_quaternion:
                for k in range(1, F):          # hemisphere continuity
                    if (q[k] * q[k - 1]).sum() < 0:
                        q[k] = -q[k]
                if q[0, 0] < 0:
                    q = -q
                ori = q
            else:
                ori = _gravity_in_body(meta["R_camera_world"], q)
            v = np.array([r["lin_vel_camera"] for r in rs], float)
            w = np.array([r["ang_vel_camera"] for r in rs], float)
            out[i] = np.concatenate([com, ori, v, w], axis=1).reshape(-1)
            spec = spec_by_name.get(str(names[i]))
            if spec is not None:
                onehot = [1.0 if spec["shape"] == s else 0.0 for s in _SHAPES]
                geom[i] = np.concatenate(
                    [_half_extents(spec["shape"], spec["dims"]), onehot])
    return out, geom


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/probe_corpus")
    p.add_argument("--features", default=None,
                   help="feature cache — aligns rows/folds with the probe arms; "
                        "omit for PILOT mode (rows built from the corpus alone)")
    p.add_argument("--targets", default=None,
                   help="targets_v2 cache (needed only with --features)")
    p.add_argument("--result_dir", default="results/probe/com_oracle")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--no_geom", action="store_true",
                   help="drop the half-extents+shape features (reproduces the "
                        "pre-2026-07-05 trajectory-only oracle)")
    p.add_argument("--use_quaternion", action="store_true", default=False,
                   help="orientation channel = camera-frame quaternion "
                        "(4/frame; the pre-2026-07-13 input) instead of the "
                        "default gravity-in-body direction (3/frame)")
    args = p.parse_args()
    root = os.path.expanduser(args.data)
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.features:
        d = np.load(os.path.expanduser(args.features), allow_pickle=False)
        t = np.load(os.path.expanduser(args.targets), allow_pickle=False)
        if not (np.array_equal(d["clip"], t["clip"]) and np.array_equal(d["name"], t["name"])):
            sys.exit("targets not row-aligned with the cache")
        clips_a, names_a = d["clip"], d["name"]
        weight = d["weight"]
        group_a, scen_a = d["group"], d["scenario"]
        bbox_a = d["bbox_diag"].astype(np.float64)
        com_a = d["com_offset"].astype(np.float64)
        com_norm_a = t["com_norm"].astype(np.float64)
    else:
        # PILOT mode: build rows straight from the corpus via the same loader
        # the extractor uses (targets/groups/bbox identical by construction;
        # no cache, no keep-mask — every visible body counts).
        from probe.dataset import find_clip_dirs, load_clip_samples
        rows = []
        for cd in find_clip_dirs(root):
            _, samples = load_clip_samples(cd, n_frames=args.frames)
            for s in samples:
                rows.append((os.path.relpath(cd, root), s["name"], s["group"],
                             s["scenario"], s["bbox_diag"], s["com_offset"]))
        if not rows:
            sys.exit(f"no usable bodies under {root}")
        clips_a = np.asarray([r[0] for r in rows])
        names_a = np.asarray([r[1] for r in rows])
        weight = np.ones(len(rows), dtype=np.float32)
        group_a = np.asarray([r[2] for r in rows], dtype=np.int64)
        scen_a = np.asarray([r[3] for r in rows])
        bbox_a = np.asarray([r[4] for r in rows], dtype=np.float64)
        com_a = np.stack([r[5] for r in rows]).astype(np.float64)
        com_norm_a = com_a / (bbox_a[:, None] / 2.0)
        print(f"PILOT mode: {len(rows)} bodies from the corpus, no feature cache")

    Xall, geom = _traj_features(root, clips_a, names_a, args.frames,
                                use_quaternion=args.use_quaternion)
    if not args.no_geom:
        Xall = np.concatenate([Xall, geom], axis=1)
    # With a cache: SAME sample set and folds as 2a/2b — apples-to-apples.
    keep = (weight >= MIN_TOKEN_WEIGHT) & np.isfinite(Xall).all(axis=1)
    X = Xall[keep].astype(np.float32)
    groups = group_a[keep]
    scenarios = scen_a[keep]
    bbox = bbox_a[keep]
    ys = {"com_norm": com_norm_a[keep], "com_offset": com_a[keep]}
    folds = group_kfold(groups, args.folds, args.seed)
    ori_desc = "quat" if args.use_quaternion else "g_body"
    print(f"{keep.sum()} samples | {len(np.unique(groups))} groups → {args.folds} folds "
          f"| feat dim {X.shape[1]} ({'traj only' if args.no_geom else 'traj+geom'}, "
          f"ori={ori_desc}) | device {device}")

    lift = {h: {n: np.full(args.folds, np.nan) for n in ys}
            for h in ("ridge", "mlp")}
    scen_lift = {n: {} for n in ys}                 # mlp only, pooled over folds
    for fi, test_groups in enumerate(folds):
        is_te = np.isin(groups, test_groups)
        is_tr = ~is_te
        for n, y in ys.items():
            pr, _ = metrics.fit_linear_probe(X[is_tr], y[is_tr], X[is_te],
                                             groups[is_tr])
            lift["ridge"][n][fi] = metrics.com_metrics(
                y[is_te], pr, bbox[is_te])["lift"]
            pm = fit_mlp(X[is_tr], y[is_tr], X[is_te], groups[is_tr],
                         seed=args.seed * 1000 + fi, device=device)
            lift["mlp"][n][fi] = metrics.com_metrics(
                y[is_te], pm, bbox[is_te])["lift"]
            for scen in np.unique(scenarios[is_te]):
                s = scenarios[is_te] == scen
                scen_lift[n].setdefault(str(scen), []).append(
                    metrics.com_metrics(y[is_te][s], pm[s], bbox[is_te][s])["lift"])
        print(f"  fold {fi}: done")

    def ms(a):
        return float(np.nanmean(a)), float(np.nanstd(a))

    print(f"\n{'head':6s} | " + " | ".join(f"{n:>20s}" for n in ys))
    report = {"folds": args.folds,
              "orientation": "quaternion" if args.use_quaternion
              else "gravity_in_body",
              "heads": {}}
    for h in lift:
        cells = {}
        for n in ys:
            m, s = ms(lift[h][n])
            cells[n] = [m, s]
        report["heads"][h] = cells
        print(f"{h:6s} | " + " | ".join(
            f"{cells[n][0]:+.4f} ± {cells[n][1]:.4f}" for n in ys))
    # Per-scenario verdicts — THE deliverable. Pre-registered bar (2026-07-05):
    # a scenario passes iff fold-mean − fold-std > 0 AND mean ≥ MIN_LIFT (well
    # above the v2 corpus's ±0.002 noise floor). The POOLED number is reported
    # for reference only: it averages nulls with carriers and misled once
    # (pilot #1: +0.0093 pooled = one null + one +0.0196 scenario).
    MIN_LIFT = 0.005
    print("\nper-scenario MLP lift (com_norm), fold mean ± std:")
    scen_tbl = {}
    passed, failed = [], []
    for scen in sorted(scen_lift["com_norm"]):
        vals = np.asarray(scen_lift["com_norm"][scen], dtype=np.float64)
        m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
        nb = int((scenarios == scen).sum())
        ok = (m - s > 0) and (m >= MIN_LIFT)
        scen_tbl[scen] = {"mean": m, "std": s, "n_bodies": nb, "pass": ok}
        (passed if ok else failed).append(scen)
        print(f"  {scen:18s} {m:+.4f} ± {s:.4f}  (n={nb})  "
              f"{'PASS' if ok else 'fail'}")
    report["per_scenario_mlp_com_norm"] = scen_tbl
    report["bar"] = {"min_lift": MIN_LIFT, "rule": "mean - std > 0 and mean >= min_lift"}

    om, osd = report["heads"]["mlp"]["com_norm"]
    verdict = (f"Per-scenario verdict (bar: mean−std>0 and mean≥{MIN_LIFT}): "
               + (f"SCALE {', '.join(passed)}" if passed else "SCALE nothing")
               + (f" · REDESIGN/investigate {', '.join(failed)}" if failed else "")
               + f". Pooled MLP lift {om:+.4f}±{osd:.4f} (reference only — do "
               f"not read a pooled positive as corpus-wide observability).")
    print(f"\n{verdict}")

    report["verdict"] = verdict
    with open(os.path.join(result_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(result_dir, "conclusions.md"), "w") as f:
        f.write("# CoM observability oracle (R9 reference) — conclusions\n\n"
                f"{keep.sum()} samples / {len(np.unique(groups))} groups / "
                f"{args.folds}-fold; input = GT camera-frame trajectories "
                f"(rel-CoM, {'quat' if args.use_quaternion else 'gravity-in-body'}, "
                f"v, ω × {args.frames} frames)"
                + ("" if args.no_geom else " + geometry (half-extents, shape)")
                + "; same folds/metrics as arms 2a/2b.\n\n"
                "| head | com_norm lift | com_offset lift |\n|---|---|---|\n"
                + "".join(f"| {h} | {report['heads'][h]['com_norm'][0]:+.4f}"
                          f"±{report['heads'][h]['com_norm'][1]:.4f} | "
                          f"{report['heads'][h]['com_offset'][0]:+.4f}"
                          f"±{report['heads'][h]['com_offset'][1]:.4f} |\n"
                          for h in ("ridge", "mlp"))
                + "\n## Per-scenario (MLP, com_norm — the deliverable)\n\n"
                "| scenario | lift (fold mean ± std) | n bodies | verdict |\n"
                "|---|---|---|---|\n"
                + "".join(f"| {s} | {t['mean']:+.4f} ± {t['std']:.4f} | "
                          f"{t['n_bodies']} | {'PASS' if t['pass'] else 'fail'} |\n"
                          for s, t in scen_tbl.items())
                + f"\n## Verdict\n\n{verdict}\n\n"
                "Caveat: input lacks the silhouette (centroid-wobble) channel, "
                "so free-flight observability is understated; contact-driven "
                "observability (the R6 mechanism) is fully represented.\n")
    print(f"report.json + conclusions.md -> {result_dir}")


if __name__ == "__main__":
    main()
