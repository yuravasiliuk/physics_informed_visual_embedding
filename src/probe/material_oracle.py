"""Material observability oracle — are restitution / friction recoverable from
GT DYNAMICS at all?

Same method as `com_oracle` (the harness outlives its target: CoM is abandoned,
EXPERIMENTS.md Graveyard 2026-07-27), repointed at the replacement targets:

    feed ground-truth trajectories — not pixels, not features — to the same
    heads/folds, and ask whether the material parameter is decodable from the
    *physics* of these clips.

WHY THIS IS BLOCKING (physics-lora-adapter.md §4.1): `restitution_range`
(0.0–0.4) and `friction_range` (0.3–0.8) are **dynamics-changing knobs**. If the
band is too thin to write itself into the motion, no encoder work can recover it,
and the fix — widening the range — requires regenerating the whole corpus. The
CoM line already paid for "generate first, discover unobservability after GPU
spend". This run costs no Isaac and no GPU and uses labels that are ALREADY
logged (body_factory.py:457-459 → meta.json `bodies[]`).

TWO READS, because a null has two very different causes:

  1. **Decodability** (ridge / MLP R² on GT trajectory features, per scenario).
     The conservative upper bound: perfect dynamics, generic features.
  2. **Direct physics diagnostics** — the coverage question. Restitution is only
     written into a trajectory BY A BOUNCE; friction only by SLIDING. If the
     scenarios never produce those events, a null says nothing about the
     parameter and everything about the scenario mix. Diagnostics measure the
     event rate and the closed-form estimator's correlation with truth
     (e ≈ |vz_out| / |vz_in|,  μ ≈ |dv_horiz/dt| / g).

  Read (1) alone misled once on the CoM line (pilot #1's pooled composite
  averaged a null with a carrier). Read (2) is what localizes the fix:
  low R² + high event coverage → the parameter band is too thin (widen it);
  low R² + near-zero coverage → the SCENARIOS are wrong (rebuild the mix).

Diagnostics deliberately run at FULL frame resolution, not the subsampled set:
a bounce is 1-2 frames, and `--frames 32` of 64 can smear it away entirely.

Per-scenario verdicts are the deliverable, and per the plan the ranges should be
set **per scenario**: high restitution is the observable in drop/throw/collide,
but would wreck the tuned mechanisms of low-energy scenarios.

    python src/probe/material_oracle.py --data ~/data/probe_corpus \\
        --result_dir results/probe/material_oracle
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe import metrics
from probe.arm_2b import fit_mlp
from probe.com_oracle import _half_extents, _traj_features   # harness reuse, not a fork
from probe.replicate_gate_b import group_kfold

G = 9.81
# Pre-registered bar. This is an UPPER BOUND from perfect trajectories: if GT
# dynamics explain less than MIN_R2 of the variance, pixels certainly will not,
# so the knob (or the scenario) must change before any collection spend.
MIN_R2 = 0.05
# Event thresholds (m/s, m/s) — chosen well above integrator noise.
BOUNCE_VZ_IN = 0.20        # min approach speed to call a bounce
SLIDE_SPEED = 0.05         # min horizontal speed to call sliding
SLIDE_VZ = 0.05            # max |vz| while sliding (i.e. actually on the floor)
SLIP_MIN = 0.05            # min |v_h - |w|*r| to call it SLIPPING, not rolling
_ROLLERS = ("sphere", "cylinder", "capsule")   # cuboids cannot roll
# static_friction DROPPED as a target (2026-07-27): it is observable only at a
# slip/stick threshold — a separatrix-decided outcome, structurally the same
# class of quantity that cost the CoM line two pilots. It is still SAMPLED and
# LOGGED (PhysX needs it); it is simply not something we ask a model to read.
TARGETS = ("restitution", "dynamic_friction")


def _world_tracks(root, clip):
    """Per-body world-frame (z, v_world) tracks at FULL frame resolution.

    state.json stores camera-frame labels (the data-gen invariant); world is
    recoverable via meta.json `R_camera_world` + `camera_position_world`, which
    is exactly what the invariant promises. Gravity/contact geometry only makes
    sense in world frame, hence the round trip.
    """
    with open(os.path.join(root, clip, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(root, clip, "state.json")) as f:
        state = json.load(f)
    R_cw = np.asarray(meta["R_camera_world"], dtype=np.float64)
    cam_p = np.asarray(meta.get("camera_position_world", [0.0, 0.0, 0.0]), dtype=np.float64)
    fps = float(meta.get("render_fps", 30.0))

    by_body = {}
    for r in state:
        by_body.setdefault(r["body"], []).append(r)
    tracks = {}
    for name, rs in by_body.items():
        rs = sorted(rs, key=lambda r: r["frame"])
        com_c = np.array([r["com_camera"] for r in rs], dtype=np.float64)
        v_c = np.array([r["lin_vel_camera"] for r in rs], dtype=np.float64)
        w_c = np.array([r["ang_vel_camera"] for r in rs], dtype=np.float64)
        com_w = com_c @ R_cw + cam_p          # R_cw.T @ x  ==  x @ R_cw
        tracks[name] = (com_w, v_c @ R_cw, w_c @ R_cw, 1.0 / fps)
    return meta, tracks


def _bounce_stats(com_w, v_w):
    """(n_bounces, e_estimate, |vz_in| of the strongest bounce).

    A bounce is the one event that writes restitution into a trajectory:
    e = |vz_out| / |vz_in| across the contact. Returns e for the STRONGEST
    bounce (largest approach speed), or nan if the body never bounces.

    ★ THE SAMPLING FLOOR (measured 2026-07-27, the finding this oracle
    produced). State is logged once per RENDERED frame, so between two samples
    gravity injects g·Δt = 9.81/30 = 0.327 m/s into vz. A rebound only survives
    to the next sample if

        e · |vz_in|  >  g · Δt

    On the v2 corpus the MEDIAN predicted rebound was 0.178 m/s — below the
    floor — so only 33.8% of the 7257 detected bounces carried measurable
    restitution, and corr(e_est, e_true) collapsed to +0.08. Restricted to
    survivable events it recovers to +0.28 (+0.32 gravity-corrected), and
    `topple` — whose impacts are high-energy — reached +0.64 unrestricted.
    Restitution is not absent from the physics; the sampling rate erases it.
    `best_vin` is returned so main() can report the survivable fraction, which
    is the number that actually decides the knob.
    """
    vz = v_w[:, 2]
    n, best_in, best_e = 0, 0.0, np.nan
    for k in range(1, len(vz)):
        if vz[k - 1] < -BOUNCE_VZ_IN and vz[k] > 0.0:
            n += 1
            vin, vout = abs(vz[k - 1]), abs(vz[k])
            if vin > best_in:
                best_in, best_e = vin, vout / vin
    return n, best_e, (best_in if n else np.nan)


def _slide_stats(v_w, w_w, dt, radius, can_roll):
    """(n_slide_frames, mu_estimate) from horizontal deceleration while SLIPPING.

    While a body slides on the floor, Coulomb friction gives a ≈ -μg on the
    horizontal speed, so μ ≈ |dv_h/dt| / g.

    ★ ROLLING MUST BE EXCLUDED (fix 2026-07-27). The naive version — "on the
    floor and moving" — gave corr(μ_est, μ_true) = **−0.123**, i.e. backwards.
    Mechanism: PhysX rolling here is lossless (zero rolling resistance — the
    same property that made eccentric_roll's rest pose structurally rare), so a
    ROLLING body decelerates ~not at all regardless of μ. Worse, higher friction
    makes the slide→roll transition happen SOONER, so high-μ bodies spend more
    of their contact rolling ⇒ lower measured deceleration ⇒ the sign flips.
    A rolling contact carries no friction-magnitude information at all.

    Fix: require genuine SLIP. For rolling-without-slipping, v_h ≈ |ω|·r, so
    slip = |v_h − |ω|·r| separates the regimes. Cuboids cannot roll, so all of
    their floor motion counts as sliding.
    """
    vh = np.linalg.norm(v_w[:, :2], axis=1)
    on_floor = (np.abs(v_w[:, 2]) < SLIDE_VZ) & (vh > SLIDE_SPEED)
    if can_roll:
        slip = np.abs(vh - np.linalg.norm(w_w, axis=1) * radius)
        on_floor &= slip > SLIP_MIN
    idx = np.flatnonzero(on_floor)
    if len(idx) < 3:
        return int(on_floor.sum()), np.nan
    # longest contiguous sliding run — decelerations across separate contacts
    # are not comparable.
    splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    run = max(splits, key=len)
    if len(run) < 3:
        return int(on_floor.sum()), np.nan
    dv = vh[run[0]] - vh[run[-1]]
    dur = (run[-1] - run[0]) * dt
    return int(on_floor.sum()), float(dv / dur / G) if dur > 0 else np.nan


def _event_features(v_w, w_w, half, window=4):
    """Bounce-ANCHORED features: v and ω in a ±`window` frame band around the
    strongest impact, plus the approach speed and geometry.

    Why this exists (2026-07-27): the generic `_traj_features` input is a
    32-frame subsampled summary of the whole clip. Restitution is written into
    1-2 frames at the impact, so that representation smears it away — which is
    why the closed-form estimator (+0.288 above floor, +0.639 on topple) BEATS a
    5-fold MLP on 391 generic features (negative R²). This function is the
    decisive test of that explanation: same head, same folds, event-aligned
    features. If R² jumps, the quantity is event-localized and the fix is the
    TARGET/WINDOW form (contact-anchored), not the parameter range.

    No label leakage: the impact frame is located from the trajectory alone.
    Returns nan for bodies that never bounce (dropped by the keep mask).
    """
    vz = v_w[:, 2]
    k_best, vin_best = -1, 0.0
    for k in range(1, len(vz)):
        if vz[k - 1] < -BOUNCE_VZ_IN and vz[k] > 0.0 and abs(vz[k - 1]) > vin_best:
            k_best, vin_best = k, abs(vz[k - 1])
    n_feat = 6 * (2 * window + 1) + 1 + len(half)
    if k_best < 0:
        return np.full(n_feat, np.nan)
    lo, hi = k_best - window, k_best + window + 1
    if lo < 0 or hi > len(vz):                 # pad by edge-clamping the band
        idx = np.clip(np.arange(lo, hi), 0, len(vz) - 1)
    else:
        idx = np.arange(lo, hi)
    return np.concatenate([v_w[idx].reshape(-1), w_w[idx].reshape(-1),
                           [vin_best], half])


def _slip_run(v_w, w_w, radius, can_roll):
    """Longest genuine-SLIP contiguous run (indices), or None. Shared by the
    μ estimator and the slide-anchored features so both mean the same thing."""
    vh = np.linalg.norm(v_w[:, :2], axis=1)
    on = (np.abs(v_w[:, 2]) < SLIDE_VZ) & (vh > SLIDE_SPEED)
    if can_roll:
        on &= np.abs(vh - np.linalg.norm(w_w, axis=1) * radius) > SLIP_MIN
    idx = np.flatnonzero(on)
    if len(idx) < 3:
        return None
    run = max(np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1), key=len)
    return run if len(run) >= 3 else None


def _slide_features(v_w, w_w, half, radius, can_roll, window=4):
    """SLIP-anchored features — friction's event, as `_event_features` is
    restitution's.

    The impact-anchored test is not a fair test for friction: an impact is
    restitution's signature, while friction writes itself into a *sustained*
    slip (a ≈ −μg). Anchoring on the longest slip run gives friction the same
    chance restitution got. Includes the run length, because how long a body
    slips before it rolls or stops is itself μ-dependent.
    """
    n_feat = 6 * (2 * window + 1) + 2 + len(half)
    run = _slip_run(v_w, w_w, radius, can_roll)
    if run is None:
        return np.full(n_feat, np.nan)
    lo = run[0]
    idx = np.clip(np.arange(lo, lo + 2 * window + 1), 0, len(v_w) - 1)
    return np.concatenate([v_w[idx].reshape(-1), w_w[idx].reshape(-1),
                           [float(len(run)), np.linalg.norm(v_w[lo, :2])], half])


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/probe_corpus")
    p.add_argument("--result_dir", default="results/probe/material_oracle")
    p.add_argument("--frames", type=int, default=32,
                   help="subsampled frames for the DECODABILITY features "
                        "(diagnostics always use full resolution)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--no_geom", action="store_true")
    p.add_argument("--event_features", action="store_true",
                   help="replace the generic 32-frame trajectory summary with "
                        "features anchored on the strongest detected impact "
                        "(the event-localization test — see _event_features)")
    p.add_argument("--event_window", type=int, default=4)
    p.add_argument("--anchor", default="impact", choices=("impact", "slide"),
                   help="which event the features are anchored on: 'impact' is "
                        "restitution's signature, 'slide' is friction's. Each "
                        "target must be tested on ITS OWN event.")
    args = p.parse_args()
    root = os.path.expanduser(args.data)
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- rows straight from the corpus (com_oracle PILOT mode pattern) -------
    from probe.dataset import find_clip_dirs, load_clip_samples
    rows, diag, evfeat = [], [], []
    for cd in find_clip_dirs(root):
        rel = os.path.relpath(cd, root)
        _, samples = load_clip_samples(cd, n_frames=args.frames)
        if not samples:
            continue
        try:
            meta, tracks = _world_tracks(root, rel)
        except (OSError, KeyError):
            continue
        spec_by_name = {b["name"]: b for b in meta["bodies"]}
        for s in samples:
            spec = spec_by_name.get(str(s["name"]))
            if spec is None or "restitution" not in spec:
                continue                       # pre-material clips: skip loudly below
            rows.append((rel, s["name"], s["group"], s["scenario"],
                         [float(spec[t]) for t in TARGETS]))
            tr = tracks.get(str(s["name"]))
            if tr is None:
                diag.append((0, np.nan, 0, np.nan, np.nan, np.nan))
                evfeat.append(np.full(
                    6 * (2 * args.event_window + 1) + (5 if args.anchor == "slide" else 4),
                    np.nan))
                continue
            com_w, v_w, w_w, dt = tr
            nb, e_est, vin = _bounce_stats(com_w, v_w)
            # rolling radius proxy: the smallest half-extent is the axis a
            # sphere/cylinder/capsule actually rolls on.
            half = _half_extents(spec["shape"], spec["dims"])
            ns, mu_est = _slide_stats(v_w, w_w, dt, float(np.min(half)),
                                      spec["shape"] in _ROLLERS)
            diag.append((nb, e_est, ns, mu_est, vin, dt))
            evfeat.append(
                _slide_features(v_w, w_w, half, float(np.min(half)),
                                spec["shape"] in _ROLLERS, args.event_window)
                if args.anchor == "slide"
                else _event_features(v_w, w_w, half, args.event_window))
    if not rows:
        sys.exit(f"no bodies with material labels under {root} — is this a "
                 "pre-material corpus? (body_factory writes them since W2)")

    clips_a = np.asarray([r[0] for r in rows])
    names_a = np.asarray([r[1] for r in rows])
    group_a = np.asarray([r[2] for r in rows], dtype=np.int64)
    scen_a = np.asarray([r[3] for r in rows])
    Y = np.asarray([r[4] for r in rows], dtype=np.float64)
    # n_bounce, e_est, n_slide, mu_est, |vz_in|, dt
    D = np.asarray(diag, dtype=np.float64)

    if args.event_features:
        Xall = np.asarray(evfeat, dtype=np.float64)   # geometry already included
        feat_desc = (f"EVENT-ANCHORED on {args.anchor.upper()} "
                     f"(±{args.event_window} frames)")
    else:
        Xall, geom = _traj_features(root, clips_a, names_a, args.frames)
        if not args.no_geom:
            Xall = np.concatenate([Xall, geom], axis=1)
        feat_desc = f"generic {args.frames}-frame trajectory summary"
    keep = np.isfinite(Xall).all(axis=1)
    X = Xall[keep].astype(np.float32)
    Y, D = Y[keep], D[keep]
    groups, scenarios = group_a[keep], scen_a[keep]
    folds = group_kfold(groups, args.folds, args.seed)
    print(f"{keep.sum()} bodies | {len(np.unique(groups))} groups → {args.folds} "
          f"folds | feat dim {X.shape[1]} | {feat_desc} | device {device}")

    # --- READ 2 FIRST: coverage. A decodability null is uninterpretable ------
    #     without knowing whether the revealing EVENT ever happens.
    print("\n=== physics diagnostics (full frame resolution) ===")
    e_r, e_n = _corr(D[:, 1], Y[:, 0])
    mu_r, mu_n = _corr(D[:, 3], Y[:, 1])
    # ★ SAMPLING FLOOR (see _bounce_stats): a rebound is only measurable when
    #   e·|vz_in| > g·Δt. This, not event scarcity, was the v2 corpus blocker.
    dt_med = float(np.nanmedian(D[:, 5])) if np.isfinite(D[:, 5]).any() else 1 / 30.
    floor = G * dt_med
    surv = (Y[:, 0] * D[:, 4]) > floor
    e_r_s, e_n_s = _corr(D[surv, 1], Y[surv, 0])
    cover = {
        "frac_with_bounce": float(np.mean(D[:, 0] > 0)),
        "frac_with_slide": float(np.mean(D[:, 2] >= 3)),
        "mean_bounces": float(np.mean(D[:, 0])),
        "sampling_floor_mps": floor,
        "median_predicted_rebound_mps": float(np.nanmedian(Y[:, 0] * D[:, 4])),
        "frac_bounce_above_floor": float(np.mean(surv)),
        "e_est_vs_restitution_r": e_r, "e_est_n": e_n,
        "e_est_vs_restitution_r_survivable": e_r_s, "e_est_n_survivable": e_n_s,
        "mu_est_vs_dynfriction_r": mu_r, "mu_est_n": mu_n,
    }
    print(f"  bodies with >=1 bounce : {cover['frac_with_bounce']:.1%} "
          f"(mean {cover['mean_bounces']:.2f} bounces)")
    print(f"  bodies with sliding    : {cover['frac_with_slide']:.1%}")
    print(f"  ** sampling floor g*dt = {floor:.3f} m/s; median predicted rebound "
          f"e*|vz_in| = {cover['median_predicted_rebound_mps']:.3f} m/s")
    print(f"  ** rebounds ABOVE the floor (measurable at all): "
          f"{cover['frac_bounce_above_floor']:.1%}")
    print(f"  corr(e_est, restitution)      = {e_r:+.3f}  (n={e_n})")
    print(f"  corr(e_est, restitution) |above floor = {e_r_s:+.3f}  (n={e_n_s})")
    print(f"  corr(mu_est, dyn_friction)    = {mu_r:+.3f}  (n={mu_n})")

    print("\n  per-scenario event coverage:")
    scen_cover = {}
    for sc in sorted(set(scenarios.tolist())):
        m = scenarios == sc
        scen_cover[str(sc)] = {
            "n": int(m.sum()),
            "frac_bounce": float(np.mean(D[m, 0] > 0)),
            "frac_slide": float(np.mean(D[m, 2] >= 3)),
        }
        print(f"    {sc:18s} n={m.sum():4d}  bounce {scen_cover[str(sc)]['frac_bounce']:5.1%}"
              f"  slide {scen_cover[str(sc)]['frac_slide']:5.1%}")

    # --- READ 1: decodability from GT trajectories --------------------------
    r2 = {h: {t: np.full(args.folds, np.nan) for t in TARGETS} for h in ("ridge", "mlp")}
    scen_r2 = {t: {} for t in TARGETS}
    for fi, test_groups in enumerate(folds):
        is_te = np.isin(groups, test_groups)
        is_tr = ~is_te
        for ti, t in enumerate(TARGETS):
            y = Y[:, ti]
            mu_tr = float(y[is_tr].mean())
            pr, _ = metrics.fit_linear_probe(X[is_tr], y[is_tr], X[is_te], groups[is_tr])
            r2["ridge"][t][fi] = metrics.r2_aggregate(y[is_te], pr, mu_tr)
            # fit_mlp was written for the 3-vector CoM target and indexes
            # y.shape[1]; material targets are scalars → feed a column and
            # flatten back. (arm_2b stays untouched — CLAUDE.md §Principles.)
            pm = fit_mlp(X[is_tr], y[is_tr][:, None], X[is_te], groups[is_tr],
                         seed=args.seed * 1000 + fi, device=device)
            pm = np.asarray(pm).reshape(-1)
            r2["mlp"][t][fi] = metrics.r2_aggregate(y[is_te], pm, mu_tr)
            for sc in np.unique(scenarios[is_te]):
                s = scenarios[is_te] == sc
                if s.sum() >= 4:
                    scen_r2[t].setdefault(str(sc), []).append(
                        metrics.r2_aggregate(y[is_te][s], pm[s], mu_tr))
        print(f"  fold {fi}: done")

    print(f"\n{'head':6s} | " + " | ".join(f"{t:>18s}" for t in TARGETS))
    report = {"folds": args.folds, "bar": {"min_r2": MIN_R2,
              "rule": "mean - std > 0 and mean >= min_r2"},
              "coverage": cover, "per_scenario_coverage": scen_cover, "heads": {}}
    for h in r2:
        cells = {t: [float(np.nanmean(r2[h][t])), float(np.nanstd(r2[h][t]))]
                 for t in TARGETS}
        report["heads"][h] = cells
        print(f"{h:6s} | " + " | ".join(
            f"{cells[t][0]:+.4f} ± {cells[t][1]:.4f}" for t in TARGETS))

    # --- verdict: per-scenario, per-target ----------------------------------
    verdicts, scen_tbl = [], {}
    for t in TARGETS:                           # restitution + dynamic friction
        passed, failed = [], []
        scen_tbl[t] = {}
        for sc in sorted(scen_r2[t]):
            vals = np.asarray(scen_r2[t][sc], dtype=np.float64)
            m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
            ok = (m - s > 0) and (m >= MIN_R2)
            scen_tbl[t][sc] = {"mean": m, "std": s, "pass": ok,
                               "n": scen_cover.get(sc, {}).get("n", 0)}
            (passed if ok else failed).append(sc)
        verdicts.append(
            f"{t}: " + (f"OBSERVABLE in {', '.join(passed)}" if passed
                        else "NOT observable in any scenario")
            + (f"; null in {', '.join(failed)}" if failed else ""))
    report["per_scenario_mlp_r2"] = scen_tbl

    # The localization rule this oracle exists for.
    if cover["frac_with_bounce"] < 0.15:
        localize = ("SCENARIOS are the blocker for restitution: <15% of bodies "
                    "ever bounce, so the parameter is never written into the "
                    "motion. Fix the scenario mix (more drop/throw/collide, more "
                    "drop height) BEFORE widening restitution_range.")
    elif (np.isfinite(e_r_s) and e_r_s > 0.20
            and report["heads"]["mlp"]["restitution"][0] < MIN_R2
            and not args.event_features):
        # ★ The case this oracle hit on the v2 corpus (2026-07-27): a hand-
        #   computed ratio beats a 5-fold MLP on 3.5k samples. That never
        #   happens when a signal is absent — it happens when the signal is
        #   localized to 1-2 frames and the features are a clip-level summary
        #   that smears it. So the generic read is NOT an upper bound here.
        localize = (
            f"EVENT LOCALIZATION, not the parameter: the closed-form estimator "
            f"reaches r={e_r_s:+.3f} on above-floor bodies while generic "
            f"{args.frames}-frame features give R²≈"
            f"{report['heads']['mlp']['restitution'][0]:+.3f}. Restitution is "
            "written into 1-2 frames at impact, so clip-level summaries (and "
            "clip-mean encoder probes) cannot see it regardless of the range. "
            "Do NOT widen the band on this evidence. Instead: (1) re-run with "
            "--event_features to confirm; (2) log CONTACT EVENTS from PhysX and "
            "make the probe/training windows EVENT-ANCHORED — the GATE-B "
            "window-alignment lesson (+0.29) one level sharper.")
    elif cover["frac_bounce_above_floor"] < 0.5:
        localize = (
            f"SAMPLING FLOOR is the blocker: bounces are plentiful "
            f"({cover['frac_with_bounce']:.0%}) but only "
            f"{cover['frac_bounce_above_floor']:.0%} of rebounds exceed "
            f"g*dt={floor:.3f} m/s, so gravity eats them before the next logged "
            "frame. Fix in this order: (1) log CONTACT IMPULSES from PhysX "
            "(exact, sampling-independent — this is why contact logging is "
            "essential, not optional); (2) raise e*|vz_in| per scenario — widen "
            "restitution upward and/or increase impact speed/drop height, "
            "targeting e*|vz_in| > ~3*g*dt; (3) optionally log state at physics "
            "rate (60 Hz) to halve the floor.")
    elif report["heads"]["mlp"]["restitution"][0] < MIN_R2:
        localize = ("BAND is the blocker for restitution: rebounds clear the "
                    "sampling floor but the range is too thin to separate. Widen "
                    "restitution_range PER SCENARIO (high for drop/throw/collide; "
                    "leave low-energy scenarios alone).")
    else:
        localize = ("Restitution is observable as-configured — keep the range and "
                    "proceed to collection.")
    verdict = " | ".join(verdicts) + "  ->  " + localize
    print(f"\n{verdict}")
    report["verdict"] = verdict

    with open(os.path.join(result_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(result_dir, "conclusions.md"), "w") as f:
        f.write("# Material observability oracle — conclusions\n\n"
                f"{keep.sum()} bodies / {len(np.unique(groups))} groups / "
                f"{args.folds}-fold; input = GT camera-frame trajectories "
                f"(rel-position, gravity-in-body, v, ω × {args.frames} frames)"
                + ("" if args.no_geom else " + geometry") + ".\n\n"
                "## Decodability (R², upper bound from perfect dynamics)\n\n"
                "| head | " + " | ".join(TARGETS) + " |\n|---|" + "---|" * len(TARGETS) + "\n"
                + "".join(f"| {h} | " + " | ".join(
                    f"{report['heads'][h][t][0]:+.4f}±{report['heads'][h][t][1]:.4f}"
                    for t in TARGETS) + " |\n" for h in ("ridge", "mlp"))
                + "\n## Event coverage (why a null is or isn't informative)\n\n"
                f"- bodies with ≥1 bounce: **{cover['frac_with_bounce']:.1%}**\n"
                f"- bodies with sliding: **{cover['frac_with_slide']:.1%}**\n"
                f"- corr(closed-form e, true restitution): **{e_r:+.3f}** (n={e_n})\n"
                f"- corr(closed-form μ, true dynamic friction): **{mu_r:+.3f}** (n={mu_n})\n\n"
                "| scenario | n | bounce frac | slide frac |\n|---|---|---|---|\n"
                + "".join(f"| {s} | {c['n']} | {c['frac_bounce']:.1%} | "
                          f"{c['frac_slide']:.1%} |\n" for s, c in scen_cover.items())
                + f"\n## Verdict\n\n{verdict}\n")
    print(f"report.json + conclusions.md -> {result_dir}")


if __name__ == "__main__":
    main()
