"""Register-collapse diagnostic — run BEFORE believing the CoM linear-null.

Pre-registered gate (IMPLEMENTATION.md P0.5(b); vjepa-embedding-structure.md §5):
the V-JEPA 2.1 paper shows vanilla V-JEPA 2 context tokens collapse toward
register-token behaviour under exactly the linear-per-token protocol arm 2a
uses — dense/local information is smeared into near-global tokens at depth.
The 2026-07 fit_matrix run makes this check decisive: kinematics decode fine
(v_clipmean 0.60 @ L32) but CoM lift is ≈ 0 at every layer in BOTH raw and
normalized form, uniformly across scenarios (2a_v2.json). CoM needs fine
STATIC geometry, not apparent motion — it is the Tier-1 target most exposed
to register collapse, so the null is uninterpretable until this rules the
artifact in or out.

Method (label-free, no fits): for a clip subsample, pool per-body features
exactly as extract does, PLUS one all-ones mask = the whole-frame-pooled
feature — then per layer:

    cos(body, frame)   ≈ 1 at depth ⇒ an object's pooled feature is just the
                       global average — collapse signature at that layer;
    cos(body_i, body_j) same-clip body pairs — near-1 with body↔frame near-1
                       confirms tokens are register-dominated, not object-coded;
    spearman(token_weight, cos) — small bodies collapsing first is the
                       expected failure shape (fewer tokens to disagree with
                       the global mean).

Reading: the per-layer CURVE is the deliverable. High-cos plateau at the CoM
layers (~20–34) ⇒ the null is plausibly the documented artifact → swap in the
V-JEPA 2.1 checkpoint arm (drop-in, encoders.py) before any escape-hatch talk.
Low/mid cosine there ⇒ objects ARE distinguishable at depth and the CoM null
survives this check → arm 2b (nonlinear) is the remaining probe-side test.

    python src/probe/register_collapse.py --data ~/data/probe_corpus \\
        --n_clips 60 --result_dir results/probe/register_collapse
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.dataset import find_clip_dirs, load_clip_samples
from probe.metrics import spearman


def _stratified_sample(clip_dirs, root, n, seed):
    """≈ even take across scenario subdirs (first path component under root)."""
    by_scen = {}
    for d in clip_dirs:
        scen = os.path.relpath(d, root).split(os.sep)[0]
        by_scen.setdefault(scen, []).append(d)
    rng = np.random.default_rng(seed)
    for v in by_scen.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(len(v) > i for v in by_scen.values()):
        for v in by_scen.values():
            if i < len(v) and len(out) < n:
                out.append(v[i])
        i += 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/probe_corpus")
    p.add_argument("--n_clips", type=int, default=60)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--result_dir", default="results/probe/register_collapse")
    p.add_argument("--vjepa_repo", default="~/vis_emb/vjepa2")
    p.add_argument("--weights", default=None)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    root = os.path.expanduser(args.data)
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    from probe.encoders import VJEPA2Encoder  # torch only in this path
    enc = VJEPA2Encoder(repo=args.vjepa_repo, weights=args.weights,
                        img_size=args.img_size, device=args.device)

    clip_dirs = _stratified_sample(find_clip_dirs(root), root,
                                   args.n_clips, args.seed)
    print(f"{len(clip_dirs)} clips (stratified) | model {enc.model_id}")

    cos_bf, cos_bb, wts = [], [], []   # (n_bodies, L) / (n_pairs, L) / (n_bodies,)
    for k, clip_dir in enumerate(clip_dirs):
        frames, samples = load_clip_samples(clip_dir, n_frames=args.frames)
        if not samples:
            continue
        masks = [s["masks"] for s in samples]
        full = np.ones_like(masks[0])              # the whole-frame "body"
        feats, w = enc.encode_pooled(frames, masks + [full])
        body, frame = feats[:-1], feats[-1]        # (n, L, D), (L, D)
        keep = w[:-1] > 0
        body, wk = body[keep], w[:-1][keep]
        norm_b = np.linalg.norm(body, axis=2) + 1e-12
        norm_f = np.linalg.norm(frame, axis=1) + 1e-12
        cos_bf.append((body * frame[None]).sum(2) / (norm_b * norm_f[None]))
        wts.append(wk)
        for i in range(len(body)):                 # same-clip body pairs
            for j in range(i + 1, len(body)):
                cos_bb.append((body[i] * body[j]).sum(1)
                              / (norm_b[i] * norm_b[j]))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(clip_dirs)} clips")

    cos_bf = np.concatenate(cos_bf)                # (n_bodies_total, L)
    cos_bb = np.stack(cos_bb) if cos_bb else np.zeros((0, cos_bf.shape[1]))
    wts = np.concatenate(wts)
    L = cos_bf.shape[1]
    med_bf = np.median(cos_bf, axis=0)
    p25, p75 = np.percentile(cos_bf, [25, 75], axis=0)
    med_bb = np.median(cos_bb, axis=0)
    rho_w = np.array([spearman(wts, cos_bf[:, l]) for l in range(L)])

    flagged = [l for l in range(L) if med_bf[l] > 0.95]
    com_band = [l for l in range(20, min(35, L))]
    band_med = float(np.median(med_bf[com_band]))
    verdict = (
        f"COLLAPSE SIGNATURE at the CoM band: median body↔frame cosine "
        f"{band_med:.3f} over layers 20–34 — an object's pooled feature is "
        f"~the global average there; the CoM null is plausibly the documented "
        f"V-JEPA-2 register artifact → V-JEPA 2.1 checkpoint arm BEFORE any "
        f"escape-hatch decision." if band_med > 0.95 else
        f"NO collapse at the CoM band (median body↔frame cosine {band_med:.3f} "
        f"over layers 20–34): objects stay distinguishable at depth, the CoM "
        f"linear-null SURVIVES this check → arm 2b (nonlinear head) is the "
        f"remaining probe-side test before a data/observability verdict.")

    audit = {"model": enc.model_id, "n_clips": len(clip_dirs),
             "n_bodies": int(len(cos_bf)), "n_body_pairs": int(len(cos_bb)),
             "median_body_frame_cos": med_bf.tolist(),
             "p25": p25.tolist(), "p75": p75.tolist(),
             "median_body_body_cos": med_bb.tolist(),
             "spearman_weight_cos": rho_w.tolist(),
             "layers_cos_gt_095": flagged, "com_band_median": band_med}
    with open(os.path.join(result_dir, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    Ls = np.arange(L)
    a1.plot(Ls, med_bf, "-o", ms=3, label="cos(body, frame) median")
    a1.fill_between(Ls, p25, p75, alpha=0.2)
    if len(cos_bb):
        a1.plot(Ls, med_bb, "-s", ms=3, label="cos(body_i, body_j) median")
    a1.axhline(0.95, ls=":", c="crimson", lw=1, label="collapse flag 0.95")
    a1.axvspan(20, 34, color="0.9", label="CoM read band")
    a1.set_xlabel("layer"); a1.set_ylabel("cosine"); a1.legend()
    a1.set_title("object-pooled vs frame-pooled similarity by depth")
    a2.plot(Ls, rho_w, "-o", ms=3, color="C2")
    a2.axhline(0, ls="-", c="0.7", lw=0.8)
    a2.set_xlabel("layer"); a2.set_ylabel("spearman(token weight, cosine)")
    a2.set_title("do SMALL bodies collapse first? (negative = yes)")
    fig.suptitle(f"register-collapse diagnostic — {len(cos_bf)} bodies / "
                 f"{len(clip_dirs)} clips / {enc.model_id}")
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "cosine_per_layer.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(result_dir, "report.md"), "w") as f:
        f.write("# Register-collapse diagnostic\n\n"
                f"{len(cos_bf)} bodies / {len(clip_dirs)} clips / `{enc.model_id}`. "
                "Label-free: no fits, pure feature geometry.\n\n"
                f"## Verdict\n\n{verdict}\n\n"
                f"- layers with median cos > 0.95: {flagged or 'none'}\n"
                f"- CoM-band (20–34) median cos: {band_med:.3f}\n"
                f"- weight↔cos spearman at L32: {rho_w[min(32, L - 1)]:+.2f} "
                "(negative ⇒ small bodies collapse first)\n\n"
                "Curve: `cosine_per_layer.png` · raw: `audit.json` · context: "
                "[vjepa-embedding-structure.md §5](../../plans/probe/vjepa-embedding-structure.md).\n")

    print("\nmedian cos(body,frame) by layer:")
    print("  " + " ".join(f"{m:.2f}" for m in med_bf))
    print(f"\n{verdict}")
    print(f"report.md + audit.json + cosine_per_layer.png -> {result_dir}")


if __name__ == "__main__":
    main()
