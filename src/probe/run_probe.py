"""GATE B arm-2a runner: layer-wise linear probe of frozen V-JEPA 2 features.

The only entry point of src/probe (gate-b-probe.md §7); never imports Isaac.
Two stages, so the GPU pass happens once and fits stay re-runnable:

    # 1. encode the corpus → cached per-(clip, body) pooled features
    python src/probe/run_probe.py extract --data ~/data/probe --out out/probe/feats.npz

    # 2. group-split, ridge per (layer, target), per-layer report
    python src/probe/run_probe.py fit --features out/probe/feats.npz \
        --report out/probe/2a_vjepa2.json

The fit stage needs no GPU/torch — only numpy + the cache.

Success/kill reading (gate-b-probe.md §9–§10): check `v` first — if the
control fails, fix the harness, do not interpret physics rows. Then read
ω R² and CoM lift on the contact-rich scenario slices, not the pooled number.
"""

import argparse
import json
import os
import sys

import numpy as np

# Allow `python src/probe/run_probe.py` without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe import metrics
from probe.dataset import find_clip_dirs, load_clip_samples

TARGETS = ("v", "w", "com_offset")
# Tokens-worth of visibility below which a sample is dropped at fit time: a
# body covering less than ~half a token at every sampled frame pools mostly
# background and would only add label noise (§5's observability question is
# handled by scenario slicing, not by keeping invisible bodies).
MIN_TOKEN_WEIGHT = 0.5


def cmd_extract(args):
    from probe.encoders import VJEPA2Encoder  # torch import only in this path

    clip_dirs = find_clip_dirs(args.data)
    if not clip_dirs:
        sys.exit(f"no clip dirs (meta.json) under {args.data}")
    enc = VJEPA2Encoder(repo=args.vjepa_repo, weights=args.weights,
                        img_size=args.img_size, device=args.device)
    print(f"{len(clip_dirs)} clips | model={enc.model_id} | device={enc.device}")

    rows = {k: [] for k in ("X", "weight", "v", "w", "com_offset", "group",
                            "scenario", "kind", "bbox_diag", "clip",
                            "name", "camera_index", "variant")}
    skipped = 0
    for n, clip_dir in enumerate(clip_dirs):
        frames, samples = load_clip_samples(clip_dir, n_frames=args.frames)
        if not samples:
            skipped += 1
            continue
        feats, weights = enc.encode_pooled(frames, [s["masks"] for s in samples])
        for s, f, wt in zip(samples, feats, weights):
            if wt <= 0.0:  # vanished under token downsampling
                continue
            rows["X"].append(f.astype(np.float16))  # (L+1, D); fp16 cache
            rows["weight"].append(wt)
            for k in TARGETS:
                rows[k].append(s[k])
            rows["group"].append(s["group"])
            rows["scenario"].append(s["scenario"])
            rows["kind"].append(s["kind"])
            rows["bbox_diag"].append(s["bbox_diag"])
            rows["clip"].append(os.path.relpath(clip_dir, args.data))
            rows["name"].append(s["name"])
            rows["camera_index"].append(s["camera_index"])
            rows["variant"].append(s["variant"])
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(clip_dirs)} clips, {len(rows['X'])} samples")

    if not rows["X"]:
        sys.exit("no usable samples extracted")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        X=np.stack(rows["X"]),
        weight=np.asarray(rows["weight"], dtype=np.float32),
        v=np.stack(rows["v"]).astype(np.float32),
        w=np.stack(rows["w"]).astype(np.float32),
        com_offset=np.stack(rows["com_offset"]).astype(np.float32),
        group=np.asarray(rows["group"], dtype=np.int64),
        scenario=np.asarray(rows["scenario"]),
        kind=np.asarray(rows["kind"]),
        bbox_diag=np.asarray(rows["bbox_diag"], dtype=np.float32),
        clip=np.asarray(rows["clip"]),
        name=np.asarray(rows["name"]),
        camera_index=np.asarray(rows["camera_index"], dtype=np.int64),
        variant=np.asarray(rows["variant"]),
        model=np.asarray(enc.model_id),
    )
    print(f"wrote {len(rows['X'])} samples x {rows['X'][0].shape} -> {args.out} "
          f"({skipped} clips had no usable bodies)")


def cmd_fit(args):
    d = np.load(args.features, allow_pickle=False)
    keep = d["weight"] >= MIN_TOKEN_WEIGHT
    X = d["X"][keep].astype(np.float32)            # (n, L+1, D)
    targets = {k: d[k][keep].astype(np.float64) for k in TARGETS}
    groups, scenarios = d["group"][keep], d["scenario"][keep]
    bbox_diag = d["bbox_diag"][keep].astype(np.float64)

    is_tr, is_te = metrics.group_split(groups, test_frac=args.test_frac)
    n_layers = X.shape[1]
    print(f"{keep.sum()}/{len(keep)} samples kept | train {is_tr.sum()} / test "
          f"{is_te.sum()} (groups: {len(np.unique(groups[is_tr]))} / "
          f"{len(np.unique(groups[is_te]))}) | {n_layers} layers")
    if is_te.sum() == 0 or is_tr.sum() == 0:
        sys.exit("degenerate split — corpus too small or single rollout group")

    t_tr = {k: v[is_tr] for k, v in targets.items()}
    t_te = {k: v[is_te] for k, v in targets.items()}
    report = {
        "model": str(d["model"]),
        "arm": "2a",
        "n_train": int(is_tr.sum()),
        "n_test": int(is_te.sum()),
        "layers": [],
    }
    for layer in range(n_layers):
        res = metrics.evaluate_layer(
            X[is_tr, layer], X[is_te, layer], t_tr, t_te,
            groups[is_tr], scenarios[is_te], bbox_diag[is_te],
        )
        res["layer"] = layer  # 0 = patch embeddings, then one per block
        report["layers"].append(res)
        print(f"layer {layer:2d} | v R2 {res['v']['r2']:+.3f} | "
              f"w R2 {res['w']['r2']:+.3f} | com lift {res['com_offset']['lift']:+.4f} "
              f"(L2 {res['com_offset']['probe_l2']:.4f} vs 1/2diag "
              f"{res['com_offset']['half_median_bbox_diag']:.4f})")

    # Best-layer summary = the GATE-B headline (§9). `v` is the harness
    # control: if it fails everywhere, the physics rows are uninterpretable.
    best = {k: max(report["layers"], key=lambda r: r[k]["r2"]) for k in TARGETS}
    report["best_layer"] = {k: int(b["layer"]) for k, b in best.items()}
    print(f"\nbest layers: v={best['v']['layer']} (R2 {best['v']['v']['r2']:+.3f}) | "
          f"w={best['w']['layer']} (R2 {best['w']['w']['r2']:+.3f}) | "
          f"com={best['com_offset']['layer']} "
          f"(lift {best['com_offset']['com_offset']['lift']:+.4f})")
    if best["v"]["v"]["r2"] <= 0.5:
        print("WARNING: v control <= 0.5 at every layer — fix the harness before "
              "reading the physics rows (gate-b-probe.md §10).")

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {args.report}")


def cmd_pairs(args):
    """Layer-wise divergence of intervention twins (collect_clip --pair).

    For each (rollout, camera, body) present in both the "base" and intervened
    variant, compare pooled features per layer: twin distance normalized by the
    same layer's unrelated-pair distance (see metrics.pair_divergence_layer for
    the reading). For com pairs, also Spearman(‖true offset‖, twin distance) —
    a layer that encodes CoM should diverge *more* for bigger offsets."""
    d = np.load(args.features, allow_pickle=False)
    if "variant" not in d.files:
        sys.exit("cache predates pair support — re-run extract")
    variant = d["variant"]
    is_base = variant == "base"
    is_twin = (variant != "base") & (variant != "")
    if not is_twin.any():
        sys.exit("no intervened variants in cache — was the corpus generated "
                 "with collect_clip --pair?")

    key = lambda idx: (int(d["group"][idx]), int(d["camera_index"][idx]),
                       str(d["name"][idx]))
    base_by_key = {key(i): i for i in np.flatnonzero(is_base)}
    pairs = [(base_by_key[key(i)], i) for i in np.flatnonzero(is_twin)
             if key(i) in base_by_key]
    if not pairs:
        sys.exit("no matchable (rollout, camera, body) twins")
    ia, ib = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    X = d["X"].astype(np.float32)
    # Dose = |‖offset_base‖ − ‖offset_twin‖|: exact for collect_pairs' graded
    # scaling (twin offset is α·base, parallel), and frame-independent — the
    # twins' center-frame orientations differ post-fork, so a camera-frame
    # vector difference would mix orientation drift into the dose.
    off = d["com_offset"].astype(np.float64)
    offset_mag = np.abs(
        np.linalg.norm(off[ia], axis=1) - np.linalg.norm(off[ib], axis=1)
    )
    pool_idx = np.flatnonzero(is_base)
    print(f"{len(pairs)} twins | variants: {sorted(set(variant[is_twin].tolist()))} "
          f"| {X.shape[1]} layers")

    rng = np.random.default_rng(0)
    report = {"model": str(d["model"]), "n_pairs": len(pairs),
              "variants": sorted(set(variant[is_twin].tolist())), "layers": []}
    for layer in range(X.shape[1]):
        res, d_pair = metrics.pair_divergence_layer(
            X[ia, layer], X[ib, layer], X[pool_idx, layer],
            d["group"][pool_idx], rng,
        )
        res["layer"] = layer
        # Only meaningful for the com intervention (offset is the dose); for
        # omega twins it's reported anyway and should hover near 0.
        res["spearman_offset_vs_divergence"] = metrics.spearman(offset_mag, d_pair)
        report["layers"].append(res)
        print(f"layer {layer:2d} | twin/unrelated ratio {res['ratio']:.3f} | "
              f"cos {res['cosine_median']:.4f} | "
              f"rho(offset,div) {res['spearman_offset_vs_divergence']:+.3f}")

    peak = max(report["layers"], key=lambda r: r["spearman_offset_vs_divergence"])
    report["peak_spearman_layer"] = int(peak["layer"])
    print(f"\npeak rho(offset, divergence): layer {peak['layer']} "
          f"({peak['spearman_offset_vs_divergence']:+.3f}) — compare against the "
          f"linear-probe best layer (§ pairs reading)")
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {args.report}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="encode corpus -> features cache")
    pe.add_argument("--data", required=True, help="root of clip dirs")
    pe.add_argument("--out", required=True, help="output .npz cache")
    pe.add_argument("--vjepa_repo", default="~/vis_emb/vjepa2",
                    help="facebookresearch/vjepa2 checkout (its src/ is imported)")
    pe.add_argument("--weights", default=None,
                    help="encoder .pt (default: <repo>/weights/vitg-384.pt)")
    pe.add_argument("--img_size", type=int, default=384,
                    help="must match the weights' training resolution")
    pe.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    pe.add_argument("--frames", type=int, default=32,
                    help="frames sampled per clip (even; tubelet-aligned; 32 "
                         "matches the notebook-validated forward)")
    pe.set_defaults(fn=cmd_extract)

    pf = sub.add_parser("fit", help="ridge per (layer, target) -> report")
    pf.add_argument("--features", required=True, help=".npz from extract")
    pf.add_argument("--report", required=True, help="output report .json")
    pf.add_argument("--test_frac", type=float, default=0.2)
    pf.set_defaults(fn=cmd_fit)

    pp = sub.add_parser("pairs", help="layer-wise intervention-twin divergence")
    pp.add_argument("--features", required=True,
                    help=".npz from extract over a --pair corpus")
    pp.add_argument("--report", required=True, help="output report .json")
    pp.set_defaults(fn=cmd_pairs)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
