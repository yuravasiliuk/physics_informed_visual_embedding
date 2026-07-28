"""Corpus acceptance audit — the E1 pre-GPU gate on a freshly generated corpus.

First check on a new probe corpus (probe-corpus-spec.md §6; the EXPERIMENTS.md
ACTIVE node), run BEFORE spending the GPU extraction pass. CPU-only: reads
meta.json + state.json, never decodes video and never touches the feature
cache. The three audits split cleanly and do not overlap:

    corpus_acceptance   structure — contract files, validation flags, group
                        uniqueness, scenario balance, batch/config homogeneity,
                        settled fraction, keep-rate   (this file, pre-GPU)
    density_size_audit  latent-physics confounds off meta.json (CPU, pre-GPU)
    coverage_audit      target range/confound off the feature cache (post-extract)

HARD FAIL (fix before extract):
  * duplicate rollout-group IDs — two clips claiming one physics rollout would
    leak through the group split (gate-b-probe.md §5) and fake a pass;
  * required contract files missing from a clip that extract would ingest.
SOFT FLAGS (recorded; excluded or interpreted, not fatal):
  * validation.ok=False clips → written to exclude_clips.txt (move them out of
    the corpus root before extract — find_clip_dirs walks everything);
  * orphan clip_* dirs without meta.json (crash residue; invisible to extract);
  * scenario group-starvation / imbalance (the exact failure the corpus was
    regenerated to fix — EXPERIMENTS.md "group-starvation" node);
  * mixed generation configs across batches (this corpus is known to be a
    June-25 + July-3 + July-4 mix; resolution/fps drift would contaminate);
  * settled fraction above the spec bar (<28%).

    python src/probe/corpus_acceptance.py --data ~/data/probe_corpus \\
        --result_dir results/probe/acceptance
"""

import argparse
import collections
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # single source of truth for the "settled" thresholds (coverage_audit).
    from probe.coverage_audit import SETTLED_V, SETTLED_W
except ImportError:  # cv2 chain unavailable in a bare env — keep in sync by hand
    SETTLED_V, SETTLED_W = 0.05, 0.10

# What extract actually needs per clip (dataset.load_clip_samples); depth/normals
# are part of the save contract but unused by the probe — counted, not required.
REQUIRED = ("meta.json", "state.json", "rgb.mp4", "seg.npz", "seg_id_to_class.json")
OPTIONAL = ("depth.npz", "depth_meta.json", "normals.npz")

# Spec §6 bars this script owns (density/CoM/range bars live in the other audits).
SETTLED_BAR = 0.28     # fraction of near-motionless bodies at the probed frame
MIN_GROUPS = 3         # below this a scenario cannot survive a group split
SEED_BAND = 1000       # gen_probe_corpus.sh: scenario i gets seeds [i*1000, ...)


def _walk_clip_dirs(root):
    """(clip_dirs_with_meta, orphan_dirs). Mirrors dataset.find_clip_dirs but
    also surfaces clip_* dirs WITHOUT meta.json — extract silently skips those,
    an acceptance audit must count them (crash residue from a killed run)."""
    with_meta, orphans = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base.startswith("_"):        # _logs, _rejected — not corpus content
            dirnames[:] = []
            continue
        if base.startswith("clip_"):
            (with_meta if "meta.json" in filenames else orphans).append(dirpath)
    return sorted(with_meta), sorted(orphans)


def _center_frame(frame_count, n_frames):
    """The frame whose state extract reads: center of the sampled index set
    (dataset.load_clip_samples), NOT frame_count//2 — keep bit-identical."""
    idx = np.unique(np.linspace(0, frame_count - 1, n_frames).round().astype(int))
    return int(idx[len(idx) // 2])


def _config_key(meta):
    """The generation knobs that must be homogeneous across batches (pixels /
    timing; a mixed-config corpus is a silent domain split inside the corpus)."""
    return json.dumps({
        "schema_version": meta.get("schema_version"),
        "frame_count": meta.get("frame_count"),
        "resolution": meta.get("resolution"),
        "save_resolution": meta.get("save_resolution"),
        "render_fps": meta.get("render_fps"),
        "physics_hz": meta.get("physics_hz"),
    }, sort_keys=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="~/data/probe_corpus")
    p.add_argument("--result_dir", default="results/probe/acceptance")
    p.add_argument("--frames", type=int, default=32,
                   help="must match the extract --frames default (center-frame parity)")
    args = p.parse_args()
    root = os.path.expanduser(args.data)
    result_dir = os.path.expanduser(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    clip_dirs, orphans = _walk_clip_dirs(root)
    if not clip_dirs:
        sys.exit(f"no clip dirs with meta.json under {root}")

    clips = []            # one record per clip
    excluded = []         # relpaths extract must not see
    missing_required = [] # (relpath, [files])
    for d in clip_dirs:
        rel = os.path.relpath(d, root)
        present = set(os.listdir(d))
        miss = [f for f in REQUIRED if f not in present]
        if miss:
            missing_required.append((rel, miss))
            excluded.append(rel)
            continue
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        ok = bool(meta.get("validation", {}).get("ok", False))
        if not ok:
            excluded.append(rel)
        mv = meta.get("multiview") or {}
        group = int(mv.get("rollout_seed", meta["seed"]))

        # Settled fraction at the exact frame the probe reads (spec §4/§6 bar).
        center = _center_frame(int(meta["frame_count"]), args.frames)
        with open(os.path.join(d, "state.json")) as f:
            state = json.load(f)
        rec = {r["body"]: r for r in state if r["frame"] == center}
        vmag = {b["name"]: float(np.linalg.norm(rec[b["name"]]["lin_vel_camera"]))
                for b in meta["bodies"] if b["name"] in rec}
        wmag = {b["name"]: float(np.linalg.norm(rec[b["name"]]["ang_vel_camera"]))
                for b in meta["bodies"] if b["name"] in rec}

        clips.append({
            "rel": rel,
            "scenario": meta.get("scenario", "?"),
            "group": group,
            "ok": ok,
            "config": _config_key(meta),
            "date": datetime.date.fromtimestamp(
                os.path.getmtime(os.path.join(d, "meta.json"))).isoformat(),
            "n_bodies": len(meta["bodies"]),
            "kinds": collections.Counter(b["kind"] for b in meta["bodies"]),
            "optional_missing": [f for f in OPTIONAL if f not in present],
            "settled_v": [v < SETTLED_V for v in vmag.values()],
            "settled_w": [w < SETTLED_W for w in wmag.values()],
        })

    kept = [c for c in clips if c["rel"] not in set(excluded)]

    # --- groups: uniqueness is the one HARD invariant -------------------------
    by_group = collections.defaultdict(list)
    for c in kept:
        by_group[c["group"]].append(c["rel"])
    dup_groups = {g: rels for g, rels in by_group.items() if len(rels) > 1}

    # --- per-scenario table ----------------------------------------------------
    scen = collections.defaultdict(list)
    for c in kept:
        scen[c["scenario"]].append(c)
    scen_tbl = {}
    for s, cs in sorted(scen.items()):
        groups = sorted({c["group"] for c in cs})
        band = (groups[0] // SEED_BAND) * SEED_BAND
        # attempted ≈ span from the band start to the highest kept seed; trailing
        # failures are invisible, so this UNDERestimates attempts (labelled est.).
        attempted = groups[-1] - band + 1
        sv = [x for c in cs for x in c["settled_v"]]
        sw = [x for c in cs for x in c["settled_w"]]
        kinds = collections.Counter()
        for c in cs:
            kinds.update(c["kinds"])
        scen_tbl[s] = {
            "clips": len(cs),
            "groups": len(groups),
            "seed_band": band,
            "keep_rate_est": len(groups) / attempted,
            "settled_v_frac": float(np.mean(sv)) if sv else float("nan"),
            "settled_w_frac": float(np.mean(sw)) if sw else float("nan"),
            "bodies_per_clip": float(np.mean([c["n_bodies"] for c in cs])),
            "kind_mix": dict(kinds),
            "dates": dict(collections.Counter(c["date"] for c in cs)),
        }

    configs = collections.Counter(c["config"] for c in kept)
    dates = collections.Counter(c["date"] for c in kept)
    settled_v_all = float(np.mean([x for c in kept for x in c["settled_v"]]))
    settled_w_all = float(np.mean([x for c in kept for x in c["settled_w"]]))
    n_groups = len(by_group)
    med_g = float(np.median([t["groups"] for t in scen_tbl.values()]))
    starved = [s for s, t in scen_tbl.items() if t["groups"] < MIN_GROUPS]
    lopsided = [s for s, t in scen_tbl.items()
                if t["groups"] < 0.5 * med_g and s not in starved]

    hard_fail = bool(dup_groups) or bool(missing_required)

    # --- artifacts -------------------------------------------------------------
    audit = {
        "data_root": root,
        "n_clip_dirs": len(clip_dirs), "n_orphan_dirs": len(orphans),
        "orphans": [os.path.relpath(o, root) for o in orphans],
        "n_kept": len(kept), "n_excluded": len(excluded), "excluded": excluded,
        "missing_required": missing_required,
        "n_groups": n_groups, "duplicate_groups": dup_groups,
        "configs_distinct": len(configs),
        "configs": {k: v for k, v in configs.items()},
        "dates": dict(dates),
        "settled_v_frac": settled_v_all, "settled_w_frac": settled_w_all,
        "scenarios": scen_tbl,
        "hard_fail": hard_fail,
    }
    with open(os.path.join(result_dir, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    with open(os.path.join(result_dir, "exclude_clips.txt"), "w") as f:
        f.write("\n".join(excluded) + ("\n" if excluded else ""))

    # --- report.md -------------------------------------------------------------
    lines = [
        "# Corpus acceptance audit (E1 pre-GPU gate)", "",
        f"`{root}` · {len(clip_dirs)} clip dirs (+{len(orphans)} orphan, no meta.json) "
        f"· **{len(kept)} accepted / {len(excluded)} excluded** · "
        f"**{n_groups} independent rollout groups** · batches: "
        + ", ".join(f"{d}: {n}" for d, n in sorted(dates.items())) + ".", "",
        "## Verdict", "",
        f"- {'⛔ **HARD FAIL**' if hard_fail else '✅ no hard failures'} — "
        + ("duplicate groups / missing required files, see below; fix before extract."
           if hard_fail else "group IDs globally unique, contract files complete."),
        f"- validation.ok=False / incomplete: **{len(excluded)}** clip(s) → "
        f"`exclude_clips.txt`. Move them out of the corpus root before extract "
        f"(find_clip_dirs walks everything): `mkdir -p {root}_rejected && "
        f"while read c; do mv {root}/$c {root}_rejected/; done < "
        f"{os.path.join(result_dir, 'exclude_clips.txt')}`" if excluded else
        "- no excluded clips.",
        f"- configs across batches: **{len(configs)} distinct** "
        + ("✅ homogeneous." if len(configs) == 1 else
           "⚠ MIXED — see audit.json `configs`; a resolution/fps split inside the "
           "corpus is a silent domain confound."),
        f"- settled at probed frame: |v| {settled_v_all:.0%}, |w| {settled_w_all:.0%} "
        + (f"(bar < {SETTLED_BAR:.0%} ✅)" if settled_v_all < SETTLED_BAR else
           f"(bar < {SETTLED_BAR:.0%} ⚠ — motion-gate the sampled frame in E2, "
           f"spec §4)"),
        f"- group-starved (<{MIN_GROUPS}): "
        + (f"⚠ {', '.join(starved)}" if starved else "none")
        + (f" · imbalanced (<½ median {med_g:.0f}): {', '.join(lopsided)}"
           if lopsided else ""),
        "",
        "## Per-scenario", "",
        "| scenario | clips | groups | keep-rate (est) | settled |v| | settled |w| "
        "| bodies/clip | batches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s, t in sorted(scen_tbl.items(), key=lambda kv: -kv[1]["groups"]):
        flag = " ⚠" if s in starved or s in lopsided else ""
        lines.append(
            f"| {s}{flag} | {t['clips']} | {t['groups']} | {t['keep_rate_est']:.0%} "
            f"| {t['settled_v_frac']:.0%} | {t['settled_w_frac']:.0%} "
            f"| {t['bodies_per_clip']:.1f} | "
            + ", ".join(f"{d.replace('2026-', '')}: {n}"
                        for d, n in sorted(t["dates"].items())) + " |")
    lines += [
        "",
        "Keep-rate = kept seeds / seed-span from the band start — an *upper bound* "
        "estimate (trailing failed seeds are invisible). A low rate means the "
        "scenario's validation gate rejects most rollouts: topping it up needs "
        "proportionally more `--n_clips`, not a rerun at the same R.",
    ]
    if dup_groups:
        lines += ["", "## ⛔ Duplicate groups", ""] + [
            f"- `{g}`: {', '.join(rels)}" for g, rels in sorted(dup_groups.items())]
    if missing_required:
        lines += ["", "## ⛔ Missing required files", ""] + [
            f"- `{rel}`: {', '.join(miss)}" for rel, miss in missing_required]
    if orphans:
        lines += ["", f"{len(orphans)} orphan clip dir(s) without meta.json "
                      "(crash residue; extract skips them — delete at leisure): "
                  + ", ".join(f"`{os.path.relpath(o, root)}`" for o in orphans[:10])
                  + (" …" if len(orphans) > 10 else "")]
    lines += ["", "Next (spec §6 order): density_size_audit → extract → "
                  "coverage_audit. Raw: `audit.json` · exclusions: "
                  "`exclude_clips.txt`."]
    with open(os.path.join(result_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- console digest --------------------------------------------------------
    print(f"{len(kept)} accepted / {len(excluded)} excluded / {len(orphans)} orphan "
          f"| {n_groups} groups | configs {len(configs)} "
          f"| settled v {settled_v_all:.0%} w {settled_w_all:.0%}")
    for s, t in sorted(scen_tbl.items(), key=lambda kv: -kv[1]["groups"]):
        print(f"  {s:18s} {t['clips']:4d} clips {t['groups']:4d} groups "
              f"keep~{t['keep_rate_est']:4.0%} settled_v {t['settled_v_frac']:4.0%}")
    print("HARD FAIL — fix before extract" if hard_fail else "acceptance: OK to extract")
    print(f"report.md + audit.json + exclude_clips.txt -> {result_dir}")


if __name__ == "__main__":
    main()
