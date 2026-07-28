# b0 — preliminary encoder CoM read on the R6-certified eccroll slice

**What this is.** The fast path from [EXPERIMENTS.md](../EXPERIMENTS.md) ("Next: (b0)"):
probe the frozen encoder for CoM on the 46 eccentric_roll pilot rollouts —
the first corpus slice where the ★oracle *proved* CoM is decodable from the
dynamics (pooled pilots n=96 bodies: ridge lift +0.0199±0.0058,
MLP +0.0161±0.0087). Zero new generation, zero Isaac time: the pilots are
full clips already on disk. ~10 GPU-min total.

**Why it matters.** Every previous GATE-B CoM null was uninformative (R6
violated in the corpus — see IMPLEMENTATION.md §Where-we-are). This is the
first *interpretable* encoder CoM read: on this slice a null is finally
evidence about the encoder, not the data.

## Data (verified 2026-07-06)

`~/data/pilot_w6_ecc_all/` — 46 clip dirs (hardlinked pool of pilot v1 + v2),
all `scenario=eccentric_roll`, full payloads (rgb.mp4/seg/state/meta),
`validation.ok=true`, 46 unique rollout seeds (21000s / 23000s → v1 and v2
land in disjoint group-folds automatically), frame_count mixed 64/128
(extract samples 32 frames per clip regardless — handled per clip).

## Commands (run in `amass2` env, repo root)

```bash
# 1. Extract frozen V-JEPA 2 features (the only GPU step, ~5 min)
python src/probe/run_probe.py extract --data ~/data/pilot_w6_ecc_all \
  --result_path results/probe/ecc46/feats.npz --frames 32

# 2. Window-aligned + normalized targets, row-aligned to the cache
python src/probe/build_targets_v2.py --features results/probe/ecc46/feats.npz \
  --data ~/data/pilot_w6_ecc_all --result_path results/probe/ecc46/targets_v2.npz

# 3. Ridge, all variants x layers, 5-fold group split (the primary read)
python src/probe/fit_matrix.py --features results/probe/ecc46/feats.npz \
  --targets results/probe/ecc46/targets_v2.npz \
  --result_dir results/probe/ecc46/fit_matrix

# 4. OPTIONAL: MLP arm at the usual slots (n is tiny — see caveats)
python src/probe/arm_2b.py --features results/probe/ecc46/feats.npz \
  --targets results/probe/ecc46/targets_v2.npz \
  --ref results/probe/ecc46/fit_matrix/report.json \
  --result_dir results/probe/ecc46/arm_2b
```

## Pre-registered read (decide BEFORE looking, as with the oracle bar)

1. **Control first:** `v_clipmean` best-layer R² must be clearly positive
   (expect ≥0.4-ish at layers ~28–32; the 0.60 corpus number may not
   replicate exactly at 46 groups). If the control fails, the run is void —
   pipeline/slice problem, no CoM conclusion of any sign.
2. **Primary:** `com_norm` (and `com_offset`) lift-over-centroid at
   layers 28–32. Yardstick = the oracle ceiling on this same data (~+0.02
   from GT trajectories):
   - lift mean−std > 0 near that scale → encoder carries CoM signal →
     strong go for the mid-batch + folding eccroll into the corpus.
   - lift ≈ 0 with control passing → first **informative** encoder-side
     CoM null. Not yet decisive (n=46 groups, and the oracle effect itself
     is only ~11–13% of baseline) — decisive version needs the mid-batch.
3. **Directional only.** 46 groups / ~96 bodies, 5 folds ≈ 9 groups per
   test fold: expect large fold std. Ridge (step 3) is the primary read;
   the MLP arm at this n is unstable — treat it as a sanity check, never
   as the headline. No verdict from this run overrides the pre-registered
   mid-batch test in EXPERIMENTS.md — b0 only orders what to do next.

Paste `results/probe/ecc46/fit_matrix/report.json` headline block back for
interpretation.
