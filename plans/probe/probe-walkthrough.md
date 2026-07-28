# Probe Walkthrough — supervision signals & step-by-step (for researchers)

**Audience:** a researcher picking up the probe for the first time. **Goal:** know
*exactly what we supervise and why*, then run the gate end-to-end. This is the
onboarding layer; the design rationale is in [gate-b-probe.md](gate-b-probe.md), the
representation theory in [../theory/world-lines.md](../theory/world-lines.md), the gate
definitions in [../IMPLEMENTATION.md](../IMPLEMENTATION.md#hard-gates-do-not-skip).

---

## 0. The one question the probe answers

> Is physical state **recoverable from a frozen generalist encoder** (V-JEPA 2), and
> **at which depth** — *before* we spend anything building a model?

Nothing here finetunes the encoder. We freeze it, pool its features per object, and fit a
**small head** per (layer, target). The per-layer curve *is* the deliverable: where the
signal peaks tells the real architecture where to tap (GATE B). If no signal exists at any
depth, the escape hatch is justified — but we have to *measure* that, not assume it.

## 1. Mental model (one diagram in words)

```
RGB clip (K frames)
   └─ frozen V-JEPA 2 ─► per-block token grid  (16 temporal × 24×24 spatial × 1408)
        └─ masked-pool per object over GT seg ─► per-body feature  (per layer)
             └─ small head (linear=2a / MLP=2b) ─► SUPERVISION SIGNAL (below)
```

We **cache the pooled per-body features once** (the GPU pass), then iterate fits on CPU.
Never store the raw token grid (52 MB/clip) — only the pooled per-body vectors.

---

## 2. Supervision signals — the transparent spec (THE core of this doc)

Two tiers. **Tier 1 is implemented now** (`src/probe`, `TARGETS=("v","w","com_offset")`)
and is what the *gate* needs. **Tier 2 is the world-lines upgrade** — richer, trajectory-
level, dimensionless — layered on once Tier 1 reads cleanly.

All Tier-1 labels are **per dynamic body**, in **camera frame** (data-gen invariant; world
recoverable via `R_camera_world`), **masked-pooled over GT segmentation** so the head sees
one feature vector per object.

### 2.1 Tier 1 — what we regress today

| Signal | Symbol / shape | Frame & units | How it's derived | Metric | Pass bar (best layer) | Role |
|---|---|---|---|---|---|---|
| **Linear velocity** | `v` ∈ ℝ³ | camera frame, m/s | from `state.json` `lin_vel_camera` | R² | **v R² > 0.5** | **CONTROL** |
| **Angular velocity** | `ω` ∈ ℝ³ | camera frame, rad/s | from `state.json` `ang_vel_camera` | R² (per-axis + agg) | **ω R² > 0.5** | intermediate |
| **CoM offset** | `CoM − centroid` ∈ ℝ³ | camera frame, m | body-local offset (`com_local_authored` for primitives, `com_local_analytic` for composites) rotated to camera: `R(quat_camera_wxyz) · com_local`. Prim origin == centroid, so this **is** CoM − centroid; the zero vector is exactly the centroid baseline | L2 + lift | **L2 < ½ median bbox diag AND lift > 0** | hard |

Why exactly these three, and not "absolute CoM" or "mass":
- **`v` is the control, and you read it first.** Linear velocity has a strong apparent-
  motion signature; any video encoder should nail it. **If `v` fails, the harness is
  broken — not the embedding.** It is the only thing separating "dataloader/probe bug"
  from "no physics signal," which the gate threshold alone cannot.
- **`ω`** is intermediate: motion-observable but subtler than translation.
- **CoM offset, not absolute CoM (R7).** Absolute CoM scores high R² for free just by
  localizing the object in frame — telling you nothing about mass distribution. We predict
  the **offset from the geometric centroid** and report **lift over the centroid
  baseline**. The centroid is computable from the silhouette with no physics; the *lift* is
  the encoded physics. A probe that only matches the centroid baseline has learned nothing.

> **Update 2026-07-05 (755-group corpus, `results/probe/fit_matrix`): Tier-1
> kinematic targets are now WINDOW-ALIGNED.** The features pool the whole
> clip, so regress the **clip-mean** `v`/`ω` (mean over the sampled frames),
> not the center-frame instantaneous state — alignment alone added +0.29 v R²
> and the control now **passes** (0.60 @ layer 32; instantaneous peaked at
> 0.31). CoM is unchanged (per-body constant) but prefer the **normalized**
> offset (÷ half-extent) to close the |CoM|~size route. Builder:
> `src/probe/build_targets_v2.py`; A/B evidence: `src/probe/fit_matrix.py`.

### 2.2 The reading discipline (do not skip)

1. **Check `v` first.** Control fails → fix the harness; do **not** interpret ω/CoM rows.
2. **Report CoM split by scenario** (free-fall vs contact-rich). CoM offset has near-zero
   *static* signature — it is observable **only through dynamics/contact (R6)**. A flat
   CoM failure on free-fall + success on `topple`/`collide` is the **healthy, expected**
   result, not a gate failure. If the corpus were all free-fall, CoM is unobservable *by
   construction* and a failure says nothing about the encoder.
3. **2a vs 2b** (linear vs MLP, same features & splits): the gap is "nonlinear/multi-layer
   recoverability." Per-layer curves locate the peak depth.

### 2.3 The cleanliness rules behind the formulations (R6/R7/R8)

These are *why the signals look the way they do* — enforce them or the probe lies:
- **R7 — never absolute CoM** (above): offset + lift only.
- **R8 — no appearance cue.** The invariant signals (CoM, later mass) must be inferable
  from geometry+dynamics only, never colour/texture. Upstream: one uniform material per
  body. In the head: feed invariants from *dynamics features*, not raw appearance, or
  "mass" silently becomes an appearance classifier.
- **R6 — observability must be in the scenarios.** Logging the label is necessary but not
  sufficient; the *motion* has to make it visible. Hence scenario-split reporting.

### 2.4 Tier 2 — the world-lines upgrade (next, not first)

Once Tier 1 reads cleanly, the richer supervision is the **world-line** representation
([../theory/world-lines.md](../theory/world-lines.md), encoder in
[../theory/world-lines-encoding.md](../theory/world-lines-encoding.md), code
`src/worldlines/`). Differences, stated transparently:

| Aspect | Tier 1 (now) | Tier 2 (world-lines) |
|---|---|---|
| Granularity | instantaneous per-frame state | whole-clip **trajectory** (basis coefficients) |
| Frame | camera | **allocentric world** + **ego-line** (camera factored out) |
| Units | metric (m/s, rad/s, m) | **dimensionless, gravity-natural** (`g≡1`, τ∈[0,1]) |
| Signals | v, ω, CoM-offset | trajectory `pos_coeffs`/`rot_coeffs`, + intrinsics: **mass ratio**, restitution, CoM-offset/size, existence `e(t)` |
| Scale-dependence | `v` needs scale; ω, CoM-ratio scale-free | **all scale-free** (mass is **ratio** only — vision never gives absolute mass) |

Tier 2 is *strictly richer supervision over the same frozen features* — same harness, more
targets. Do not start here: it adds the gauge, the ego-line, and the basis, three moving
parts that obscure a clean gate read. Land Tier 1 first.

---

## 3. Data & the one rule that fakes a pass if you break it

- **Corpus:** `~/data/sim_physics` — 110 clips = **18 physics rollouts** fanned to 3–7
  camera views (`multiview` block). Reads `rgb.mp4`, `state.json`, `seg.npz`, `meta.json`
  (depth/normals unused).
- **SPLIT BY `rollout_seed`, NEVER BY CLIP.** Sibling camera views of one rollout in both
  train and test = trivial leakage → inflated R². Group-split on the rollout seed
  (leave-rollout-out). This is the single most common way to fake a passing gate.
- **Generate more for a real result.** 18 rollouts only justifies a linear probe; ~1500
  diverse rollouts (all 9 scenarios, full DR) let 2b (nonlinear) be meaningful. Save lean
  (RGB + state + seg; skip depth/normals) and cache pooled features, not raw embeddings.

## 4. Step-by-step run

```bash
# 1. GPU pass once → cached per-(clip, body) pooled features per layer
python src/probe/run_probe.py extract \
    --data ~/data/sim_physics --result_path results/probe/feats.npz

# 2. group-split (by rollout) + ridge per (layer, target); CPU only, re-runnable
python src/probe/run_probe.py fit \
    --features results/probe/feats.npz --report results/probe/2a_vjepa2.json
```

Module map (`src/probe/`): `encoders.py` (frozen V-JEPA wrapper, per-block taps) ·
`dataset.py` (clip → per-body samples + targets + masks) · `metrics.py` (R², lift) ·
`run_probe.py` (the only entry point: `extract` then `fit`).

## 5. How to read the report

1. `v` R² ≥ 0.5? No → **harness bug**, stop and fix.
2. Per-layer **curve** for ω and CoM-lift → where does each peak? That depth is the GATE-B
   answer (where Rung-1 taps).
3. CoM-lift on **contact-rich** scenario slices (not the pooled number).
4. 2b − 2a gap → nonlinear recoverability.
5. Compare V-JEPA 2 vs DINO/SigLIP under the identical harness (is motion-training needed).

Gate outcome: signal present at some depth → **frozen-first main line** (and which blocks).
Signal absent at *every* depth, with the `v` control passing → escape hatch justified.

## 6. Where to contribute (plug-in points)

- **Add a Tier-1 target:** extend `TARGETS` + its derivation in `dataset.py` + a metric in
  `metrics.py`. Keep it R7/R8-clean (ratios/offsets, dynamics-derived).
- **Add an encoder:** new wrapper in `encoders.py` exposing per-block features; the rest is
  encoder-agnostic.
- **Tier-2 targets:** wire `src/worldlines/encode_clip` outputs (trajectory coeffs +
  intrinsics) as additional regression targets on the same cached features.
- **Scenario coverage / scale-up:** the highest-leverage contribution right now — more
  diverse rollouts (§3).

## 7. Links
[gate-b-probe.md](gate-b-probe.md) (design) · [intphys2-surprise.md](intphys2-surprise.md)
(the deprioritized external eval) · [../theory/world-lines.md](../theory/world-lines.md) ·
[../IMPLEMENTATION.md](../IMPLEMENTATION.md) (gates & status).
