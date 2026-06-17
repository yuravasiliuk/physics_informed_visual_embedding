# GATE B — Frozen-Encoder Physics Probe (implementation sub-plan)

**Status:** v0.0 implemented — arm 2a only, V-JEPA 2 only, in `src/probe/`
(dataset, encoder wrapper with per-block taps, closed-form ridge probe,
`run_probe.py extract|fit`). Untested against a real corpus; 2b / DINO / SigLIP
not started. Operational sub-plan — the *why* lives in
[BRAINSTORM §2](../BRAINSTORM.md#2-architecture) (capacity ladder) and the gate
definitions in [IMPLEMENTATION.md](../IMPLEMENTATION.md#hard-gates-do-not-skip).

**Brutal-honesty mandate** (inherited from
[physics-latent-ladder.md](../theory/physics-latent-ladder.md)): sections marked
**[Honest take]** are meant to be argued with. The whole point of this probe is to
get an *honest* yes/no before any encoder/training spend — so do not soften the
kill criteria.

---

## 1. The one question this answers

> **Does a frozen V-JEPA 2 embedding contain CoM/ω signal that a head can recover
> from rendered video — and at what depth?**

This is **GATE B / Rung 0** ([BRAINSTORM §2.2](../BRAINSTORM.md#22-the-capacity-ladder-each-rung-adds-capacity-subtracts-stability)).
Its outcome decides the architecture: signal present → the frozen-first main line
is justified and its peak layer tells the eventual Rung-1 head where to tap;
signal absent at *every* depth even nonlinearly → the frozen embedding is the
bottleneck and only the raw-pixel escape hatch (Rung 2) could help.

The user's framing — "use V-JEPA embeddings as-is (baseline) vs. train with
CoM/ω supervision heads, and if physics knowledge doesn't help, no sense to
continue" — *is* this gate. Under a **frozen** encoder the contrast cannot be
"no heads vs. heads" (two heads on a frozen trunk are independent; one cannot help
the other). The real variable is **probe capacity over the same frozen features**:

- **Arm 2a — linear probe (Rung 0).** One linear layer per target, **probed at
  each intermediate block** independently. The floor: what is *linearly exposed*.
- **Arm 2b — nonlinear multi-layer head (Rung 1).** Fuse several intermediate
  blocks → attentive-pool / shallow-MLP head. The ceiling of *frozen-feature
  recoverability* without ever touching raw pixels.

## 2. Scope / non-goals

- **Frozen encoder, heads only.** No fine-tune, no LoRA, no adapter. Decided with
  the user; preserves the sim→real robustness the freeze buys ([BRAINSTORM §2.1](../BRAINSTORM.md#21-why-frozen-features-first-is-the-stable-choice-and-its-honest-cost)).
- **Eval = held-out split of our own sim corpus.** Direct CoM/ω regression. Chosen
  for simplicity: if physics is unrecoverable *in-distribution*, there is nothing
  to transfer and we stop. **External benchmarks (IntPhys2) are explicitly a later
  iteration**, not in this cut (and IntPhys2 is VoE/plausibility — it has no
  CoM/ω labels, so it can never be the *primary* metric for this contrast).
- **State read-out, not future-prediction** ([BRAINSTORM §7](../BRAINSTORM.md#7-open-decisions)):
  regress the *current* ω/CoM/v. Forecasting re-does V-JEPA's native task and
  conflates encoding with dynamics modelling.
- **No dense heads** (depth/normals/seg) here — those are the P2 main-line head,
  not a gate. This probe is physics-only.

## 3. Targets (R7-/R8-clean)

All labels in **camera frame** (data-gen invariant; world recoverable via
`R_camera_world`). Per dynamic body, masked-pooled over its GT segmentation.

| Target | Formulation | Metric | Pass bar (best layer) |
|---|---|---|---|
| **CoM offset** | `CoM − geometric_centroid`, **lift over centroid baseline** (never absolute CoM — **R7**) | L2 error; lift = (centroid-baseline L2 − probe L2) | CoM L2 < ½ median bbox diagonal **and** lift > 0 by a clear margin |
| **Angular velocity ω** | camera-frame ω vector | R² (per-axis + aggregate) | ω R² > 0.5 |
| **Linear velocity v** | camera-frame v vector | R² | v R² > 0.5 — **control target** |

`v` is the **control**: linear velocity has a strong apparent-motion signature, so
a video encoder should recover it easily. **If `v` fails, the harness is broken —
not the embedding.** It separates "probe/dataloader bug" from "no physics signal,"
which the gate alone cannot. CoM offset is the *hard* target (near-zero static
signature; observable only via dynamics/contact — **R6**); ω is intermediate.

Thresholds reuse the **GATE A** numbers ([IMPLEMENTATION.md](../IMPLEMENTATION.md#hard-gates-do-not-skip))
deliberately, so GATE A (perfect-RGB feasibility) and GATE B (frozen-encoder
recoverability) are read on the same ruler.

## 4. Design

### 4.1 Encoder wrapper
- **V-JEPA 2** frozen, primary. Expose **per-intermediate-block** token features
  (not just the final embedding) — the whole probe hinges on having every depth.
- **Implementation (v0.1): repo-native ViT-g/384**, not the HF checkpoint — a
  local `facebookresearch/vjepa2` checkout + `weights/vitg-384.pt`, loaded via
  `vit_giant_xformers_rope` with the model's own `out_layers` hook (every block
  tapped through the final LayerNorm), and the repo's eval transform (short
  side → 256/224·384, center-crop 384²). Validated interactively in
  `vjepa2/notebooks/Untitled.ipynb`; the per-body mask path mirrors the same
  resize+crop arithmetic so mask→token pooling stays aligned. 40 layers, D=1408.
- Compare **DINOv2 / SigLIP** under the identical harness ([BRAINSTORM §2.2](../BRAINSTORM.md#22-the-capacity-ladder-each-rung-adds-capacity-subtracts-stability)).
  V-JEPA 2 is the one plausibly carrying ω-relevant dynamics (video/motion-trained);
  the image encoders are the "is motion training actually necessary" control.
- **Temporal handling** ([BRAINSTORM open arch Q](../BRAINSTORM.md#open-architecture-questions)):
  V-JEPA 2 is natively temporal (feed the K-frame clip). For DINO/SigLIP (per-frame)
  use a small temporal pool over per-frame features so ω/v are even expressible.
  Clip length **K = 4** to start.

### 4.2 Probe heads
- **2a:** `Linear(d_layer → target_dim)` per (layer, target). No nonlinearity. Cheap;
  run all layers.
- **2b:** select the top-k layers from 2a, concat/attentive-pool → LayerNorm → MLP
  (1–2 hidden) → target. Per-object via masked pool over GT seg before the head.
- Both consume **identical features and identical splits** — the only difference is
  head capacity, so any 2b−2a gap is "nonlinear/multi-layer recoverability."

### 4.3 Reporting
- **Per-layer curves** for every target (this is the GATE-B deliverable: *where*
  the signal peaks → where Rung-1 taps).
- Always against the **trivial baseline** (centroid for CoM; mean-predictor for ω/v).
- **R9 caveat surfaced, not solved:** a wide error can mean unobservable *or*
  under-capacity. We do not have a reference upper-bound model in this cut; the `v`
  control + the 2a→2b lift are the partial separators. Note it; don't claim to have
  removed it.

## 5. Data contract & split protocol

Reads the data-gen output contract (P0.4): `rgb.mp4`, `state.json` (per-frame
camera-frame ω/v/CoM + `R_camera_world`), `seg.npz` (per-object masks), `meta.json`
(`bodies[].kind/com_local_analytic/mass`, multiview provenance). Depth/normals
unused here.

**[Honest take] — leakage is the failure mode that fakes a pass:**
- **Split by physics rollout, not by clip.** Multiview fan-out emits N sibling
  clips from *one* rollout (linked by the `multiview` provenance block). Sibling
  views in both train and test = trivial leakage → inflated R². **Group-split on
  rollout seed.**
- **Cover both body kinds and both observability scenarios.** Composites (W5) carry
  the invisible CoM offset; `topple`/`collide` (W3) are what make CoM/mass
  *dynamically visible* (**R6**). If the corpus is mostly free-fall, CoM is
  unobservable *by construction* and a CoM failure says nothing about the encoder.
  Stratify the split so this is checkable, and report CoM metrics **split by
  scenario** (free-fall vs contact-rich) — a flat free-fall failure + contact-rich
  success is the expected, healthy result, not a gate failure.

## 6. Prerequisite chain (model side is greenfield)

Nothing here runs until these land — flagged honestly so "data-gen done → probe"
doesn't skip them:

1. **`save_resolution` lock (§14, non-retrofittable).** Pixels are baked at
   generation; depends on the V-JEPA 2 checkpoint. **Resolved: the ViT-g 384²
   checkpoint is the encoder (§4.1)**, so `save_resolution` must stay ≥ 384²
   (current 1024² is fine — the wrapper downsamples at encode time).
2. **Probe-sized corpus** (a few hundred clips, *not* the full P1 pilot) across
   `topple`/`collide`/`cluttered`, with W5 composites on. Needs the P0.4 end-to-end
   smoke + critique-#3 (don't-persist-failed) first, or the split is polluted.
3. **Dataloader** — per-clip dirs → batches; **dequantize depth** (uint16 +
   `depth_min/max`), camera-frame state at load (raw log stays world-frame),
   group-split metadata.
4. **Frozen encoder wrapper** (§4.1) with intermediate-block taps.
5. **(Recommended) run GATE A once on public data first** (Kubric/CLEVRER). Our
   corpus now has exact labels so GATE A *can* run on it — but a public set first
   isolates "method bug" from "my-data bug." ~1 GPU-day of insurance.

## 7. Proposed module layout

```
src/probe/
  encoders.py     # frozen V-JEPA2 / DINOv2 / SigLIP wrappers, per-block tap API
  dataset.py      # clip dir -> (clip_frames, per-object {com_offset, ω, v}, seg); group-split
  heads.py        # LinearProbe (2a), MultiLayerHead (2b)
  metrics.py      # R², CoM-offset lift-over-centroid, per-layer/per-scenario aggregation
  run_probe.py    # CLI: --encoder --arm {2a,2b} --layers ... -> per-layer report (json + plot)
```
`run_probe.py` is the only entry point; it must not import Isaac (probe runs on a
normal GPU env, not `${ISAAC_SIM}/python.sh`).

## 8. Run procedure
```bash
# Arm 2a — linear probe at every block, all encoders
python src/probe/run_probe.py --encoder vjepa2 --arm 2a --data ~/data/probe --report out/probe/2a_vjepa2.json
python src/probe/run_probe.py --encoder dinov2 --arm 2a --data ~/data/probe --report out/probe/2a_dino.json
python src/probe/run_probe.py --encoder siglip --arm 2a --data ~/data/probe --report out/probe/2a_siglip.json
# Arm 2b — nonlinear multi-layer head on the top-k layers from 2a
python src/probe/run_probe.py --encoder vjepa2 --arm 2b --layers <topk-from-2a> --data ~/data/probe --report out/probe/2b_vjepa2.json
```

## 9. Success criteria (climb to Rung 1)
At least one encoder's **best layer** clears, on the **held-out** split:
- `v` R² > 0.5 (harness sanity — required, else results are uninterpretable), **and**
- ω R² > 0.5 **or** CoM-offset L2 < ½ median bbox diagonal with positive lift,
- **on the contact-rich (`topple`/`collide`) slice** (where R6 says the signal can
  exist). → frozen-first main line confirmed; that layer sets the Rung-1 tap.

## 10. Kill criteria (stop / escape hatch)
- **`v` control fails** → the probe/dataloader/encoder wrapper is broken. Fix the
  harness before concluding anything about physics. (Not an encoder verdict.)
- `v` passes but **ω and CoM fail at every layer for every encoder, including 2b,
  on the contact-rich slice** → the signal is not recoverable from frozen features.
  This is the honest "no sense to continue (frozen)" verdict → the only remaining
  lever is the raw-pixel escape hatch (Rung 2, [BRAINSTORM §2.3](../BRAINSTORM.md#23-alternative--escape-hatch-the-raw-pixel-two-stream-the-pdf-design)),
  entered only with R1's load-bearing ablation.
- **2b ≈ 2a everywhere** → no nonlinear/multi-layer gain; either the signal is
  linearly exposed (good, cheap) or absent (see above) — disambiguated by the
  absolute level, not the gap.

## 11. Risks / [Honest take]
- **Ambiguous failure (the big one).** With own-sim held-out only, a CoM failure
  conflates three causes: (a) encoder lacks the signal *(the gate's intended
  reading)*, (b) the corpus lacks observability — R6, (c) too little data → 2b
  overfits. The `v` control kills (a-vs-harness), scenario-split reporting kills
  (a-vs-b), and a train-vs-test gap on 2b flags (c). **If we don't instrument all
  three, a null result is uninterpretable** — which would waste the gate.
- **Small corpus + nonlinear 2b** → overfitting masquerading as recoverability.
  Watch the train/test gap; keep 2b shallow; regularize.
- **IntPhys2-as-future is not free.** It tests plausibility, not CoM/ω; wiring it
  later means a *different* readout (head outputs → surprise score), not a metric
  swap. Noted so the "future iteration" isn't under-scoped.
- **Checkpoint/resolution coupling.** Probing 256² features but generating 384²
  clips (or vice-versa) wastes the non-retrofittable resolution lock. Decide §6.1
  before the corpus.

## 12. Open decisions (settle before code)
1. **V-JEPA 2 checkpoint** (ViT-L/H/g; 256² vs 384²) → fixes `save_resolution`.
2. Which/how-many intermediate blocks to tap for 2b (data-driven from 2a, but cap
   for compute).
3. Per-frame encoders' temporal pool (mean vs small attention) — keep minimal.
4. Corpus size / scenario mix for a *statistically* meaningful per-layer R².

## 13. Pairs — controlled-intervention divergence (companion, not a gate)

The probe's necessary-condition companion: generate twin rollouts that are
identical except for one latent (`collect_pairs.py`, a standalone additive
driver — same scene, same rig, same frame-0 state, zero extra RNG draws; the
CoM is re-authored to `α·sampled_offset` between simulate passes, one twin per
`--com_scales` value, α=0 = centroid), then measure layer-wise feature
divergence between twins (`run_probe.py pairs`). Each twin logs a ground-truth
kinematic-divergence diagnostic (max CoM-path separation, fork frame) — tune
α from that log *before* any probe extraction: visible fork, but separation
≲ body scale (past that, divergence saturates and is uninformative). Dose for
the per-layer Spearman = |‖offset_base‖ − ‖offset_twin‖| = (1−α)·‖offset‖.

Reading (per layer, twin distance normalized by the same layer's
unrelated-clip distance — raw L2 is not depth-comparable):
- **ratio ≈ 0 at every depth** → the encoder collapses the intervention; no
  head of any capacity can recover it. Strengthens a probe-kill verdict
  against the R9 "maybe under-capacity" objection.
- **divergence present, linear probe fails** → signal present but not
  linearly exposed → direct justification for arm 2b.
- **Spearman(‖true offset‖, twin divergence) per layer** is the dose-response
  check; its peak depth should agree with the 2a peak layer if both are real.

Caveats: V-JEPA keeps object *trajectories* (its own framing: it ignores
unpredictable detail, not motion), so late-window twin divergence can mean
"the videos differ", not "CoM is encoded" — prefer short/early windows and
the dose-response curve over the raw ratio. Pairs corpora are primitives-only
(`composite_frac=0`); the gate itself still needs W5 composites (§5).

**Pair-tuned scenarios** (`pair_scenarios.py`, registered additively into
`scenarios.BUILDERS` when `collect_pairs.py` imports it; `collect_clip` is
untouched). Both honor `--n_objects` exactly via `_sample_n_bodies` — the
built-in `collide` hardcodes 2 bodies and `topple` uses its own 1–2 range:
- **`weighted_roll`** — spheres pushed horizontally, ω₀=0, lanes spaced so
  bodies never interact. The offset CoM modulates rolling speed at the rolling
  frequency ("loaded die" wobble) and curves the path: a *periodic, dose-graded*
  signature in every frame window, instead of drop's one-shot chaotic fork.
  For this scenario `max_divergence_m` undersells the effect (separation
  oscillates); read `final_divergence_m` / fork frame alongside it.
- **`collide_static`** — bodies launched at a static wall, ω₀=0. Pre-impact
  flight is torque-free → twins are pixel-identical until contact; the wall's
  off-CoM normal impulse converts linear momentum into spin: one clean
  single-contact fork at a predictable frame, no second dynamic body, no
  inter-body chaos.

## 14. Links
- Gate definitions & status: [IMPLEMENTATION.md P0.5](../IMPLEMENTATION.md#p05--gating-probes---gate-a--gate-b)
- Architecture / capacity ladder: [BRAINSTORM §2](../BRAINSTORM.md#2-architecture)
- CoM formulation (offset, lift, R7/R9): [BRAINSTORM §4](../BRAINSTORM.md#4-center-of-mass-the-well-posed-formulation)
- Observability (R6) loss weighting (later, F-config): [observability-weighted-loss.md](../theory/observability-weighted-loss.md)
- Sibling brainstorm (latent-dynamics ladder): [physics-latent-ladder.md](../theory/physics-latent-ladder.md)
- Data contract this consumes: [IMPLEMENTATION.md P0.4](../IMPLEMENTATION.md#p0--single-clip-data-gen-pipeline-isaac-sim)
