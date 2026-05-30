# Plan — Training Data for the Physics-Grounded Embedding

**Date:** 2026-05-15. Status: pre-collection scoping; supersedes ad-hoc choices.

Companion to [theoretical-paper-plan.md](theoretical-paper-plan.md). This document covers *only* the data side of training the encoder — what we pull from real sources, what we render in sim, and how labels are obtained per head.

**Why mixed sim + real, not sim-only or real-only.** The deliverable is a *domain-agnostic* encoder: policies trained on `φ(X_sim)` must deploy on real frames without re-training. This is only achievable if the encoder is exposed to both domains *during pretraining* — sim gives us cheap, dense, physically correct supervision; real gives the encoder the visual statistics that the deployment-time policy will actually see. The mixed corpus is therefore not a coverage convenience, it is the sim→real bridge itself. See [domain-marginalization-regularizer.md](domain-marginalization-regularizer.md) for the alignment mechanism.

## Label availability by head

The architecture from [../Physics informed embedding.pdf](../Physics%20informed%20embedding.pdf) has heads that split cleanly by data domain:

| Head | Real-data label source | Sim label source |
|---|---|---|
| Depth | RGB-D sensor (Kinect/ZED/iPhone LiDAR) or VFM pseudo-label (Video Depth Anything) | rendered ground truth |
| Normals | derived from depth or VFM (Omnidata, DSINE) | rendered ground truth |
| Segmentation | SAM2 / DEVA pseudo-labels | rendered instance/semantic masks |
| **CoM** | — (effectively unavailable) | simulator rigid-body state |
| **Angular velocity ω** | — (only from mocap + rigid-body fit, rare) | simulator state |
| **Contact forces** | — (FT sensors only at end-effector, sparse) | simulator contact reports |

**Implication:** real data trains only the top half. Sim-only heads are sim-exclusive. Real data's role in training is:
1. Supervise the sim+real heads at scale on natural-image statistics.
2. **Anchor the encoder so `φ(X_sim) ≈ φ(X_real)` in distribution.** This is the load-bearing sim→real bridge: downstream policies trained purely on sim embeddings must work on real frames. Real data is what gives the regularizer something to align against.
3. Prevent the sim-only heads from dragging the encoder into "sim-detector" shortcut mode (the asymmetric-supervision risk in [theoretical-paper-plan.md §2](theoretical-paper-plan.md)).

## Architecture — temporal modeling

The encoder operates on **K-frame clips**, not single frames. ω, v, and the time-derivative of CoM are not observable from a single image; the architecture must reflect that.

> **Main line is frozen-first.** The default is a **frozen encoder + a multi-layer
> nonlinear head over its features** (no raw-pixel trainable backbone) — trainable
> parameters never see raw pixels, so sim→real stability holds by construction. The
> raw-pixel conv branch + cross-attention below is an **escape hatch**, entered only
> if a layer-wise probe finds no recoverable physics signal in the frozen encoder at
> any depth. See [BRAINSTORM.md §2](BRAINSTORM.md#2-architecture). The **B/F split
> is orthogonal to this** — it selects *head sets* (B = dense only; F = + physics),
> not the backbone, and survives either choice.

- **K = 4** for the Phase-0.5 feasibility probe and Phase 1a/1b. **K = 8 or 16** for Phase 1c onward (matches Isaac Sim clip length).
- **Frozen encoder.** Use V-JEPA 2 with its native temporal tokens. DINOv2 is single-frame and is **not eligible** for the F config (no motion in a single frame); it remains a candidate only for a geometry-only B-config ablation, with frames encoded independently.
- **Main-line head (frozen-first).** Fuse several **intermediate** V-JEPA 2 blocks (à la DPT / DINO dense probes) into an attentive-pool / shallow head; the layer-wise linear probe (Rung 0) locates which depth carries the dynamics signal and gates whether to build anything heavier. A nonlinear multi-layer head over frozen features is a strict superset of the linear probe's reach while keeping its domain robustness.
- **Escape-hatch branch (only if Rung 0 finds no signal).** (2+1)D or 3D-conv U-Net on the raw K-frame stack, factorized over (space, time). The only path that can recover physics the frozen encoder discarded, because it sees full-information pixels — and the most transfer-fragile, so it carries R1's load-bearing ablation (measured on real).
- **Escape-hatch fusion.** Cross-attention with queries from learned-branch tokens and K/V from V-JEPA 2's spatiotemporal tokens. (Feeding *features* instead of pixels into this branch collapses it back to the main-line head — Rung 2's only justification is raw pixels.)
- **Head input.** Heads consume **clip-level features**. Dynamics heads (CoM, ω, v) read the last-frame slice for their prediction target; depth/normals/seg are predicted per-frame across the K-frame window.

## Label semantics

Heads split into two families by what they predict:

- **Dense (per-pixel) heads:** depth, normals, segmentation. Predicted per-frame across the K-frame clip. Background uses an *ignore mask*, not zero-fill.
- **Per-object heads:** CoM, ω, v, contact force. Predicted **per-body**, not per-pixel — one vector per body produced by masked-pooling the encoder features over the body's predicted segmentation mask, then an MLP. Per-pixel formulations of these quantities are degenerate (constant within each segment), satisfiable by `seg → MLP(seg_id)` without any geometric or physical learning.

All targets are in **camera frame** to avoid needing pose on real data and to make supervision invariant to camera-fixed viewpoint.

| Object category | Depth / Normals / Seg (per-pixel) | CoM (per-object) | ω, v (per-object) | Contact force (per-object) |
|---|---|---|---|---|
| Dynamic rigid body (object, robot link) | sim | sim if compact | sim | sim (sparse) |
| Static rigid body (table, fixed prop) | sim | sim if compact else ignore | 0 | sim if contacted else 0 |
| Ground plane (floor) | sim depth/normals; `ground` seg class | ignore | 0 | sim if contacted else 0 |
| Void / sky | far-clip or ignore; `background` seg class | n/a | n/a | n/a |

**CoM compactness filter** still applies. Among the heads, CoM is the one whose target is a 3D point that may live arbitrarily far from the body's pixels. For compact bodies (cube, robot link) the CoM is inside the silhouette and learnable from masked-pooled features. For structural bodies (wall, ceiling, infinite ground plane) the CoM is asset-author-defined and not physically meaningful. Skip the CoM target for bodies failing *either*:

- `kinematic=False` or `mass < threshold` (excludes architectural / fixed colliders), **and**
- `‖CoM − bbox_center‖ / bbox_diagonal < τ` (excludes non-compact bodies).

Static walls still contribute `ω=v=0` (useful "rigid + still" negatives) and contact when pushed.

**Contact-force loss weighting.** Contact labels are non-zero only when bodies are in contact, so per-batch sparsity is high. Apply a γ-multiplier on the contacting-body loss term (cf. RoboScape γ=5, [../summaries/roboscape.md](../summaries/roboscape.md)) or the head collapses to "always predict zero."

## Real-data candidates

Pick by downstream domain. Cross-domain mixing is allowed and probably desirable for the encoder's generalization.

### Robotic manipulation (matches RoboScape / Cosmos Policy stack)
- **AgiBot-World-Beta** — ~194k episodes; used by [RoboScape](../summaries/roboscape.md) and [Cosmos Policy](../summaries/cosmos-policy.md). Default match if downstream is manipulation.
- **DROID** — ~76k Franka trajectories, in-the-wild.
- **Open X-Embodiment** — aggregate of ~22 robots, ~1M trajectories.
- **BridgeData V2** — 60k+ demos, some RGB-D.

### Indoor scenes with sensor-grade RGB-D / 3D
Strong for depth + normals supervision; weak for dynamics (mostly static).
- **ScanNet / ScanNet++** — RGB-D + meshes.
- **Matterport3D, HM3D**.
- **ARKitScenes** — lidar + RGB.

### Egocentric / scaled video (weak labels, strong generalization)
- **Ego4D**, **EpicKitchens**.
- **SAYCam** — the corpus used by V-JEPA in the [intuitive-physics opinion paper](../summaries/opinion-intuitive-physics.md). Useful as an apples-to-apples comparison point.

### Driving (optional, only if domain matters later)
- nuScenes, Waymo Open.

### Recommended starting mix
**AgiBot-World-Beta + ScanNet++.** AgiBot covers the manipulation domain Cosmos and RoboScape pretrained on (clean baseline comparisons); ScanNet++ provides sensor-grade depth and normals across many static scenes, so the geometry heads get ground-truth supervision rather than pseudo-labels for at least part of the corpus.

## Synthetic data generation

Trade-off space: **physics fidelity × rendering realism × label richness × speed**.

### Simulator options

- **Isaac Sim / Isaac Lab** (PhysX 5 + RTX path-tracing). Best label coverage out of the box: depth, normals, instance/semantic seg, per-rigid-body CoM, linear/angular velocity, contact forces and contact points, mass, friction. GPU-parallel. **Default choice** — matches Cosmos and IntPhys 2 simulation stack so results are positioned against the standard.
- **MuJoCo MJX** (JAX, GPU-batched). Best contact-force accuracy and fastest for batched rollouts; rendering is OpenGL-basic. Good for millions of physics-rich clips when photorealism is secondary. Pairs naturally with a downstream Real2Sim translator if appearance matters.
- **Genesis** (2024). Newer, very fast, **differentiable physics** out of the box. Differentiability is interesting for [theoretical-paper-plan.md §4 (counterfactual / first-order supervision)](theoretical-paper-plan.md) — `∂label/∂param` is queryable. Less battle-tested than Isaac.
- **Kubric** (Google). Bullet + Blender Cycles. Renders short clips with rich annotations (depth, normals, seg, optical flow, instance masks, 3D object positions/velocities). No manipulation; great for free objects under gravity / collisions / drops.
- **TDW (ThreeDWorld)**. Unity + PhysX/Flex; supports rigid + soft + cloth + fluid. Houses Physion / Physion++ — good for eval rather than re-rendering.

### Off-the-shelf rendered corpora (zero generation cost)
Use for evaluation or as a quick smoke-test of the training pipeline.
- **Physion / Physion++** — 8 physics scenarios, pre-rendered.
- **CLEVRER, CATER** — synthetic with full ground-truth state.
- **Kubric MOVi-{A..F}** — ready datasets at increasing complexity.
- **The PhysX + Isaac Sim 800-clip evaluation set from Cosmos-Predict1 Sec 5.3** — free-fall, slopes, stable/unstable stacks, dominoes, seesaw, gyroscope (see [cosmos-predict1.md](../summaries/cosmos-predict1.md)).

### Recommended starting stack

**Default**: **Isaac Lab** for ~100k–500k clips split between manipulation (mirrors AgiBot's domain so real/sim is symmetric) and free-object scenes (drops, collisions, stacks — to densify supervision on contact-force and ω heads). Record at 30 Hz with all heads logged per frame. Augment scene appearance with **domain randomization** (textures, lighting, materials) so the encoder doesn't lock onto Isaac's PathTracer aesthetic.

**Cheap alternative** if Isaac is heavy to set up: **Kubric MOVi-E/F-style** free-object scenes + **Physion** for evaluation — operational in days, no manipulation coverage, but enough to validate the training loop.

## Open design decisions (resolve before pipeline build-out)

1. **Manipulation in scope, or just passive physics?**
   - Passive only → Kubric is sufficient; no robot URDF, no controller.
   - Manipulation → Isaac Lab or MuJoCo with a Franka/UR5/AgiBot URDF.
2. **Appearance bridging strategy.** Three options, mutually exclusive at first pass:
   - **A.** Sim as-is + heavy domain randomization. Cheapest; tests the brainstorm's actual hypothesis (encoder bridges appearance gap).
   - **B.** Sim translated to real-looking via [Cosmos-Transfer2.5](../summaries/cosmos-predict2.5.md). Reduces appearance gap before training; partially co-opts the competing paradigm.
   - **C.** Hybrid: train on (sim raw + sim translated + real), let the encoder choose what's domain-invariant.
   - Defaults to **A** because it makes the experimental claim cleanest.
3. **Pseudo-labels on real data: how much to trust?** Video Depth Anything and SAM2 are themselves models with biases — if both real and sim sim+real heads are supervised from pseudo-labels on the real side, we're partly distilling pretrained-perception biases rather than learning from data. Options:
   - Use sensor-grade real data (ScanNet++, ARKitScenes) where possible.
   - Use pseudo-labels only on the subset where sensor labels are unavailable.
   - Weight the pseudo-labeled loss term lower (e.g., λ = 0.3).
4. **Differentiable rendering?** Required for [§4 counterfactual supervision](theoretical-paper-plan.md). Genesis is the cheapest entry; alternative is Mitsuba 3 + Isaac state.

## Experimental configurations: baseline vs full

To measure whether physics supervision actually helps the encoder — rather than asserting it — train two encoder variants on the same corpus, differing only in head set:

| Config | Heads supervised | Asymmetric across domains? | Purpose |
|---|---|---|---|
| **B — Baseline (middle ground)** | depth, normals, segmentation | No — all heads have labels on real and sim | "Geometric grounding alone." No domain-detector shortcut. Strong, fair baseline. |
| **F — Full (original thesis)** | B + CoM, ω, v, contact force | Yes — sim-only heads fire on sim/translated batches only | Tests the brainstorm's actual hypothesis. Requires the domain-marginalization regularizer from [theoretical-paper-plan.md §2](theoretical-paper-plan.md) to avoid the "sim-detector" shortcut. |

Same data corpus, same architecture, same optimizer, same training compute — the *only* difference is which heads are active. Compare both on the held-out eval suite (Physion++, IntPhys 2, RoboCasa/LIBERO) and on downstream policy fine-tuning.

Three possible outcomes for B vs F:

- **F > B by a clear margin** → physics supervision is the right call; the brainstorm thesis is supported empirically.
- **F ≈ B** → geometric grounding alone is sufficient; the sim-only heads cost complexity without payoff. Report as a useful negative result and revert to B.
- **F < B** → the asymmetric-supervision shortcut beats the regularizer, or the sim-only heads actively harm transfer. Diagnose before rescuing.

Optional further variants worth running once B and F are clean:

- **F − one head** — leave-one-out ablations on CoM, ω, v, contact to identify which sim-only head carries the signal. RoboScape's analogous ablation found that keypoint-dynamics carried action-conditioning while depth was nearly flat ([../summaries/roboscape.md](../summaries/roboscape.md) Table 2); same kind of structure may show up here.
- **A — pretrained V-JEPA 2 untouched** (no auxiliary heads at all) as the floor of the comparison — the recipe the [opinion paper](../summaries/opinion-intuitive-physics.md) argues is insufficient.
- **F without translated sim** (raw sim + real only) to isolate the contribution of Cosmos-Transfer2.5 appearance bridging.

## Pipeline sketch (to be expanded once §1–§4 above are answered)

```
Real corpus:
  AgiBot-World-Beta            → RGB + actions      → SAM2 seg, Video-Depth-Anything depth, DSINE normals
  ScanNet++                    → RGB-D + meshes      → sensor depth, mesh-derived normals, sensor seg
  (optional) Ego4D / SAYCam    → RGB only           → all pseudo-labels

Sim corpus:
  Isaac Lab — manipulation     → AgiBot/Franka URDFs, 100k clips
  Isaac Lab — free objects     → drops, slopes, stacks, dominoes, 100k–400k clips
  per frame: RGB, depth, normals, instance seg, semantic seg,
             per-body {CoM_world, v, ω, contact_points, contact_forces, mass}

Augmentation:
  Domain randomization on sim (textures, lights, materials, cameras).
  (Optional) Cosmos-Transfer2.5 pass on sim for appearance-translated copies.

Train: per-head losses, mixed batches (sim+real on shared heads, sim-only on physics heads),
       domain-marginalization regularizer on sim-only-head gradients (see theoretical-paper-plan §2).
```

## Targets (placeholder — revise after pilot)
- Real: 200k–500k clips (16 frames @ 30 Hz, 320×320 cropped views).
- Sim: 200k–500k clips of equivalent length; ≥ 30 % free-object scenes to densify dynamics supervision.
- Eval (held out): Physion++ for passive physics, IntPhys 2 for violations-of-expectation, RoboCasa or LIBERO if downstream is manipulation.

## What this plan does *not* yet specify
- Per-head loss weighting and the diffusion noise schedule for scalar heads (cf. [Cosmos Policy](../summaries/cosmos-policy.md) Appx A.2.1: log-normal `σ` schedule needs adjustment for scalar targets, and [DreamerV3](../summaries/dreamer-v3.md)'s symlog/symexp for wide-dynamic-range quantities).
- Tokenization choice (per-frame ViT patches vs MAGVIT-2 video tokens vs WAN2.1 latents).
- Eval protocol (covered separately once benchmarks are committed).
