# CrossJEPA — Cross-Modal Joint-Embedding Predictive Architecture for Efficient 3D Representation Learning from 2D Images

Perera, Hewagamage, Nazar, Abeywardana, Gallella, Rodrigo, Afham. arXiv 2511.18424v1, Nov 2025 (U. of Moratuwa / TU Darmstadt). [PDF](../papers/crossJEPA.pdf).

## TL;DR
Pretrain a small (8.5M-param) **point cloud** encoder by predicting **frozen 2D image embeddings** of rendered views of the same object. The predictor is conditioned on the view's **camera pose** (yaw/pitch, sinusoidal) and a **48-D RGB color histogram** of the target view — these act as a *latent-information sink* that absorbs teacher-specific nuisance gradients (pose, color) so the 3D encoder learns only the shared, geometric semantics. No masking, no contrastive pairs, smooth-L1 in embedding space, frozen-teacher embeddings cached once. SOTA on ModelNet40 (94.2 %) and ScanObjectNN OBJ-BG (88.3 %) at 14.1M total params and ~6 GPU-h on a single 4090.

## Relevance to the brainstorm — short answer
**Tangential as a baseline, but high-value as a design pattern.** The end product (a 3D point cloud encoder distilled from 2D) is the inverse of the brainstorm's goal (a 2D RGB encoder informed by simulator/3D signal). The brainstorm cannot reuse the pipeline directly. However, three of CrossJEPA's design choices are directly transferable and address open questions in the brainstorm:

1. **Latent conditioning as a "nuisance sink"** for the predictor (§3.1, Tab. 6) is the most novel idea in the paper and the one most worth porting.
2. **JEPA without masking** is explicitly defended both empirically (Tab. 7) and theoretically (§6, Suppl. §D) — useful ammunition since the brainstorm also doesn't naturally fit a masking paradigm.
3. **Caching frozen-teacher embeddings** is an obvious-in-hindsight efficiency win the brainstorm should adopt if the semantic branch (DINOv2 / V-JEPA2 / SigLIP) is truly frozen.

The brainstorm's strongest direct baseline remains [scope.md](scope.md) and [why-auxiliary-tasks-jepa.md](why-auxiliary-tasks-jepa.md). CrossJEPA is a *methodology* contribution, not a target baseline.

## Method, in one diagram
```
point cloud p  ──► P (ViT-style, learnable, 8.5M) ──► s_p ─┐
                                                           ├─► H_p2i ──► ŝ_i^[z_v]   ≈   s_i^[z_v]
rendered view i^[z_v]  ──► I (frozen DINOv2 ViT-B/14) ──► s_i^[z_v] (cached)
                                                           │
              z_v = (yaw, pitch, color-hist of view v)  ───┘   (latent info)
```
Loss `L_p2i = smooth-L1(ŝ_i^[z_v], s_i^[z_v])` summed over a subset of 36 precomputed views per object, over all objects. Pretrain on ShapeNet + 160k Objaverse-XL renders.

## What's actually new
JEPA literature has assumed masking is intrinsic (I-JEPA, V-JEPA, Point-JEPA all mask). CrossJEPA argues this is a **misconception inherited from I-JEPA's first instantiation**: LeCun's original JEPA framework needs only an EBM in latent space, not masking. In cross-modal settings, naively masking across modalities is *non-trivial and induces inconsistencies* (e.g. masked image regions don't correspond to point-cloud regions in any clean way). They replace masking with a **view-prediction task**: predict the embedding of a specific 2D view from the full 3D input, conditioned on that view's pose.

## The latent-information sink — the idea worth stealing
Standard distillation forces the student to predict the *full* teacher representation, including factors the student literally cannot know (e.g. exact 2D pose, exact RGB color of a render). These factors enter the gradient as **noise** that the student can never resolve, and they slowly corrupt the semantically-useful signal.

CrossJEPA's fix: feed those factors as side input `z_v` directly to the **predictor** (not the encoder). Intuitively, this changes the question from *"predict the image embedding of this car"* to *"predict the image embedding of this car **from yaw=30°, pitch=15°, with this color histogram**."* Mechanically, gradients corresponding to pose/color now flow into `z_v` (a sink) instead of back through the point encoder. Result (Tab. 6):

| Latent info             | Acc % | Saturation epoch |
|-------------------------|-------|------------------|
| None                    | 93.68 | 93               |
| + Camera pose           | 93.84 | 71               |
| + Pose + color hist.    | 94.04 | 67               |

So latent conditioning **+0.36 pp accuracy and 1.4× faster convergence**.

**Why this matters for the brainstorm.** The brainstorm proposes RGB+geometry → cross-attention fusion with a frozen semantic encoder → multiple sim/real heads. The *same nuisance problem* exists:
- Predicting depth/normals/segmentation from RGB shouldn't require the geometry encoder to memorize the specific lighting or camera intrinsics of the synthetic renderer.
- Predicting CoM / contact forces is an embedding-space target — if the frozen semantic branch carries lighting/style nuance, the geometry encoder may waste capacity reconstructing it.

Concrete port: in the fusion/predictor stage, **inject camera intrinsics, light direction, and a global color/exposure summary as conditioning tokens**. This should let the geometry encoder forget renderer-specific style and focus on shape/physics — directly addressing the sim2real gap. This is also consistent with [why-auxiliary-tasks-jepa.md](why-auxiliary-tasks-jepa.md)'s message that head choice defines what the embedding must distinguish: latent conditioning is the dual, defining what it is *allowed to ignore*.

## Other empirical knobs worth noting
- **Masking ratio (Tab. 7).** Accuracy degrades monotonically with mask ratio: 0 → 93.96, 0.3 → 93.84, 0.5 → 93.55, 0.75 → 93.51. In the cross-modal JEPA setting, masking hurts.
- **Parallel views (Tab. 8).** Predicting 1, 6, 12, 18, 24, 30, 36 views simultaneously is roughly flat in accuracy; sweet spot 6–12 views balances redundancy vs. per-epoch cost.
- **Model scaling (Tab. 9).** Small (6.4M) 93.7 → Base (14.1M) 93.8 → Large (24.8M) 94.0. Diminishing returns; the contribution lives in the *recipe*, not in scaling.
- **Teacher choice.** Frozen DINOv2 ViT-B/14. They emphasize DINOv2's spatial consistency; same reasoning as SCOPE — relevant if the brainstorm picks a frozen semantic backbone.

## What CrossJEPA does NOT contribute
- No physics labels, no simulator-derived signal, no contact / CoM / depth targets. It is geometric-semantic only.
- The 2D-to-3D direction is *opposite* to the brainstorm's 2D-encoder goal. If anything, CrossJEPA argues that 2D embeddings already carry enough geometry to supervise a 3D model — which mildly *weakens* the brainstorm's argument that visual data alone is insufficient (compare [opinion-intuitive-physics.md](opinion-intuitive-physics.md)). The reconciliation: CrossJEPA evaluates on *category-level shape classification*, not intuitive physics. The opinion paper's "visual data is insufficient" claim is about *dynamics / permanence / solidity*, not shape — both can be true.
- Single-task evaluation (classification + part-seg). No multi-head supervision, no dynamics, no embodied evaluation.

## Limitations relative to the brainstorm
- Fixed projection parameters (yaw/pitch only, roll = 0). Conditioning vocabulary is small and discrete; the brainstorm would need a richer `z` (intrinsics, light direction, exposure, possibly material).
- Color histogram is a hand-engineered nuisance descriptor — would need to be replaced with something more general for the brainstorm's natural-image setting (e.g. learned style token, or simple per-channel mean/std).
- Object-centric pretraining (single CAD-model renders). Real-scene applicability not shown.

## Pointers worth chasing
- TI-JEPA [80], A-JEPA [30], PrediCIR [71] — other cross-modal JEPA flavors cited in §2.
- Pic@Point [40] — closest concatenation-style image-point baseline; CrossJEPA beats it.
- Suppl. §D — predictive-coding / information-theoretic justification for the nuisance sink. Could feed [theoretical-paper-plan.md](../plans/theory/theoretical-paper-plan.md) if it formalizes the "noise decoupling" claim.

Code: stated as forthcoming, not yet released as of v1.
