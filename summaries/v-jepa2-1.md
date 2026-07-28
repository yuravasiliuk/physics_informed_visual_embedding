# V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

Mur-Labadia, Muckley, Bar, Assran, Sinha, Rabbat, LeCun, Ballas, Bardes — FAIR Meta, June 2026.
arXiv 2603.14482. [PDF](../papers/v-jepa2-1.pdf).

## TL;DR
Diagnoses *why* V-JEPA 2's frozen features are weak on dense/local tasks (22.2 mIoU
ADE20K, 0.682 RMSE NYUv2) — the predictive loss is applied **only to masked
tokens**, so context (visible) tokens have no incentive to encode local structure
and instead behave like register tokens, aggregating global info. Fixes it with:
1. **Dense predictive loss** — a distance-weighted `L_ctx` applied to context tokens
   too (weight ∝ 1/√(distance to nearest masked token)), so every token stays
   grounded in its own spatio-temporal location.
2. **Deep self-supervision** — the same loss applied at multiple intermediate
   encoder blocks via a shared multi-level predictor, not just the output layer.
3. **Multi-modal tokenizer** — native 2D conv for images / 3D conv for video
   (no more temporally-duplicating images into fake 16-frame clips).
4. **Scale** — ViT-L→ViT-G (300M→2B), VideoMix22M→VisionMix163M (adds LVD-142M
   curated images, rebalances toward motion-rich video).

Net effect: 0.682→0.307 RMSE (NYUv2 depth), 22.2→47.9 mIoU (ADE20K seg), while
*recovering* (deep self-sup) the global-task accuracy that the context loss alone
cost (SSv2 72.8→62.5→76.1→77.7 across the ablation chain). Also: +20% grasp
success over V-JEPA 2-AC, 10× faster/more-accurate navigation planning, SOTA
short-term object-interaction anticipation on Ego4D. Open-source: ViT-G/2B plus
distilled ViT-L/300M and ViT-B/80M (the distilled ViT-L gets within ~1 point of
ViT-G on most benchmarks at 1/7th the params).

## Why it matters for this brainstorm
This is not a new architecture family — it's a **retroactive diagnosis and fix of
the exact failure mode our capacity ladder is designed to detect.**
[BRAINSTORM.md §2.2](../plans/BRAINSTORM.md#22-the-capacity-ladder-each-rung-adds-capacity-subtracts-stability)'s
Rung 0 asks "is physics signal present in the frozen encoder, linearly, at any
depth?" — and this paper independently shows that vanilla V-JEPA 2's context
tokens are starved of local-structure supervision by construction (masked-only
loss → context tokens behave like global aggregators, citing the same
register-token mechanism as Darcet et al. 2023). That is a **confound on any Rung
0 result run against vanilla V-JEPA 2**: a "no signal" verdict could mean *encoder
discarded the geometry* (real negative, escape-hatch-justifying) or just *this
specific pretraining objective never asked context tokens to keep it* (an
artifact of V-JEPA 2's masked-loss design, fixed by V-JEPA 2.1). The paper's own
numbers (mIoU 22.2 vs. 47.9, depth RMSE 0.682 vs. 0.307) show the gap is mostly
the latter for dense spatial structure.

Concretely fruitful:
- **Swap the frozen-encoder default.** [README.md "Encoder candidates"](../README.md)
  and BRAINSTORM §2.2 name V-JEPA 2 (video/motion-trained) as the Rung-0 favorite
  for ω-relevant dynamics. V-JEPA 2.1 is a strict superset improvement for the
  *dense* heads (depth/normals/seg — exactly the B-config heads) and ships a
  300M-param distilled ViT-L within ~1pt of the 2B teacher, which fits the
  project's workstation-scale compute budget far better than committing to ViT-g/1B.
  Re-running Rung 0 (and Step 3b's layer-wise probe) on V-JEPA 2.1 instead of/in
  addition to 2 is now the more informative experiment — a "no signal" result
  would carry much more weight once the context-token confound is controlled for.
- **Rung 1's design is already validated, internally, by this paper.** Our
  main-line head (BRAINSTORM §2.2 Rung 1: fuse several intermediate blocks,
  attentive-pool, DPT-style decoder) is structurally what V-JEPA 2.1's *own*
  training recipe does internally (multi-level fusion MLP + deep self-supervision
  across blocks, Fig. 4–5, Table 1). Their ablation is effectively a clean
  existence proof that "intermediate-layer fusion recovers local structure the
  final embedding abstracts away" — the exact bet Rung 1 is built on — and gives a
  concrete recipe (concat normalized intermediate outputs → lightweight MLP fuse
  → multi-level predictor) to borrow directly for our head architecture, rather
  than inventing the fusion mechanism from scratch.
- **Distance-weighted loss is a directly portable idea for our own under-determined
  targets.** `λ_i ∝ 1/√(d_min(i, masked))` — weight supervision by proximity to the
  thing that's hard to determine — is the same shape as R6's
  [observability-weighted-loss.md](../plans/theory/observability-weighted-loss.md)
  (weight CoM/ω loss by angular displacement + contact). Both fixes exist because
  naive uniform-weight loss on a partially-unobservable target either collapses
  (predict-zero / predict-global-average) or destabilizes training when forced
  everywhere — worth citing as a second independent precedent for that design
  pattern, from a different model family.
- **What it does *not* touch:** no IntPhys2 / intuitive-physics number is reported,
  and nothing here addresses ω/CoM/contact-force regression — only dense spatial
  (depth/seg/tracking) and motion-*classification* (action recognition/
  anticipation) tasks. **R2/R6 (sim-only physics heads unverifiable on real,
  dynamically-unobservable-without-rich-contact targets) are completely
  untouched.** Don't over-read "V-JEPA 2.1 fixes V-JEPA's physics problem" — it
  fixes its *dense-geometry* problem. The opinion-paper/IntPhys2 argument for
  needing explicit simulator supervision (see
  [opinion-intuitive-physics.md](opinion-intuitive-physics.md)) is unweakened.

## Limitations / things to watch
- ADE20K/Cityscapes (cluttered, many small classes) remain behind DINOv3 — the
  paper attributes this to VisionMix163M having fewer highly-cluttered scenes
  than DINOv3's training mix, a data-curation gap rather than an architecture one.
- TemporalBench/TOMATO (fine-grained motion-event QA on longer videos) got
  slightly *worse* than V-JEPA 2 — the dense-feature gains don't uniformly help
  every temporal-reasoning axis.
- Context loss alone, without deep self-supervision, *trades away* global-task
  accuracy (SSv2 72.8→62.5) — the fix only works as the full recipe; cherry-picking
  just the context loss for our head design would reproduce that regression.

Code/weights: https://github.com/facebookresearch/vjepa2
