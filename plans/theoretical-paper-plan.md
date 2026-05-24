# Plan — Theoretical Paper on Physics-Supervised Visual Encoders

**Date:** 2026-05-13. Status: outline; theory-only first pass.

## Working title (placeholder)
*Multi-Head Physics Supervision for Visual Encoders: Identifiability, Symmetry, and Sim-Asymmetric Generalization.*

## What's already in the literature (anchors)
- [SCOPE](../summaries/scope.md) — frozen-DINO + RGB+normals **input** branches, cross-attn fusion. Architectural baseline, but normals are *inputs*, not heads.
- [Why and How Auxiliary Tasks Improve JEPA](../summaries/why-auxiliary-tasks-jepa.md) — single auxiliary head defines an equivalence relation on latents; "No Unhealthy Collapse" theorem.
- [PI-JEPA](../summaries/pi-jepa.md) — physics-as-residual (PDE) on a known operator; honest negative that raw residual loss is neutral, gains come from operator-splitting and VICReg.
- [V-JEPA 2](../summaries/v-jepa2.md) and the [intuitive-physics opinion paper](../summaries/opinion-intuitive-physics.md) — video at scale still fails [IntPhys 2](../summaries/intphys2.md) (96 % human vs 56 % best model).
- [Cosmos-Transfer2.5](../summaries/cosmos-predict2.5.md) — Sim2Real-by-pixel-translation, the *competing paradigm* to latent alignment.

## Why we expect to escape PI-JEPA's null result

PI-JEPA is the closest negative empirical precedent: their ablation (Table 5 in the paper, summary in [pi-jepa.md](../summaries/pi-jepa.md)) reports the PDE-residual loss as **neutral** on top of operator splitting and VICReg. Any plan that pattern-matches to "physics loss bolted onto JEPA" inherits this null until argued otherwise. The argument follows.

**Different signal class.** PI-JEPA's physics loss is a *residual* on a known operator: the encoder must produce a latent that, when decoded back to physical fields, satisfies a PDE up to autodiff'd residual. This is supervision on a *constraint*, not on an *observable*. Residual losses are easy to satisfy trivially when the residual operator has a large null space (any encoder mapping into the residual's kernel pays no cost). The brainstorm's heads are *direct regression onto observable physical quantities* (CoM in camera frame, ω in camera frame, contact force magnitudes per body). There is no null space — the target is a real number, the prediction is a real number, the loss is L2. The PI-JEPA failure mode does not transfer mechanically.

**Closer empirical analogue: RoboScape.** RoboScape's keypoint-dynamics auxiliary head — a *direct regression* on per-keypoint motion derived from a pretrained perception model — is the right reference point, and it is **positive**: dropping it costs ~21 % on action-conditioning MSE ([roboscape.md](../summaries/roboscape.md) Table 2). Our heads sit in the same family (direct-regression on per-object physical state), so we should expect RoboScape's positive result to be the prior over PI-JEPA's null.

**What this commits us to predicting.** If RoboScape is the right analogue, the F − B gap should come **from dynamics heads (CoM, ω, v, contact), not from depth / normals / seg** — RoboScape's own ablation found depth nearly flat next to dynamics. Translating to our F-vs-B design ([training-data.md](training-data.md#experimental-configurations-baseline-vs-full)):
- Geometric heads stabilize the encoder but should not drive F > B by themselves.
- Per-head leave-one-out on F (the "F − one head" ablation) should show one or two dynamics heads carrying the gain; if every head matters equally, the result is suspect.
- If F ≈ B *and* the per-object dynamics linear probe (the Phase 0.5 diagnostic) is near-chance on held-out scenes, we are in PI-JEPA's regime, not RoboScape's — the head set is too weak to differentiate the encoder. Pivot.

**Honest fallback.** PI-JEPA's authors recovered their gains via *operator splitting* (architecture) and *VICReg* (collapse prevention), not the physics loss. The architectural analogue here is the multi-head split itself (one head per physical quantity, no shared trunk after fusion) — which we already do. The collapse-prevention analogue is the domain-marginalization regularizer, which we also do but for a different reason. If the dynamics heads contribute nothing, the project's residual value is exactly these two architectural pieces, framed honestly as "what survives the null."

## Gaps we can theoretically close (beyond "physics as prediction head")
The initial brainstorm idea is: add sim-only heads (CoM, ω, v, contact) on top of sim+real heads (normals, segmentation, depth). What the existing theory does *not* cover, and what we can add:

### 1. Multi-head equivalence composition
Current theory ([Why-Aux-JEPA](../summaries/why-auxiliary-tasks-jepa.md)) handles **one** auxiliary head: each head induces a partition of the input space (equivalence classes of "same-label" inputs). With *K* simultaneous heads the encoder is constrained to the **meet** of those partitions. Open:
- When is the meet *strictly finer* (i.e. heads complementary), and when is it redundant?
- Information-theoretic lower bound on the latent dimension as a function of the heads.
- Generalization of "No Unhealthy Collapse" to multi-head with possibly conflicting equivalence relations.

### 2. Domain-asymmetric supervision (sim-only heads)
Half the heads are supervised only on sim. Existing theory implicitly assumes labels are available everywhere. Risks:
- Encoder learns a "sim-vs-real" detector, then routes sim-only heads through a shortcut.
- Sim-only heads bias the embedding toward sim statistics → catastrophic transfer.

Theoretical contributions to make:
- Cast the setup as **two-environment IRM / invariant risk** with a *partial-label* twist: sim+real heads are invariant signals, sim-only heads are *risky* but *strong*.
- Prove conditions (head Lipschitzness, domain-density overlap, encoder capacity split) under which sim-only heads still yield non-negative real-domain transfer.
- Propose a **domain-marginalization regularizer**: sim-only head's gradient w.r.t. encoder must be in the column space of the sim+real heads' Jacobians on the *real* side. This is novel as far as I can tell and is a direct theoretical justification of the brainstorm architecture.

### 3. Equivariance structure of physical heads
The heads are not arbitrary — they carry **group structure**:
- Depth, normals: SE(3) on the camera frame.
- CoM: SE(3) on the world frame.
- Angular velocity ω: SO(3) under world rotation, *invariant* under translation.
- Contact forces: covariant with surface normal frame.
- Segmentation: SE(3)-invariant (up to occlusion).

No paper in the read set treats the encoder + heads as an equivariant pair. Open:
- Define the encoder as an equivariant map φ: X → Z under a known group action (camera SE(3), world SO(3) for orientation-defined quantities).
- Prove: equivariant multi-head supervision strictly increases sample efficiency over invariant supervision by a factor tied to the symmetry-group volume (cf. Elesedy & Zaidi 2021).
- Identify the *minimum* equivariance structure that is sufficient for the head set — likely SE(3)_camera × SO(3)_gravity (gravity breaks rotational symmetry to a vertical-axis subgroup).

### 4. Counterfactual differential supervision (this is the most distinctive new piece)
The simulator is *differentiable* in physical parameters (mass m, friction μ, restitution e, …). For any clip we can render `(I, label)` *and also* a perturbed `(I', label')` from `(m+δm, …)`. The available signal is therefore not just labels but **derivatives**:
- `∂label / ∂parameter` (sensitivity of physical quantity to physical parameter, ground-truth).
- `∂I / ∂parameter` (sensitivity of the rendered image — available from differentiable rendering).
- Chain rule: encoder should satisfy `∂φ(I)/∂param · ∂head/∂φ ≈ ∂label/∂param`.

This is *first-order auxiliary supervision* on top of standard (zeroth-order) auxiliary supervision. To my knowledge, no paper in the read set uses this. It is a strict superset of the brainstorm's "physics as prediction head" idea and admits a clean theory:
- Information gain of first-order labels vs zeroth-order labels (Fisher-information argument).
- When the rendered ∂I/∂param is degenerate (e.g. friction in a static frame), the supervision contributes nothing — this gives a principled head-pruning rule per frame.

### 5. Conservation laws as constraints, not regressions
CoM, momentum, energy are conserved across time. Predicting them at each frame independently throws away the strongest signal: the **derivative** of CoM is the linear momentum, divided by mass. A theoretical contribution:
- Encoder is constrained to a manifold where time-derivative of predicted CoM matches predicted momentum / mass.
- Cast as Noether-style symmetry: time-translation invariance ↔ energy conservation.
- Prove that the conservation-law constraint reduces the function space more than a pointwise loss with equivalent supervision budget.

### 6. Bias–variance trade-off: latent alignment vs pixel translation for Sim2Real
[Cosmos-Transfer2.5](../summaries/cosmos-predict2.5.md) is a direct competitor. We can frame both as estimators of P(Y_real | X_real):
- **Translation:** `X_sim → X_real_like → encoder → Y`. Variance from the translator, bias from translator's distributional gap.
- **Latent alignment:** `X_sim → encoder → Y` with auxiliary heads tying φ(X_sim) to φ(X_real). Variance from sim/real label asymmetry, bias from the encoder's domain-confusion.

Quantify the regimes (small/large sim corpus, fidelity of translation, head budget) under which one beats the other. This is a clean theoretical addition that **does not require any experiments** and immediately positions the paper against the dominant industry approach.

## What we are *not* trying to prove
- That physics-prediction heads work in practice (deferred to a separate experimental paper).
- That the proposed architecture beats Cosmos-Transfer on a benchmark.
- Anything specific to a chosen encoder (DINO/V-JEPA/SigLIP); the theory should be encoder-agnostic.

## Proposed paper structure
1. **Introduction.** Pose the visual-physics gap (IntPhys 2 + Opinion paper). Frame the trade-off space: input-side conditioning (SCOPE) vs output-side supervision (the brainstorm) vs pixel translation (Cosmos-Transfer).
2. **Setup and notation.** Encoder φ: X → Z. Head set {h_k: Z → Y_k} partitioned into sim+real and sim-only. Domains D_sim, D_real. Group actions G_camera, G_gravity.
3. **Multi-head equivalence theory** (gap §1). Lift the JEPA aux-task theorem; characterize the meet of equivalence relations; identifiability statement.
4. **Domain-asymmetric supervision** (gap §2). Two-environment IRM with partial labels; statement of the domain-marginalization regularizer; transfer bound.
5. **Equivariance** (gap §3). Sample-complexity gain; identification of the minimum group; how to handle gravity symmetry-breaking.
6. **First-order supervision** (gap §4). Fisher-info gain; degeneracy detector; relation to differentiable rendering.
7. **Conservation laws** (gap §5). Encoder-on-manifold formulation; Noether interpretation.
8. **Sim2Real estimator comparison** (gap §6). Bias-variance theorem; regime diagram (small/large data, weak/strong translator).
9. **Discussion.** Limits of theory, what remains empirical, connection to world-model RL ([DreamerV3](../summaries/dreamer-v3.md), [V-JEPA-2-AC](../summaries/v-jepa2.md)).

## Theoretical deliverables (concrete prop/theorem checklist)
- **Prop 1.** Multi-head meet identifiability — when φ is identifiable up to a residual symmetry given K heads.
- **Thm 1.** Multi-head no-unhealthy-collapse extension.
- **Thm 2.** Sim-only-head transfer bound under domain-marginalization regularization.
- **Prop 2.** SE(3)_camera × stab(g)_world is the minimum equivariance group for the proposed head set.
- **Thm 3.** Sample-complexity improvement from equivariant encoder vs invariant encoder, with the proposed heads.
- **Prop 3.** First-order supervision strictly increases Fisher information when ∂I/∂param is non-degenerate; explicit degeneracy criterion.
- **Thm 4.** Conservation-law constraint reduces VC-dimension of the encoder hypothesis class.
- **Thm 5.** (Bias–variance) Regime in which latent-alignment Sim2Real dominates translation Sim2Real.

## Open questions to resolve before writing
1. Is there an existing multi-task identifiability result (Khemakhem et al. iVAE family, or recent multi-task ICA) we can lift instead of proving from scratch?
2. Does the domain-marginalization regularizer reduce to a known IRM/IB form?
3. Is the differentiable-rendering assumption (gap §4) realistic given Isaac Sim / Genesis / Cosmos pipelines, or should we present it as an aspirational primitive?
4. Are there published equivariant-multi-task theorems we should cite or extend (likely yes — equivariant SSL recent literature, not in the current reading pile)?

## Reading still owed
Before writing, fill these holes (none currently covered in [summaries/](../summaries/)):
- Multi-task identifiability / nonlinear ICA (Khemakhem 2020, Hyvärinen 2023+).
- Equivariant SSL (Garrido et al. 2023+, Elesedy 2021).
- Differentiable rendering for supervision (Mitsuba 3, Genesis, Nvdiffrast).
- IRM and follow-ups (Arjovsky 2019; RICE / V-REx).
- Noether-style symmetry losses in deep learning (Liu et al. 2023+).

## Sequencing
1. Resolve the four open questions above (literature pass, ~1 week).
2. Draft §3, §5, §6 first — they have the cleanest theoretical hooks.
3. §4 (first-order supervision) is the **distinctive** contribution; write it second so it gets the most polish.
4. §2 (multi-head equivalence) and §7 (conservation) last — they are extensions of known results.
5. Defer experiments entirely; the empirical paper is a separate project.
