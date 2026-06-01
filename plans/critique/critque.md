# Critique: Physics-Informed Vision Embedding Proposal

## Executive Summary

The proposal has a defensible high-level motivation (video-only encoders fail IntPhys 2; simulator labels are free) and the architecture diagram borrows reasonable choices from SCOPE (cross-attention fusion) and RoboScape (multi-head auxiliary supervision). However, the research is **not thorough** — it is an idea sketch with a literature survey stapled on top. The core technical questions remain unanswered, and the closest empirical evidence (PI-JEPA) is a **negative result** for physics-loss utility.

---

## Critical Failures

### 1. The velocity/CoM heads are physically undefined from the proposed input

**This is the fatal flaw.** The architecture diagram shows a single RGB frame as input. Angular velocity (ω), linear velocity (v), and temporal change of CoM are **not observable from a single image** — they require at least two frames. This is not an engineering detail; it is a fundamental physical constraint.

The plans mention "16-frame clips at 30 Hz" but the architecture has **no temporal modeling mechanism**:
- The geometry encoder is described as "convolutions" (spatial, not temporal)
- The frozen encoder candidates (DINO, SigLIP) are single-frame models
- V-JEPA 2 *is* temporal, but it's frozen — the learned branch has no temporal structure
- Cross-attention fuses spatial patches, not temporal features

**Question the student must answer**: What is the input to the velocity heads? If it's a single-frame embedding, predicting ω and v is impossible by definition. If it's a multi-frame embedding, where is the temporal architecture?

### 2. CoM/ω/v as "per-pixel" predictions are semantically degenerate

The training-data plan says targets are "per-pixel, per-rigid-body" — every pixel of object B predicts the same (CoM, ω, v) vector for body B. This means:

- **The prediction is constant within each object segment.** A network that perfectly segments objects and assigns each a learnable constant already minimizes this loss. The physics head reduces to: "segment the image, then predict a single number per segment."
- **This is strictly less informative than segmentation + a tiny per-segment MLP.** There is no spatial structure to learn — the "per-pixel" formulation adds no geometric information beyond what segmentation already provides.
- **The gradient signal is redundant.** If object B occupies 2000 pixels, all 2000 predict the same 3-vector. The encoder gets 2000 copies of the same gradient, not 2000 independent constraints.

The plan acknowledges "CoM is spatially nonlocal" and proposes a compact-body filter, but this doesn't fix the fundamental issue — it just reduces the problem to *fewer* bodies with the same degenerate structure.

### 3. The closest prior art shows a null result for physics-residual losses

The student's own PI-JEPA summary honestly reports (Table 5):
- Operator splitting: +16%
- VICReg regularization: +17.7%  
- **PDE residual loss (the physics part): neutral — zero measurable benefit**

The summary concludes: *"physics-residual loss is hard to make load-bearing."* This is the single most relevant empirical data point for the brainstorm's hypothesis, and it is negative. The student noted this but did not explain why the proposed approach would escape the same failure mode.

### 4. The sim-only supervision problem is acknowledged but has no working solution

The domain-marginalization regularizer document correctly identifies the failure mode (encoder learns a sim-vs-real detector), proposes four candidate solutions (A–D), and then:
- Selects A+B (DANN + CORAL) as default — both are known to Goodhart
- Parks the theoretically novel one (D, Jacobian column-space) for "a later pass"
- Acknowledges "F < B" (full < baseline, i.e., physics heads *hurt*) is a realistic outcome

This is not a solved problem. It is a listed risk with a hope attached.

---

## Weak Research Claims

### 5. "Thorough research" that has explicit gaps

The theoretical-paper-plan lists under "Reading still owed":
- Multi-task identifiability / nonlinear ICA
- Equivariant SSL
- Differentiable rendering for supervision
- IRM and follow-ups
- Noether-style symmetry losses

These are not peripheral — they are the mathematical foundations required for the proposed theoretical contributions. A "thorough" literature review that hasn't read IRM (2019), the equivariant SSL literature, or the nonlinear ICA results cannot claim the proposed contributions are novel.

### 6. Zero experimental evidence

- The smoke test (`collect_smoke.py`) has "not been run end-to-end"
- No toy experiment demonstrating that predicting ω from pixels is even feasible
- No probe showing that a V-JEPA 2 embedding + linear head on ω performs above chance on even synthetic data
- No ablation on CLEVRER/Kubric (pre-rendered, free) showing any signal

The student has written 4 detailed plan documents and 14 paper summaries but executed zero experiments. The ratio of planning to doing is a red flag.

### 7. The "theoretical paper" framing dodges empirical accountability

The plan explicitly states: *"The first paper is theory-only. No experiments needed to ship it."* And then lists 5 theorems and 3 propositions — none of which have been proven or even sketched. A theory paper with no proofs is not a paper; it is a wish list.

---

## The Compute Waste Argument

The proposed workstation pilot alone (10k free-object + 5k manipulation clips, train B and F configs to convergence, leave-one-out ablations) is **weeks of GPU time** on a 4090. The cluster extension (100k–500k clips) is substantial A100/H100 allocation.

The expected outcome based on available evidence:
1. B (depth + normals + seg) will train fine — it's essentially SCOPE/RoboScape with different data
2. F (B + physics heads) will either equal B or underperform B:
   - PI-JEPA precedent: physics loss is neutral when other regularizers are present
   - The degenerate per-pixel formulation makes the physics heads trivially satisfiable
   - The sim-only domain gap makes the physics heads potentially harmful

You are spending compute to validate a hypothesis whose closest empirical analogue is a null result, using a formulation (constant-per-segment prediction from single frame) that is either physically impossible or trivially satisfiable.

---

## What Would Make This Viable

If the student wants to rescue this direction, the following must be addressed **before any compute is spent**:

1. **Define the temporal architecture.** You cannot predict velocity from a single frame. Either:
   - Use V-JEPA 2's temporal tokens (unfrozen or with temporal adapters on the geometry branch)
   - Add an explicit temporal module (3D convolutions, temporal transformer layers)
   - Reframe as optical-flow-derived supervision (but then it's not "physics-informed," it's motion estimation)

2. **Run a feasibility probe on synthetic data first.** Take CLEVRER or Kubric MOVi-E (free, pre-rendered, with ground-truth state). Train a small model to predict ω from 4-frame RGB sequences. If a 10M-parameter model can't learn this on perfect synthetic data with perfect labels, a 300M-parameter model won't learn it on mixed sim+real data with a domain gap.

3. **Abandon the per-pixel formulation for dynamics heads.** CoM, ω, v are per-object quantities. Predict them per-object (after a segmentation/pooling step), not per-pixel. This is more honest architecturally and avoids the constant-field degeneracy.

4. **Prove one theorem before planning five.** The "domain-marginalization regularizer transfer bound" (Thm 2) is the load-bearing theoretical contribution. Sketch it. If it doesn't go through, the entire theoretical paper collapses.

5. **Establish a non-trivial baseline comparison.** The real question is: does physics supervision outperform simply training on more real data with pseudo-labels (depth, normals, optical flow from off-the-shelf models)? RoboScape already showed this works. The student needs to articulate why simulator ground-truth labels are worth the domain gap they introduce.

---

## Bottom Line

The proposal is an idea with a literature survey, not a research plan with a working hypothesis. The specific mechanism by which physics quantities (ω, v, CoM) are encoded is undefined at the architectural level — and where it is defined (per-pixel, single-frame), it is either physically impossible or semantically degenerate. The closest empirical precedent (PI-JEPA) is a null result. The student should run a cheap feasibility experiment on synthetic data before committing any meaningful compute.
