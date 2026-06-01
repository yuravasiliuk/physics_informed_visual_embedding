# Domain-marginalization regularizer

**Date:** 2026-05-21. Status: design note. Companion to [theoretical-paper-plan.md §2](theoretical-paper-plan.md) and [training-data.md](training-data.md).

## Why this exists

This regularizer is the **central sim→real bridging mechanism** of the project, not an auxiliary safeguard. The deliverable is a domain-agnostic encoder so that policies trained on `φ(X_sim)` deploy on real frames without finetuning. That works only if `φ(X) | D=sim` and `φ(X) | D=real` are aligned in distribution — which does not happen by default. This document specifies the forcing function that makes it happen.

The encoder is also trained with **asymmetric supervision**: depth/normals/segmentation have labels on both sim and real, but CoM / ω / v / contact force have labels only on sim (and translated sim). Naive masked-loss training in this setting has a well-defined failure mode that compounds the domain-gap problem:

> The encoder learns a *sim-vs-real* detector — trivially available from pixel statistics (lighting, texture, shadow softness, sensor noise) — and routes the sim-only heads through that subspace. The real branch never receives gradient on physics heads, so physics features never need to be encoded for real inputs. At inference on real frames, the physics-aware embedding is effectively untrained, and downstream sim-trained policies break.

The regularizer therefore serves two ends at once: (a) it aligns the two domain marginals so downstream policies generalize, (b) it forbids the sim-only physics heads from carving out a sim-only feature subspace they cannot generalize from.

## What the regularizer must achieve

Two statements of the same constraint:

1. **Feature-level.** The encoder's output distribution `φ(X) | D=sim` is indistinguishable from `φ(X) | D=real` — no classifier on `φ(X)` can recover the domain label better than chance.
2. **Gradient-level (the novel form, from theoretical-paper-plan §2).** The sim-only heads' gradient back through the encoder must lie in the **column space of the sim+real heads' Jacobians on the real side**. Equivalently: any direction in feature space that the physics heads push toward must also be a direction the symmetric heads care about for real images. The encoder is forbidden from carving out a "sim-only feature subspace."

Statement (1) is the operational target; statement (2) is the cleaner theoretical object and what we'd prove a transfer bound about.

## Candidate realizations

Listed by increasing novelty. The full config (`F`) in [training-data.md](training-data.md#experimental-configurations-baseline-vs-full) uses (at least) one of these; the baseline config (`B`) uses none.

### A. Domain-adversarial (DANN-style)
- Auxiliary classifier `d: φ(X) → {sim, real}` trained to predict domain.
- Encoder trained with a **gradient-reversal layer** on `d` to make the domain unpredictable from `φ(X)`.
- Cheap, well-understood, plug-in.
- **Risk:** the classifier learns weak cues and the encoder Goodharts against the classifier — sim/real become *classifier-indistinguishable* but not actually aligned. Standard issue with adversarial alignment.

### B. Moment-matching (MMD / CORAL)
- Penalize a distributional distance between `{φ(X_i) : D=sim}` and `{φ(X_j) : D=real}` within each batch.
  - **MMD** (kernel mean discrepancy) — matches all moments via a kernel.
  - **CORAL** — matches first and second moments (means and covariances).
- No adversary, no instability. But matches *marginal* distributions only, which can be satisfied trivially while still leaving a per-sample sim/real shortcut.

### C. Translated-pair consistency
- For each sim frame, render its Cosmos-translated counterpart. They differ in appearance only.
- Penalize `‖φ(X_sim) − φ(X_translated)‖²` on matched pairs.
- The *strongest* signal because it is pixel-paired — but only available where translation is available, i.e. the sim half of the corpus. Doesn't directly constrain real.
- Best treated as a complement to A or B, not a replacement.

### D. Jacobian-column-space constraint (novel, from §2)
For each batch, with sim-only head `h_phys` and symmetric heads `{h_sym}`:

- Compute `g_phys = ∂L_phys / ∂φ` (gradient of physics loss on encoder features, sim batch).
- Compute the Jacobian basis `J_sym = [∂h_sym_k / ∂φ]_k` evaluated on the *real* batch.
- Project `g_phys` onto `range(J_sym^real)`; penalize the residual.
- Operationally: the encoder is updated by physics-head gradients only to the extent that those gradients lie in directions the real-side symmetric heads also care about.

This is what would let us prove a non-vacuous transfer bound (Thm 2 in the theoretical plan). It is also the most expensive — requires Jacobian-vector products on the real batch every step. Practical version: low-rank projection via a few randomly chosen sym-head output dimensions.

### Composition
A + C is the cheap, defensible default. D is the version that earns the paper its theoretical contribution. B is a sanity-check baseline against A.

## Practical recipe (defaults for first run)

- **Batch composition.** 50/50 sim/real per step, with the sim half further split sim-raw / sim-translated. Real includes translated-pair index when available.
- **Losses.**
  - Symmetric heads (depth, normals, seg): unweighted, both domains.
  - Sim-only heads (CoM, ω, v, contact force): unweighted on sim, ignore-masked on real (see [training-data.md "Label semantics"](training-data.md#label-semantics)).
  - **Regularizer term:** `λ_dom · L_dom`, where `L_dom` is DANN+CORAL (A+B) for the first pass, with C added once translated pairs are available.
- **`λ_dom` schedule.** Warm up linearly over the first ~5% of steps from 0 → target. Adversarial term is unstable at random init; ramping avoids early collapse.
- **Target `λ_dom`.** Start at the value that drives the DANN classifier accuracy from ~1.0 (trivially separable) down to ~0.55–0.65. Below 0.55 the encoder is over-aligning and likely throwing away useful structure; above 0.7 the regularizer is too weak.
- **Per-head loss weights** stay as in the corpus design — the regularizer does not replace head loss balancing, it operates orthogonally.

## Diagnostics — is the regularizer actually working?

Track these every N steps; cheap, all of them:

1. **Domain classifier accuracy** on a held-out probe over `φ(X)`. Target: ~0.5–0.65. If it stays near 1.0, the regularizer isn't biting.
2. **Real-side physics head probes.** Freeze the encoder periodically, train a *linear* head on real frames with whatever weak labels are available (depth-derived contact heuristics, video-derived ω estimates). If F+regularizer's linear-probe error is lower than B's, the masked-loss strategy is transferring.
3. **Sim/real centroid distance** of `φ(X)` after PCA. Should shrink and stabilize, not collapse to zero (zero ⇒ representation has lost domain-relevant variation that may also be task-relevant).
4. **Symmetric-head loss on real.** Must not degrade vs. config B. If it does, the regularizer is harming the heads that *do* have real labels, which means it's the wrong regularizer or `λ_dom` is too high.
5. **Failure trip:** if (1) is ~0.5 *and* (4) degrades, the encoder is collapsing — domain-invariance achieved by losing information. Lower `λ_dom` or add C.

(3) and (5) are the most informative — they catch the "trivial alignment" failure mode that A is prone to.

## Open questions

- **Does (D) reduce to a known IRM/IB form?** Listed as Q2 in [theoretical-paper-plan.md §"Open questions"](theoretical-paper-plan.md). Worth checking before claiming novelty — it has a flavor of IRM's gradient-penalty formulation with a partial-label twist.
- **Translated-pair scope.** If Cosmos-Transfer translates only a fraction of sim, does C generalize to the untranslated portion or only locally smooth the matched neighborhood?
- **Interaction with the counterfactual-differential supervision (§4 of the theoretical plan).** First-order labels also flow through the encoder; do they need their own column-space constraint, or is the head-level one sufficient?
- **What's the right baseline for ablation?** The cleanest comparison is F-with-regularizer vs F-without, both run on the same corpus. F-without is the diagnostic for whether the shortcut actually happens at our data scale — possibly it does not, and the regularizer is theoretical hygiene rather than necessity.

## Connection to the rest of the plan

- The regularizer is the **load-bearing sim→real bridging mechanism** for the whole project, not just config F. Even config B benefits: without the regularizer, B's encoder is free to specialize on sim or real pixel statistics, and a policy trained on `φ(X_sim)` will not deploy on real frames. Run the regularizer on both configs.
- The regularizer is *additionally* the load-bearing assumption behind config **F** in [training-data.md](training-data.md#experimental-configurations-baseline-vs-full). Without it, F is expected to underperform B because the sim-only heads carve out a sim-only feature subspace.
- The transfer bound (Thm 2 in [theoretical-paper-plan.md](theoretical-paper-plan.md)) is what gives the regularizer formal justification; this file is the engineering side of that theorem.
- The diagnostics above feed the B-vs-F decision rule: if F (with regularizer) ≤ B, either the regularizer failed or the shortcut wasn't really happening — both are publishable results.
- **Downstream policy deployment** is the ultimate eval: train a small sim-only policy on `φ(X_sim)` for a controlled task (e.g. RoboCasa pick-place), deploy on real frames, measure success rate. Domain classifier accuracy on `φ(X)` is a *proxy* — policy transfer is the truth.
