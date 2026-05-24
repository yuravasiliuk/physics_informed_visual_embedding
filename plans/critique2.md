# Critique v2: Physics-Informed Vision Embedding Proposal (Post-Revision)

**Date:** 2026-05-23.

## Status of Previously Identified Issues

### ✅ ADDRESSED: Temporal architecture (was "Fatal Flaw #1")

The student switched from DINO/SigLIP (single-frame) to **V-JEPA 2** (native spatiotemporal tokens) as the frozen branch, specified K-frame clip inputs (K=4 → K=16), and added a (2+1)D or 3D conv U-Net as the learned temporal branch. Dynamics heads now consume "clip-level features" with targets at the last-frame slice.

**Verdict:** The physics impossibility is resolved. ω and v are now observable in principle from the multi-frame input. The architecture has a temporal pathway.

### ✅ ADDRESSED: Per-pixel degeneracy (was "Critical Failure #2")

The student explicitly abandoned per-pixel predictions for dynamics heads. The new formulation: masked-pooling encoder features over the body's predicted segmentation mask → MLP → per-object (CoM, ω, v, contact). The plan explicitly notes the old per-pixel approach was "degenerate (constant within each segment), satisfiable by `seg → MLP(seg_id)` without any geometric or physical learning."

**Verdict:** Correct fix. Per-object prediction is the only defensible formulation.

### ✅ PARTIALLY ADDRESSED: PI-JEPA escape argument (was "Critical Failure #3")

The student added a section distinguishing direct-regression heads (their approach) from residual-constraint heads (PI-JEPA), and citing RoboScape's keypoint-dynamics ablation as the closer positive analogue. The argument: PI-JEPA's PDE residual has a large null space (any encoder mapping into the residual's kernel pays no cost); direct regression on an observable has no null space.

**Verdict:** The null-space argument is technically correct and is a real structural difference. However, the RoboScape comparison is misleading in a specific way — see New Issue #2 below.

### ⏸️ DEFERRED (acknowledged): Domain-marginalization transfer bound

### ⏸️ DEFERRED (acknowledged): Four missing-lit items

---

## New and Remaining Critical Issues

### 1. The learned branch is architecturally redundant with V-JEPA 2

V-JEPA 2 is a 1B-parameter ViT-g trained on 22M videos that already captures temporal dynamics well enough for **zero-shot closed-loop robot control** (65–80% success on grasp/pick-and-place per the student's own v-jepa2.md summary). The student proposes adding a separate (2+1)D conv U-Net that processes the same K-frame clip, then cross-attends to V-JEPA 2's features.

**The question that must be answered:** What does the learned branch extract that V-JEPA 2 doesn't already encode?

- If V-JEPA 2's frozen features already contain the information needed for ω/v/CoM prediction (likely, given it enables closed-loop control), then the dynamics heads are essentially **linear probes on V-JEPA 2's existing representation**, with the learned branch as an over-parameterized adapter. This is fine as an experiment — but it means the contribution is "V-JEPA 2 already encodes physics, here's a probe showing it" rather than "physics supervision teaches the encoder something new."
- If V-JEPA 2's features do NOT contain physics information (its IntPhys 2 failure suggests this), then the learned branch must extract physics from raw pixels independently. But the cross-attention fusion means the learned branch's output is modulated by V-JEPA 2's features — it cannot easily learn something V-JEPA 2 missed because its queries are shaped by V-JEPA 2's representation.

**The architecture has an identity crisis:** is it fine-tuning V-JEPA 2 through a side-channel (the learned branch as adapter), or is it building an independent physics encoder that merely consults V-JEPA 2 for semantics? The plan doesn't clarify which, and the answer changes the entire experimental interpretation.

**What this means concretely:** Before Phase 0.5, run a **V-JEPA 2 linear probe baseline** on Kubric MOVi-E: freeze V-JEPA 2, train a linear head on its spatiotemporal tokens to predict per-object ω. If this already works (R² > 0.5), the entire learned branch + cross-attention architecture is solving a problem that doesn't exist. The Phase 0.5 probe as currently designed (train a fresh 30M-param encoder) doesn't test this — it tests whether dynamics are learnable from video *in general* (obvious yes), not whether the *specific architecture adds value over V-JEPA 2 alone*.

### 2. The RoboScape analogy is weaker than presented — domain gap invalidates the comparison

The plan cites RoboScape's positive ablation (dropping keypoint-dynamics costs ~21% on action-conditioning) as evidence that dynamics heads work. But there is a critical disanalogy:

- **RoboScape's keypoint-dynamics head trained on BOTH sim and real data.** Labels came from pretrained perception models (SpatialTracker), available on all training videos. There was no domain-asymmetric supervision.
- **The brainstorm's dynamics heads train on SIM ONLY.** Labels come from the simulator and do not exist for real data.

RoboScape never faced the domain-asymmetric supervision problem. Its positive result says: "dynamics heads help **when you have labels everywhere**." The brainstorm's situation is: "dynamics heads help **when you have labels only on one domain and must generalize to the other**." These are different claims.

The honest prior is: RoboScape shows the *signal* is useful (direct-regression dynamics carry action-conditioning); the *delivery mechanism* (sim-only labels + regularizer) is untested. The project's actual hypothesis is not "are dynamics heads useful?" (yes, per RoboScape) but "can sim-only dynamics labels transfer to a real-domain encoder via alignment?" — and for this, there is no positive precedent in the reading pile.

### 3. Masked-pooling creates a hard bootstrapping problem

The new per-object formulation: dynamics heads pool encoder features using the **predicted segmentation mask**, then an MLP predicts (CoM, ω, v). This creates a circular dependency:

1. **Early in training, the segmentation head is inaccurate.** Masks are noisy, bleed across objects, miss small bodies.
2. **Noisy masks → noisy pooled features → noisy dynamics targets → noisy gradients from dynamics heads back to the encoder.**
3. **The dynamics heads cannot provide useful gradient signal until segmentation is good.** But segmentation quality depends on the encoder, which the dynamics heads are supposed to improve.

This is a **bootstrapping problem** the plan does not address. Possible mitigations:
- Use ground-truth segmentation masks during early training (available in sim). But this means the dynamics heads never learn to be robust to segmentation errors, and on real data (where you'd use predicted masks), performance degrades.
- Pre-train the segmentation head to convergence before enabling dynamics heads. But this changes the training protocol from "all heads jointly" to "staged," and the "meet of equivalence relations" theory assumes simultaneous joint training.
- Use a separate frozen segmentation model (SAM2) to provide masks. This removes the dependency but also means the dynamics heads aren't actually training the encoder's segmentation ability — they're free-riding on an external model.

None of these are mentioned. The plan implicitly assumes segmentation and dynamics heads train simultaneously from scratch without interference.

### 4. Contact force prediction is architecturally unsupported and should be dropped

Contact force is by far the hardest head and receives the least architectural discussion. Predicting contact force between bodies requires:
- Knowing **which** bodies are in contact (contact detection from video — itself an open problem)
- Knowing the **contact normal** (requires 3D geometric understanding at the contact point)
- Knowing the **relative velocity** at the contact point (not the same as body CoM velocity for extended bodies)
- Material properties (friction, restitution) that are not observable from RGB

The plan acknowledges sparsity (most frames have zero contact) and applies a γ-multiplier. But the fundamental issue is that contact force is not a function of the encoder's input in any meaningful sense — it depends on material properties, constraint solver internals, and collision geometry that cannot be inferred from appearance alone.

**Evidence:** No paper in the reading pile (RoboScape, PI-JEPA, SCOPE, V-JEPA 2) attempts to predict contact forces from video. RoboScape's closest analogue is keypoint consistency near high-motion regions — a much weaker (and achievable) surrogate.

**Recommendation:** Drop contact force from the head set. It will either: (a) collapse to "predict zero always" (the loss is minimized by the prior, since most frames are contact-free), or (b) if γ is large enough to overcome the sparsity, it will inject noisy gradients that harm the encoder. Neither outcome helps the project.

### 5. Compute feasibility of V-JEPA 2 in the training loop is not verified

V-JEPA 2 (ViT-g/16) has ~1B parameters. Running inference on 16-frame clips at 320×320:
- 16 frames × (320/16)² patches/frame = 16 × 400 = 6400 tokens through a 1B-param transformer
- Even frozen (no gradients stored for V-JEPA), this forward pass requires significant memory for activations
- The learned branch ((2+1)D conv U-Net, trainable) + 7 heads + backprop through all of this

On a single RTX 4090 (24GB VRAM), this is likely **infeasible** at the proposed clip length and resolution without:
- Aggressive gradient checkpointing on the learned branch
- Mixed precision throughout
- Batch size = 1 or 2 (which makes batch-norm-based heads and the DANN domain classifier unstable)
- Or reducing K from 16 to 4 (which reduces the temporal context available for dynamics)

The plan states "Single workstation, 1–2 GPUs" as the compute budget. A quick memory estimate should have been done before committing to V-JEPA 2 at this scale. If the architecture doesn't fit, the whole plan needs revision — you can't just "try a smaller K" because smaller K means less temporal context means harder dynamics prediction.

### 6. Phase 0.5 is necessary but tests the wrong hypothesis

Phase 0.5 as designed: "Train a small (≤30M-param) clip-level encoder + per-object dynamics heads on Kubric MOVi-E, 4-frame RGB inputs, ground-truth per-body ω/v/CoM targets."

This will almost certainly **pass**. Dynamics are learnable from video on clean synthetic data — this has been known since at least CLEVRER (2020). Optical flow + rigid-body tracking gives you ω and v analytically; a neural network can learn this regression.

**What Phase 0.5 actually tests:** "Can a 30M-param model regress ω from video?" → Yes, trivially.

**What it should test:** "Does the proposed full architecture (V-JEPA 2 frozen + learned branch + cross-attention + masked-pool + 7 heads) learn anything that V-JEPA 2 + a linear probe does not?" → This is the actual research question.

The Phase 0.5 pass criterion (ω R² > 0.5) is too easy. The bar should be: "Does the full architecture's ω probe on held-out data beat V-JEPA 2 linear probe by a statistically significant margin?" If it doesn't, the entire learned branch + physics heads architecture is overhead.

### 7. The "honest fallback" reveals the project has no unique value proposition under the null

The plan says: *"If the dynamics heads contribute nothing, the project's residual value is exactly these two architectural pieces [multi-head split and domain-marginalization regularizer], framed honestly as 'what survives the null.'"*

But:
- The multi-head split is SCOPE/RoboScape architecture — not novel
- The domain-marginalization regularizer is the deferred item with no proof and no implementation

So under F ≈ B (the most likely outcome per available evidence), the project's residual value is: existing architecture + unproven theory. That is not a publishable contribution. The plan acknowledges F ≈ B as realistic but doesn't acknowledge that this outcome makes the project unpublishable, not just "a useful negative."

A useful negative requires a clean, well-motivated hypothesis that *surprisingly* fails. "Physics heads might help but probably won't, and our closest prior says they don't" is not a surprising negative — it's a confirmation of the prior.

### 8. The two-paper strategy still has zero deliverable progress toward either paper

- **Theory paper:** 5 theorems, 3 propositions. Zero proofs sketched. No LaTeX. No lemma structure. The plan is a table of contents for a paper that does not exist.
- **Empirical paper:** Zero experiments run. No code beyond `collect_smoke.py` (never executed). No model trained. No single number reported.
- **What exists:** 5 plan documents (now ~4000 words each), 14 paper summaries, 1 architecture PDF.

The ratio of planning to doing has gotten **worse**, not better, since the last review. The student responded to a critique about over-planning by... producing more detailed plans. This is the wrong response. The correct response was: run Phase 0.5 on Kubric (1 GPU-day), report a number.

---

## What Has Actually Improved

To be fair, the revised plans are significantly better-structured than before:

1. **The architecture is now physically coherent.** Temporal input → temporal encoder → per-object dynamics. This is learnable in principle.
2. **The per-object formulation is correct.** Masked-pool + MLP avoids the constant-field degeneracy.
3. **The PI-JEPA differentiation is legitimate.** Direct regression ≠ residual constraint, and the null-space argument is real.
4. **The Phase 0.5 gate is the right kind of thinking.** "Don't spend Isaac Sim compute until you know the task is learnable on free data" is good engineering discipline, even if the specific test is too easy.
5. **The B-vs-F experimental design is clean.** Same data, same architecture, same optimizer, only heads differ. If the experiment is ever run, it will give a clean answer.

---

## Revised Recommendations (in priority order)

1. **Run the V-JEPA 2 linear probe NOW.** Download `vjepa2-vitg-fpc64-256`, run it on Kubric MOVi-E (or even CLEVRER), train a linear head on its spatiotemporal tokens to predict per-object ω. This is 1 GPU-day and answers the most important question: does V-JEPA 2 already encode the physics you want? If yes, the project simplifies to "physics-probing V-JEPA 2" (small paper, quick). If no, the full architecture is justified.

2. **Verify memory feasibility.** Run a single forward pass of V-JEPA 2 (frozen) + your proposed learned branch + one head on a 4-frame 320×320 clip on your actual hardware. Does it fit? What batch size? This takes 30 minutes and either confirms or kills the architecture.

3. **Drop contact force.** Replace it with a binary "is this body in contact? yes/no" classification head. This is learnable (it's basically collision detection from video), provides a meaningful training signal, and doesn't require predicting unmeasurable material properties.

4. **Pick one paper, not two.** Either:
   - **Empirical paper:** Simplify the architecture, run B-vs-F on Kubric (free data, perfect labels, no domain gap), report the number. This is achievable in weeks.
   - **Theory paper:** Pick Thm 2 (transfer bound), sketch the proof. If it goes through, write around it. If not, the theory paper is dead.

   Doing both simultaneously means neither gets done.

5. **Address the bootstrapping problem.** Decide: do dynamics heads use ground-truth seg masks (available in sim), predicted seg masks (realistic but circular), or external SAM2 masks (removes dependency but adds a frozen model to the already-heavy pipeline)? This isn't a minor detail — it changes what the dynamics heads actually train.

---

## Bottom Line (Revised)

The plan is now physically coherent — the former fatal flaw (single-frame input) is fixed, and the per-object formulation is correct. Credit where due: the student listened and made structural improvements.

However, the project's **core risk** has shifted, not disappeared. It was: "this is architecturally impossible." Now it is: **"this is architecturally possible but almost certainly redundant with V-JEPA 2's existing representation, and the sim-only domain gap has no tested solution."**

The single most informative experiment — a V-JEPA 2 linear probe on ω — has not been run. Until it is, everything else is planning in the dark. If V-JEPA 2 already linearly encodes ω (plausible, given it does robot control), the entire multi-head architecture is solving a problem that doesn't exist. If it doesn't (plausible, given it fails IntPhys 2), the project has a shot — but the domain gap remains the unaddressed load-bearing assumption.

Run the probe. Report a number. Then decide what to build.

---
---

# Alternative Research Directions (More Promising)

**Context:** The student asks for directions with genuine novelty, given: thesis-scale contribution needed, engineering/empirical preference, compute-constrained to 1-2 GPUs (4090-class).

---

## Direction A: "What Physics Do Video Foundation Models Actually Encode?"

### The idea
Nobody has systematically measured which physical quantities are linearly (or nonlinearly) decodable from modern video foundation model representations. V-JEPA 2 does zero-shot robot control but fails IntPhys 2 at near-chance. What does it actually know?

### What you build
1. **A parametric physics probe benchmark** from simulation (Kubric/Isaac Sim/CLEVRER):
   - Generate clips with ground-truth per-object (CoM, ω, v, mass-ratio, contact-state, stability)
   - Vary physical parameters (mass, friction, restitution) to create "minimal pairs" — two clips identical except for one parameter
2. **A probing toolkit** that evaluates any frozen video encoder:
   - Linear probes per quantity (train on sim, test on sim; train on sim, test on real)
   - Nonlinear probes (1-layer MLP) to distinguish "linearly inaccessible" from "not encoded at all"
   - Sensitivity probes: can the model distinguish the minimal-pair videos?
3. **Systematic evaluation** of V-JEPA 2, DINOv2 (with temporal stacking), VideoMAE-v2, InternVideo2, SigLIP-video, maybe Cosmos-Predict latents

### Why it's novel
- No such systematic comparison exists. The closest is IntPhys 2's binary VoE metric, which doesn't tell you *what* is missing.
- Minimal-pair parametric evaluation (varying one physical parameter) is a genuinely new evaluation paradigm for video models — finer-grained than "possible/impossible."
- The probing gap (what's linearly accessible vs. what requires nonlinear decoding vs. what's absent) has direct implications for whether adapters can help or full retraining is needed.

### Why it works for you
- **Compute:** Probing frozen models is cheap. The expensive part (training the FMs) is already done by Meta/Google/NVIDIA. You just run inference + train tiny probes.
- **Achievable timeline:** Benchmark generation (1-2 weeks), probing infrastructure (1 week), evaluation runs (1-2 weeks), analysis + writing (2-3 weeks). Total: ~2 months for a full paper.
- **Thesis narrative:** "I measured the gap → I identified what's missing → I fixed it" (the fix is your second paper).
- **Engineering artifact:** The benchmark + probing toolkit is reusable by the community.

### Risk assessment
- **Risk of being scooped:** Moderate. The IntPhys 2 result is embarrassing enough that someone else might do this. Move fast.
- **Risk of uninteresting result:** Low. Either "V-JEPA 2 secretly encodes physics but IntPhys 2's evaluation method is broken" or "V-JEPA 2 genuinely lacks physics" — both are publishable and actionable.
- **Risk of being too simple:** The probing study alone might be seen as "just evaluation." The parametric benchmark (minimal pairs) is what elevates it to a contribution.

### How it connects to the original brainstorm
If probing reveals that physics quantities are NOT encoded → you have motivation and a baseline for the next paper (training adapters or full architectures with physics supervision). If they ARE encoded → the original brainstorm's premise is wrong, but you've discovered something useful (the IntPhys 2 failure is in reasoning, not representation).

---

## Direction B: "Physics-Supervised Adapters for Video Foundation Models"

### The idea
Don't train a new encoder. Don't bridge sim→real. Instead: take V-JEPA 2 frozen, train a **lightweight adapter** (LoRA or a small temporal transformer, ≤10M params) supervised by physics quantities from simulation. Test whether the adapted representation improves downstream tasks (policy transfer, physics reasoning, IntPhys-style VoE).

### What you build
1. **Physics-supervised adapter module:** LoRA layers or a small (4-layer) temporal transformer inserted between V-JEPA 2's frozen layers and a set of physics heads (ω, v, CoM, contact-state).
2. **Training pipeline on synthetic data only** (Kubric MOVi-E/F + optionally Isaac Sim). No real data needed for adapter training — you're betting that the adapter generalizes because V-JEPA 2's base representation already handles appearance.
3. **Downstream evaluation:**
   - Linear probe quality on real data (does the adapter improve physics decodability on real videos?)
   - IntPhys-style VoE (does the adapted model assign higher likelihood to plausible videos?)
   - Policy transfer: train a small policy on adapted embeddings in sim, deploy on real (LIBERO or a simple task)

### Why it's novel
- V-JEPA 2-AC trains an action-conditioned predictor on top. Nobody has trained a **physics-conditioned adapter** that modifies the base representation itself.
- The "train on sim-only, test on real" paradigm for adapters (as opposed to full fine-tuning) is unexplored. The bet: a 10M-param adapter can't overfit to sim appearance the way a 300M-param encoder can, so domain gap is mitigated by capacity constraint.
- If it works, it's a recipe: take any video FM → add physics adapter → get physics-grounded representation. Cheap, modular, stackable.

### Why it works for you
- **Compute:** Adapters are tiny (10M params). Training on Kubric (free data) with a frozen 1B-param encoder in inference mode (no gradients through V-JEPA 2) is feasible on a 4090 at small batch sizes.
- **Builds on your existing knowledge:** You already understand V-JEPA 2's architecture, the physics head design, and the synthetic data pipeline (Kubric/Isaac).
- **Clear thesis contribution:** "I showed that sim-only physics supervision through lightweight adapters improves real-world physics understanding in video FMs."
- **Incremental:** Start with LoRA on Kubric (1-2 GPU-days), measure, iterate.

### Risk assessment
- **Risk of null result:** Moderate-to-high. If V-JEPA 2 already encodes the physics (Direction A would tell you), the adapter adds nothing. If the adapter overfits to sim appearance despite capacity constraint, real-domain probes degrade.
- **Risk of being "just LoRA + loss":** The novelty is thin if the adapter doesn't demonstrably improve downstream tasks. You need a clear win on at least one evaluation.
- **Mitigation:** Run Direction A first (the probing study). If probes show physics is NOT linearly accessible → adapter has clear motivation. If physics IS accessible → adapter direction is dead before you waste time.

### How it connects to the original brainstorm
This is a dramatically simplified version of the original idea: same hypothesis (physics supervision helps encoders), same synthetic labels, but without the new encoder architecture, without the domain-marginalization regularizer, without the multi-head equivalence theory. If the adapter works, the follow-up paper is: "now let's scale it up" (the original brainstorm). If it doesn't, the original brainstorm is dead too.

---

## Direction C: "Small Equivariant Model Beats V-JEPA 2 on Physics" (High-Risk, High-Reward)

### The idea
The IntPhys 2 result + the opinion paper together say: scaling observation-only video models doesn't help for physics. The hypothesis: the bottleneck is **inductive bias, not scale**. Build a small (50-100M param) model with the right equivariance structure (SE(3) camera group + gravity vector) trained on synthetic physics scenarios, and show it outperforms V-JEPA 2 (1B params, 1M hours of video) on IntPhys-style tasks.

### What you build
1. **An equivariant video encoder** that respects:
   - SE(3) camera pose changes (representation transforms predictably under camera motion)
   - Gravity direction as a privileged axis (breaks SO(3) to SO(2) around vertical)
   - Per-object factorization (slot-attention or similar)
2. **Training on synthetic physics scenarios** with physics-quantity supervision (the same heads from the brainstorm, but now in a model designed to use them).
3. **Evaluation on IntPhys 2** (or a proxy, since IntPhys 2 may not be publicly released as a benchmark with standard eval code).

### Why it's novel
- Nobody has tried equivariant architectures on intuitive physics benchmarks. The equivariant SSL literature (Garrido et al.) works on images; extending to video + physics is new.
- The positioning is bold: "Meta spent 1B params and 1M hours to get 56% on IntPhys 2. We get X% with 50M params and 10k synthetic clips. The difference is structure, not scale."
- If it works, it's a genuine paradigm shift for the intuitive physics community.

### Why it's risky
- **Equivariant architectures are hard to implement correctly.** The SE(3)-equivariant transformer literature (SE(3)-Transformers, EGNN, equiformer) is mostly for 3D point clouds, not video. Adapting to video patches is non-trivial.
- **IntPhys 2 might be hard for reasons unrelated to equivariance** (e.g., it might require multi-step counterfactual reasoning, not just representation quality).
- **The student hasn't read the equivariant SSL literature** (it's in the "reading still owed" list). Starting from zero on a hard architectural class is risky for a thesis.
- **Evaluation infrastructure for IntPhys 2 may not be straightforward** to set up.

### Why it might work for you (if you have the appetite)
- **Compute:** Small equivariant models + synthetic data = feasible on 1-2 GPUs.
- **Thesis-worthy:** If it works, it's a strong thesis ("structure > scale for physical reasoning"). If it partially works, it's still publishable ("equivariant inductive biases help physics probes but don't solve reasoning").
- **Long-horizon:** This could be a 2-3 paper thesis arc (equivariant architecture → physics supervision → downstream robotics).

---

## Direction D: "Sim-Only Physics World Model for Policy Evaluation" (Most Practical)

### The idea
Forget trying to bridge sim→real in the encoder. Instead: build a **physics-grounded world model that operates entirely in sim** (Isaac Sim), using physics heads as internal consistency checks. The downstream application: **policy evaluation** — given a policy checkpoint, roll out in your world model and predict whether it will succeed in the real world, without deploying on a real robot.

RoboScape showed this works (r=0.953 correlation between generated-video success rate and real success rate). The gap: RoboScape's physics is implicit (keypoint consistency). You make it explicit (predict CoM, ω, contact → check against simulator ground truth → physics-consistent rollouts are more predictive of real success).

### What you build
1. **A small autoregressive video model** (150-300M params, a la RoboScape) trained on Isaac Sim data with physics auxiliary heads.
2. **Physics-consistency scoring:** After rollout, measure whether the predicted physics quantities (ω, v, contact) are physically consistent (conservation of momentum, Newton's third law at contacts). Score each rollout by physics consistency.
3. **Policy evaluation pipeline:** Roll out candidate policies in the world model → score rollouts by physics consistency → predict real-world success rate.
4. **Validation on a simple real task** (ball-pushing, pick-and-place) where you can also measure ground-truth success rate.

### Why it's novel
- RoboScape shows the *concept* (world model as policy evaluator) works. Your contribution: **physics-consistency scoring** as a new evaluation metric for generated rollouts, and showing it improves the success-rate correlation.
- Nobody has used explicit physics quantities as internal consistency checks for video world models.
- The "sim-only" framing removes the domain gap problem entirely for training. The domain gap only appears at evaluation time (does physics-consistent generation in sim correlate with real success?).

### Why it works for you
- **No domain gap during training.** Everything trains on sim. You sidestep the hardest unsolved problem.
- **Clear engineering pipeline.** Data generation → model training → physics scoring → evaluation.
- **Compute:** RoboScape trained 544M params on 32 A800s, but their smallest model (34M) already showed signal. A 100-150M model on 10k clips is feasible on 1-2 GPUs.
- **Thesis narrative:** "Physics-consistent world models are better policy evaluators than appearance-only world models."
- **Builds naturally on your existing knowledge** (Isaac Sim, RoboScape architecture, physics heads).

### Risk assessment
- **Risk of null result:** Moderate. Physics consistency might not improve success-rate prediction over pure appearance quality (LPIPS, FVD). If RoboScape's implicit physics (keypoint consistency) already saturates the correlation, explicit physics adds nothing.
- **Risk of being "just RoboScape + physics heads":** The novelty needs to come from the consistency-scoring mechanism, not just adding heads. Emphasize the post-hoc verification aspect.
- **Need real robot access for validation:** Even one simple real task (or borrowing someone else's real eval data).

---

## My Recommendation

**Start with Direction A (probing study), use it to decide between B and D.**

Here's the sequencing:

1. **Week 1-2: Run the V-JEPA 2 linear probe.** Download model, run on Kubric MOVi-E, train linear heads for ω, v, CoM. Report numbers. This is the fork:
   - If V-JEPA 2 linear probe fails (R² < 0.3 on ω) → physics is genuinely NOT encoded → Direction B (adapters) has a shot, Direction A becomes the first paper.
   - If V-JEPA 2 linear probe succeeds (R² > 0.5 on ω) → physics IS encoded, just not used by IntPhys-style evaluations → Direction A focuses on "why does encoding not imply reasoning?", Direction D (world model) becomes the right build project.

2. **Month 1-2: Direction A paper.** Parametric benchmark + systematic probing. This is publishable as a standalone paper and provides the foundation for whatever comes next.

3. **Month 3-6: Build project (B or D depending on probe results).** This is the thesis meat.

This sequencing is:
- Cheap to start (probe is 1 GPU-day)
- Information-maximizing (the probe result determines which direction has a shot)
- Risk-managed (you don't commit months to a direction that a 1-day experiment could falsify)
- Thesis-structured (measurement paper → build paper → possible third paper extending)

---

## What Makes These Better Than the Original Brainstorm

| Property | Original brainstorm | Directions A/B/D |
|---|---|---|
| Domain gap | Unresolved load-bearing assumption | A: not relevant (probing). B: mitigated by capacity constraint. D: avoided entirely (sim-only training). |
| Closest prior result | PI-JEPA (null) | A: no prior exists (genuinely new). B: V-JEPA 2-AC (positive). D: RoboScape (positive). |
| Compute requirement | 1B-param V-JEPA 2 + learned branch + 7 heads = likely infeasible on 4090 | A: inference-only. B: 10M-param adapter. D: 100-150M model on sim data. |
| Novelty risk | "Physics heads on an encoder" has been tried (PI-JEPA, negative) | A: new benchmark paradigm. B: new adapter paradigm. D: new evaluation paradigm. |
| Publishable under the null | No — F≈B means no contribution | A: always publishable. B: "adapters don't help" is informative. D: "physics consistency doesn't improve correlation" is informative. |
| First experiment cost | "Weeks of GPU time" before any signal | A: 1 GPU-day. B: 1-2 GPU-days. D: few GPU-days for first results. |
