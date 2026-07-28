# Physics-Informed Latent Dynamics from Rendered Video — Pragmatic Ladder

**Status:** brainstorm, pre-experiment. For review by ML domain experts before any code is written.

**Brutal-honesty mandate:** this document is meant to be picked apart. Sections marked **[Honest take]** are deliberately critical and should be argued with, not nodded at.

**Related:** the geometric instantiation of Path B (§8) — allocentric, dimensionless world-lines — is now its own design doc, [world-lines.md](world-lines.md) (encoding layer implemented in `src/worldlines/`).

---

## 1. Origin of the idea

A V-JEPA-style framing was proposed: throw a ball with unknown impulse `J` at a ball of unknown mass `m`, observe the trajectory in rendered video, and have a model infer the (force, mass) pair *in latent space* — i.e., without direct scalar supervision. The claim was that visual cues (ball radius → plausible material density on Earth) bound an otherwise infinite manifold of `(F, m)` explanations.

Two structural problems were flagged during brainstorm:

1. **Identifiability:** kinematics gives `a = F/m`. Even with gravity as a scale anchor (the `½g·t²` term fixes time/length units), the impulse contribution reduces to `J/m`. So the trajectory contributes a **1-D constraint** on `(J, m)`; everything else is prior knowledge encoded as visual cues + an Earth-materials prior.
2. **"No direct supervision" is a fiction** if you train an encoder of `(F, m)` to define the JEPA target. The supervision is smuggled in via that encoder.

Both problems are real and should not be papered over.

## 2. The proposed first rung

Smallest jump from the current sim setup (single ball, gravity-only, impulse) to a setup that actually does physics-informed representation learning from rendered video:

- **Sim extension:** add quadratic drag, `m·v̇ = m·g − c·v·|v|`. Renderer unchanged.
- **Data:** rendered RGB clip + ground-truth trajectory + ground-truth `(m, J, c, radius, material-ish color)`.
- **Model:** an encoder `φ: video → z_t` plus a latent-space dynamics module that satisfies an ODE residual between consecutive `z_t`. Style is Latent Neural ODE / JEPA hybrid: predict `z_{t+Δ}` from `z_t` via a learned vector field, with a physics-shaped inductive bias on the vector field.
- **Targets:** recoverable physical parameter ratios — `J/m`, `c/m`, and `g_eff` if not fixed.

The drag term is deliberately chosen because it **breaks** the pure projectile degeneracy in a useful way: trajectories with different `(m, c)` but identical `J/m` are no longer congruent, so the model is forced to use more of the trajectory than just the initial velocity.

## 3. What is already done in the literature

**[Honest take]** A non-trivial fraction of this rung has been solved, and reviewers will know it. Specifically:

- **Neural ODEs / Latent ODEs** (Chen 2018; Rubanova 2019) — learn ODE vector fields in latent space from observations. Projectile, pendulum, double-pendulum demonstrations exist.
- **Hamiltonian / Lagrangian Neural Networks** (Greydanus 2019; Cranmer 2020) — bake conservation laws into the architecture. Drag breaks Hamiltonian structure but Lagrangian-with-dissipation variants exist.
- **PDE-Net / SINDy / equation discovery** (Long 2018; Brunton 2016) — learn the symbolic form of dynamics from observations.
- **Physics-informed world models** (PlaNet, DreamerV3, learned simulators à la Sanchez-Gonzalez) — extract dynamics from pixels at much larger scale than the proposed rung.
- **V-JEPA / V-JEPA-2** — masked-prediction in latent space on video; physics evaluation is part of the benchmark suite (IntPhys, GRASP).

A clean "drag from video → Neural ODE → recover `c/m`" experiment is a **week-scale exercise**, not a research contribution. If the ladder stops at rung 1, the only output is a teaching demo.

## 4. What could be novel (and what probably isn't)

**Probably not novel:**

- Recovering `(J/m, c/m)` from rendered trajectories via a Neural ODE.
- Showing that visual cues + trajectory together identify mass better than trajectory alone.
- Demonstrating that latent-ODE forecasting outperforms a black-box RNN on this toy.

**Potentially novel, conditional on careful framing:**

- **Equivalence-class representation learning under physical ambiguity.** Train the latent so that throws with the same `(J/m, c/m)` but different absolute `(J, m, c)` map to the same `z` — and throws on different equivalence classes separate. The interesting claim is that JEPA-style masked-trajectory prediction *induces* this equivalence class without it being labeled. That is testable.
- **Compositional / OOD generalization** to drag regimes, masses, or material priors not seen in training — using a latent that "knows" the ODE structure vs. one that doesn't.
- **A clean ablation of how much each information source contributes** (trajectory only / visual only / both) to closing the identifiability gap. The community has strong intuitions here but few rigorous toy-scale measurements.

**[Honest take]** Item 1 above is the only thing I'd defend as worth a workshop paper. Items 2 and 3 are good experiments but framing them as a contribution requires comparison against an existing baseline that already does the same thing implicitly — and many world-model papers do.

## 5. Success criteria

The experiment is worth running only if it can answer **at least one** of the following with a clean yes/no:

1. Does JEPA-style masked latent prediction induce an equivalence-class structure over `(J/m, c/m)` *without* labels, measurable via linear probe or clustering?
2. Does the physics-shaped latent transfer to unseen `(m, c)` regimes better than a black-box latent of the same capacity?
3. Does removing visual cues (mask the ball, leave only a tracked point) degrade mass estimation in the predicted way (i.e., it collapses to the `J/m` manifold)?

If none of these can be cleanly answered, the rung is not worth climbing.

## 6. Kill criteria (when to stop)

- A two-day SINDy / Neural-ODE baseline on tracked `(x, y)` state already recovers `(J/m, c/m)` to within sim noise. → The video pipeline adds nothing; pivot to a scene where tracking is hard (clutter, occlusion, multi-body).
- The JEPA latent collapses to encoding raw trajectory regardless of physics structure (i.e., a linear probe on `z` predicts pixel coordinates better than it predicts `J/m`). → The latent is not learning physics; either the inductive bias is wrong or the task is too easy.
- The "equivalence-class" claim (§4 item 1) is just a rephrasing of "the model learned `a = F/m`." → No contribution.

## 7. Risks and known weaknesses

- **The toy is too small to require a JEPA.** A 4-parameter ODE with 60-frame clips can be fit by an RNN with a regression head. The JEPA framing only earns its keep if the no-label / equivalence-class story is the contribution, not the regression accuracy.
- **Renderer–reality gap is zero.** Everything trains and evaluates on the same renderer. Generalization claims will be weak. At minimum, a domain-randomized eval set is needed before any claim about "learning physics."
- **The V-JEPA inspiration may be decorative.** Stripping the framing, this is "Neural ODE on rendered video." If the JEPA loss is replaced by a regression loss on `(J/m, c/m)` and results are similar, the JEPA wasn't doing work.
- **Drag is non-conservative**, which rules out Hamiltonian NN baselines and forces a Lagrangian-with-dissipation or vanilla Neural ODE. Fine, but worth knowing before architectural choices.

## 8. Bigger-picture honesty

**[Honest take]** The interesting research question behind all of this — *can passive video observation teach a model the structure of physical laws, in a way that transfers?* — is live, important, and being attacked at scale by V-JEPA-2, Genie, Sora-style world models. A single-ball toy is unlikely to move that needle. Its value is **diagnostic**, not headline-grabbing: a controlled setup where you can measure exactly what your method is and isn't learning, before scaling to multi-body scenes where measurement gets ambiguous.

Two paths forward, with honest expected payoff:

- **Path A (low risk, low novelty):** climb the ladder as stated, end up with a tidy toy demonstration. Useful as internal calibration; unlikely to be publishable on its own.
- **Path B (higher risk, higher novelty):** skip rung 1 and design directly for the equivalence-class claim (§4 item 1) on a multi-scenario sim where the prior literature has not been measured. Higher chance of dead end, higher chance of a real contribution.

Recommendation for the expert discussion: **decide between A and B first**, then come back to the rung-1 details. The rung-1 design is fine; what's missing is a believable claim that climbing it teaches us something new.

## 9. Open questions for the experts

1. Does the equivalence-class-induction claim (§4 item 1) already exist in the world-model literature in a form I missed?
2. Is there a published toy that already isolates "how much does visual prior contribute to mass identifiability"? If yes, this whole document is redundant.
3. Is the right comparison baseline a Neural ODE on tracked state, a Latent ODE on video, or a vanilla V-JEPA? (Probably all three, but ranked.)
4. Should drag be the dynamics extension, or is something like a bouncing ball (contact-rich, non-smooth) a better stress test? Trade-off: drag is clean and identifiable; contact is closer to real-world hard cases but breaks ODE assumptions.
