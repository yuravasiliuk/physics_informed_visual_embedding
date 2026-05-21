# Why and How Auxiliary Tasks Improve JEPA Representations

Yu, Chen, Liu, Horiuchi, Braverman, Xu, Haramati, Balestriero. arXiv 2509.12249v2, Oct 2025. [PDF](../papers/why_and_how_auxiliary_tasks_improve_jepa.pdf).

## TL;DR
A theory paper that formalizes what an auxiliary regression head does to a JEPA encoder. Main result — **No Unhealthy Representation Collapse theorem**: in a deterministic MDP, if both the latent transition-consistency loss and the auxiliary regression loss reach zero, then any two observations that differ in either their auxiliary value *or* in their transition dynamics under some action sequence must be mapped to **distinct** latent codes.

Practical implication: **the auxiliary target is not a heuristic; it is what defines the equivalence relation the encoder is forced to preserve.**

## Why this matters for the brainstorm
The brainstorm proposes a long menu of auxiliary heads (normals, depth, segmentation, center-of-mass, contact forces, angular velocities, distance-to-ground). This paper says: each head is a knob that *changes the geometry of the embedding space*. So:

- A head like **center-of-mass** forces two visually-similar but physically-different scenes (e.g. solid vs. hollow cup) to live at different latent points.
- A head like **contact forces** anchors the encoder to distinguish scenes that *look* the same but have different force flow (in-contact vs. about-to-collide).
- Heads that are too easy (predictable from low-level RGB) add little distinguishing power; heads that are too hard never converge.

**The choice of which sim-only signal to supervise on is the most important design decision**, not the architecture.

## Setup (P-JEPA)
A simplified TD-MPC2-style architecture:
- Encoder `E_φ`: observation → latent `z`.
- Latent transition model `T_ψ`: predicts `z_{t+1}` from `(z_t, a_t)`.
- Auxiliary regressor `P_θ`: predicts an auxiliary function `p(o)` from `z`.
- **No stop-gradient, no EMA target** — all three trained jointly with `L = L_dyn + c_p · L_p`.

The auxiliary function `p` can be the reward, a Q-function, or — in this paper's experiments — even a random projection.

## Main theorem (informal)
Define apartness `B*` as the largest set of `(o_i, o_j)` pairs that are distinguishable now or after some sequence of actions (this is essentially `O × O` minus the largest bisimulation under `p`). Then a well-trained P-JEPA satisfies:

$$o_i \not\equiv_{B^*} o_j \implies E_\phi(o_i) \neq E_\phi(o_j).$$

A finite-data version (Theorem 3 in Appendix B.1) tightens this to pairs that the dataset already certifies as non-bisimilar.

## Counting-environment experiment
- 64×64 RGB scenes with k ∈ {0..8} objects (shape and color sampled once per episode). Actions = inc/dec the count. Auxiliary task = predict reward = 1[count == 4].
- Theory predicts the latent should have **at least 9 distinct clusters**, one per count, even though only two reward values exist. Empirically observed (top row of Fig. 2).
- Decoder trained *without* backprop into the encoder cannot recover shape/color/position → encoder discards them as irrelevant.
- Latent-loss alone collapses to one point. Reward-loss alone gives only coarse separation. Combined — clean 9 clusters.
- Replacing reward with a **random 256-D linear projection** also prevents collapse and partially preserves position/color/shape, confirming the mechanism is about "what the head asks to distinguish," not about reward semantics specifically.

## Take-aways for the brainstorm design
1. **Stack heads with mutually informative targets.** Depth + normals + segmentation give 2.5D structure; CoM + contact forces give *physical* structure. They are complementary equivalence relations.
2. **Don't fear redundancy.** Theorem says any auxiliary that distinguishes more pairs is harmless — the encoder won't collapse the new distinctions.
3. **Beware the inverse**: if an auxiliary signal is *too low-dimensional* (e.g. a single binary "is in contact"), the equivalence classes it induces are huge — most of the discriminative work must come from other losses (predictive consistency, other heads).
4. **The same theory backs sim-only heads working at training time even though they're absent at inference.** As long as the auxiliary loss reaches zero on the sim distribution, the encoder gains the distinctions; real-data evaluation only needs the encoder forward pass.

## Caveats
- Theory assumes deterministic MDP; predictive sim is approximately deterministic, real video is not.
- "Well-trained" is the load-bearing assumption — failure modes in practice come from imperfect optimization, not the theorem.

Code: https://github.com/jasonyu48/concept_discovery
