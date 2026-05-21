# Cosmos Policy — Fine-Tuning Video Models for Visuomotor Control and Planning

Kim, Gao, Lin, Lin, Ge, Lam, Liang, Song, Liu, Finn, Gu. NVIDIA + Stanford. arXiv 2601.16163v1, Jan 2026. [PDF](../papers/cosmos_policy.pdf).

## TL;DR
A **single-stage post-training recipe** that turns the pretrained [Cosmos-Predict2-2B](cosmos-predict2.5.md) video model into a robot policy that jointly predicts **(action chunk, future state, future state value)** — no architectural modifications, all modalities trained through the same video diffusion objective.

Mechanism: **latent frame injection**. Tokenize multi-view RGB into latent frames via the WAN2.1 VAE; then *overwrite* additional placeholder latent frames with normalized + duplicated copies of robot proprioception, action chunks, and value scalars. The model denoises the whole sequence; the diffusion loss applies uniformly. Modalities are extracted at inference by averaging the duplicated cells of the relevant latent slots.

Single training stage on demonstrations + (later) policy rollouts. Same model serves as **policy** *p(a, s′, V(s′) \| s)*, **world model** *p(s′, V(s′) \| s, a)*, and **value function** *p(V(s′) \| s, a, s′)* — distinguished only by which slots of the latent sequence are masked at training/inference.

**Results.** SOTA on LIBERO (98.5 % avg) and RoboCasa (67.1 % avg) with 50 demos/task vs ≥300 for VLA baselines; beats π₀, π_{0.5}, OpenVLA-OFT, GR00T-N1.5, Diffusion Policy, Dita, UniVLA, CogVLA, UVM, Video Policy, FLARE. On real-world ALOHA bimanual tasks (put-X-on-plate, fold-shirt, candies-in-bowl, candy-in-ziploc-bag) achieves the highest average score (62.8) over π₀, π_{0.5}, OpenVLA-OFT+, Diffusion Policy. Model-based planning via best-of-N with the value head adds +12.5 absolute points on the two hardest ALOHA tasks.

## Relevance to the brainstorm — limited and indirect
This paper is **largely tangential** to the physics-informed-embedding idea. The mapping:

| Cosmos Policy                                    | Brainstorm                                          |
|--------------------------------------------------|-----------------------------------------------------|
| Output: action chunks + future states + values   | Output: a reusable visual *encoder*                 |
| Pretrained video model = initialization          | Pretrained video model = candidate frozen branch    |
| Supervision: demonstrations + on-policy rollouts | Supervision: simulator-rendered physical quantities |
| Multi-modality via latent-frame injection        | Multi-task via auxiliary prediction heads           |
| Physics priors: *implicit*, inherited from video pretraining | Physics priors: *explicit*, supervised via heads |
| Architecture: zero changes vs. base model        | Architecture: dual-branch + cross-attn (specific structural choice) |

They share *thematic* DNA — "use multiple modalities jointly through one model" — but the brainstorm is about *injecting structure into the encoder*, while Cosmos Policy is about *exploiting an already-good video model as a function approximator for everything*.

## Where it does touch
1. **Auxiliary supervision ablation** (Sec 5.2): training Cosmos Policy *without* the world-model + value heads (so it's just `p(a|s)`) drops average success rate by 1.5 absolute %; training from random weights drops it by 3.9 %. So joint multi-target supervision contributes mildly; pretrained video priors contribute substantially. This is **mild empirical support** for the brainstorm's multi-head hypothesis, *but* the auxiliary targets here are not physical quantities — they're future state and value (a kind of TD target).
2. **Latent-frame injection** is a clever alternative to the brainstorm's auxiliary-head structure: encode every new modality as latent tokens *in the input space of the base model*, train with the base model's native loss. Trade-off — no new heads to design (simple), but each modality consumes spatial latent volume (expensive at scale). Worth knowing as a *non-head* implementation pathway if the brainstorm ever wants to attach simulator quantities to a frozen video generator without modifying it.
3. **Dual deployment** (Sec 4.3): one checkpoint trained on demos → serves as the policy; refined on rollouts → serves as world model + value function. Templates a recipe for **separating "what the encoder learns" from "what it's used for"** that may be useful if the brainstorm encoder is later refined per downstream task.
4. **Failure mode of pretrained video priors alone**: Fig. 6 shows the base Cosmos Policy world model mis-predicts the ziploc-bag dynamics (loses grip) when trained only on successful demos. *On-policy rollouts* (including failures) are required to fix the world model. Implication for the brainstorm: **demonstrations-only training is not sufficient to give a world model accurate physics**. Sim provides exactly the kind of densely-sampled, label-rich failure data this paper had to collect via rollouts.
5. **Noise distribution tuning** (Appendix A.2.1): the default Cosmos-Predict2 log-normal `ln(σ) ~ N(1.39, 1.2²)` is wrong for action generation — too little mass at high σ → poor initial denoising → cascading errors. Concrete heads-up that the brainstorm's heads (especially scalar physical quantities with wide dynamic range) likely need their own σ-schedule if a diffusion target is used. Same concern raised in [DreamerV3](dreamer-v3.md)'s symlog tricks.

## What this paper does *not* contribute to the brainstorm
- No physical-quantity supervision (CoM, contact, ω, normals, depth — none).
- No sim/real domain split, no Sim2Real story (training is real demos + simulator demos, but as the same kind of supervision).
- No equivariance, no conservation laws, no counterfactual supervision.
- No theory; pure systems / empirical paper.

## Where to file it
Tier: **tangential** — alongside [omnicontrol.md](omnicontrol.md) and [rt-h.md](rt-h.md), but with one extra reason to remember it: it pins down what "use a pretrained video model directly as a policy" achieves *without* the brainstorm's physics-supervision route. That is the right-hand-side of the comparison "is explicit physics supervision worth it, or does video-pretraining + joint training suffice?" — a question the planned theoretical paper ([plan §6](../plans/theoretical-paper-plan.md)) should engage with.

## Released artifacts
- Project page, code, model checkpoints: https://research.nvidia.com/labs/dir/cosmos-policy/

## Not deeply read
- Detailed planning algorithm (best-of-N with majority-mean voting, Sec 4.3).
- Q(s,a) vs V(s′) planning comparison (Sec 5.3).
- Per-task LIBERO / RoboCasa breakdowns and OOD evaluation protocol.
- Full set of failure-mode visualizations (Figs 5, 6).
