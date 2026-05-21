# V-JEPA 2 — Self-Supervised Video Models for Understanding, Prediction, and Planning

Assran et al., FAIR Meta, June 2025. arXiv 2506.09985. [PDF](../papers/v-jepa2.pdf).

## TL;DR
Two-stage recipe:
1. **V-JEPA 2 (action-free)**: scale V-JEPA-style mask-denoising in *latent space* to ViT-g (1B params) on 22 M videos / 1 M hours / 1 M images.
2. **V-JEPA 2-AC (action-conditioned)**: freeze the encoder, train a 300 M autoregressive predictor on **62 hours of unlabeled Droid robot videos**, then use it for zero-shot model-predictive control on real Franka arms (~65–80 % success on grasp, pick-and-place).

## Why it's in the reading pile
Most likely choice for the **"frozen semantic encoder"** in the brainstorm. The paper also shows that an encoder trained *purely from observation* can be repurposed for action-conditioned planning with a small post-training step — relevant to "do we add the sim-only heads as auxiliaries to the encoder, or as a downstream predictor?"

## Pre-training details (V-JEPA 2)
- **Loss**: `||P_φ(Δ_y, E_θ(x)) − sg(E_ξ̄(y))||_1` with stop-grad on the EMA target. Mask is a multiblock video mask over 2×16×16 tubelets.
- **Encoder**: ViT-g/16, 3D-RoPE position embedding (factorized over time, H, W).
- **Four scaling ingredients** that each gave measurable gains on a 6-task probe average:
  1. Data 2 M → 22 M videos (+1.0 pt)
  2. Model 300 M → 1 B (+1.5 pt)
  3. Training 90 K → 252 K iters (+0.8 pt)
  4. Higher resolution and longer clips (256→384, 16→64 frames) (+0.7 pt)
- **Progressive resolution / duration**: train short low-res clips during warmup+constant phases, then bump resolution and clip length only during the cooldown phase — 8× GPU-time savings.
- **Curated YT-Temporal 1B**: cluster-based retrieval to match the target distribution (Kinetics/SSv2/COIN/EpicKitchens).

## V-JEPA 2-AC (action-conditioned)
- Inputs per timestep: video patch features `z_k = E(x_k)` (frozen), 7-D end-effector state `s_k`, 7-D action `a_k`.
- Predictor: 300 M transformer, **block-causal attention** so each patch at time `t` attends to action/state/patches at all `t' ≤ t`.
- Two losses: teacher-forcing L1 on next-step latent + 2-step rollout loss for stability.
- Inference: cross-entropy method over Gaussian-sampled action sequences, minimize L1 between rollout latent and goal-image latent.
- Achieves grasp/reach/pick-and-place on two Franka arms in unseen labs, **zero-shot** (no robot data from those labs).

## What V-JEPA 2 tells us about the brainstorm
- **Frozen encoder + small action-conditioned head** is a viable post-hoc path; if the proposed sim-only heads (CoM, contact forces) can be cast as functions of `(state, action)` they could be added the same way without retraining V-JEPA's backbone.
- The encoder evidently already captures enough motion structure for closed-loop control without ever seeing actions during pretraining — this slightly weakens the brainstorm's premise that *physics-aware* supervision is necessary just to get usable representations for control. (See [opinion-intuitive-physics.md](opinion-intuitive-physics.md) for the counter-argument: V-JEPA 2 still fails intuitive-physics benchmarks.)
- Pretraining loss is **L1 in feature space, not pixel space**, with EMA target — standard JEPA recipe, recommended starting point for the brainstorm's "semantic patches" branch.
- ViT-g/16 backbone is open-source (HF: `facebook/vjepa2-vith-fpc64-256` etc.) — usable off-the-shelf.

## Limitations to be aware of
- V-JEPA 2-AC is sensitive to camera position; the model implicitly infers the action coordinate axis from RGB. Bad axis → bad world model.
- Long-horizon planning suffers from latent-rollout error accumulation.
- No explicit physical grounding: as IntPhys 2 / the Opinion paper show, near-chance on intuitive physics despite massive scale.

## Concrete starting points for the brainstorm
- Use `vjepa2-vith-fpc64-256` (frozen) as the semantic branch.
- The geometry/conv branch can be initialized fresh or from V-JEPA's own backbone (the V-JEPA encoder is already conv-then-ViT).
- Match the cross-attention design of [SCOPE](scope.md), with V-JEPA tokens as K/V.
- For sim-only heads, mirror V-JEPA 2-AC's pattern: keep the encoder frozen, train heads as a lightweight transformer on top.

Code: https://github.com/facebookresearch/vjepa2
