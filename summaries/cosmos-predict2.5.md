# World Simulation with Video Foundation Models for Physical AI (Cosmos-Predict2.5 + Cosmos-Transfer2.5)

NVIDIA, arXiv 2511.00062v1, Oct 2025. [PDF](../papers/2511.00062v1.pdf).

## TL;DR
Successor to [Cosmos-Predict1](cosmos-predict1.md). Two model families released:

- **Cosmos-Predict2.5** (2B, 14B) — single unified flow-matching DiT for Text2World, Image2World, Video2World. Uses **Cosmos-Reason1** (a Physical-AI VLM) as text encoder instead of T5. Trained on **200 M curated clips** (up from ~100 M); 35 M hours raw → 200 M trainable clips (only 4 % survive new strict filtering).
- **Cosmos-Transfer2.5** (2B) — ControlNet-style framework for **Sim2Real and Real2Real video translation**, conditioned on edge / blur / segmentation / depth / world-scenario maps. 3.5× smaller than Cosmos-Transfer1 but higher fidelity, supports long-horizon and closed-loop simulation.

## Why it's in the reading pile (and directly compares with the brainstorm's approach)
Cosmos-Transfer2.5 is the **inverse direction** of the brainstorm's idea:

| Cosmos-Transfer2.5                       | Brainstorm's "physics-informed embedding"   |
|------------------------------------------|---------------------------------------------|
| Sim → realistic RGB video (translation)  | Real or sim → shared physics-aware embedding |
| Solves Sim2Real by *pixel translation*   | Solves Sim2Real by *latent alignment*        |
| Sim2Real used to grow training data for downstream policies | Sim2Real used to inject supervision into a shared encoder |

Both approaches are funded by the same observation: sim provides cheap structured signal that real video doesn't. They are not mutually exclusive — Cosmos-Transfer2.5 could *augment* the brainstorm's training data (e.g. apply the translator to the rendered sim views to reduce the appearance gap before training the encoder).

## Key architectural changes vs. Predict1
- **Flow matching** (Lipman et al., 2022) instead of EDM. Velocity-prediction target, shifted logit-normal time sampling biased toward higher noise levels.
- **Unified Text/Image/Video2World** via a frame-replacement strategy: conditioning frames are substituted into the noisy sequence (with binary mask tokens) and loss is applied only on generated frames.
- **Cosmos-Reason1 VLM as text encoder** (multi-block embedding aggregation à la Wang et al. 2025) — richer than T5-XXL.
- **WAN2.1 VAE** as visual tokenizer: compression 4×8×8 across T/H/W, then 1×2×2 patchification.
- **No absolute positional embeddings**, only relative — allows generalization to higher resolution / longer sequences in post-training.
- **Multi-stage pretraining**: Text2Image @ 256p → joint Image2World/Video2World @ 256p → 480p → 720p. Then SFT on curated Physical-AI sets → RL with a novel algorithm → timestep distillation.

## Cosmos-Transfer2.5 details
A modular ControlNet over Cosmos-Predict2.5-2B:
- **Inputs**: edge / blur / segmentation / depth / world-scenario map (a metadata field encoding object types, colors, materials).
- **Outputs**: realistic RGB video, length-flexible, multi-view.
- **Downstream uses**:
  - **Robot policy learning** (Sec 6.2): augment AgiBot bimanual demos with style-transferred versions to grow data.
  - **Driving simulation** (Sec 6.3): convert simulator output to photorealistic multi-view driving video.
  - **Closed-loop simulation**: feed transferred video into policy → predict next state → re-render.

## Domain-specific data curation (Sec 2.2)
Five domain-specific subsets added to the general 200 M:
1. **Robotics** — AgiBot-Beta (194k), DROID, GR00T, Bridge, 1X, OpenX, RoboMIND.
2. **Autonomous Driving** — 3.1 M 20-s 7-camera surround clips on NVIDIA's internal platform.
3. **Smart Spaces** — 40k warehouse/factory clips.
4. **Human Dynamics** — YOLOX + RTMPose-filtered clips.
5. **Physics** — taxonomy-curated clips of physical phenomena (shattering glass, colliding balls, flowing water).

## Take-aways for the brainstorm
1. **Sim2Real-via-translation is a strong baseline** the brainstorm needs to position against. The argument is that latent alignment generalizes better than pixel translation when the encoder feeds a downstream task (control, prediction).
2. The brainstorm could **use Cosmos-Transfer2.5 in the pipeline**: render in Isaac Sim → translate to realistic RGB → use *both* (translated real-looking + raw sim with ground-truth physics) to train the encoder. This narrows the appearance gap while preserving the physics labels.
3. The brainstorm's frozen-encoder branch could plug in **Cosmos-Reason1** for semantics if a VLM is preferred over DINO/JEPA.
4. The domain-specific data curation in Sec 2.2 is a directly reusable taxonomy for collecting the brainstorm's training videos.
5. **Action-conditioned** Cosmos-Predict2.5 (`Cosmos-Predict2.5-2B/robot/action-cond`) is the closest open model to V-JEPA-2-AC — same role, different objective (pixel-space flow matching vs. latent prediction).

## Released artifacts
- Predict2.5: https://github.com/nvidia-cosmos/cosmos-predict2.5
- Transfer2.5: https://github.com/nvidia-cosmos/cosmos-transfer2.5
- Cosmos-Reason1 VLM (text encoder & reasoning model).
