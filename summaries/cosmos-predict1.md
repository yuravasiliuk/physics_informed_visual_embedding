# Cosmos World Foundation Model Platform for Physical AI (Cosmos-Predict1)

NVIDIA, arXiv 2501.03575v3, July 2025. [PDF](../papers/2501.03575v3.pdf).

## TL;DR
A platform of **video world foundation models** (WFMs) for Physical AI, released open-weight. A WFM is `W: (x_{0:t}, c_t) → x̂_{t+1}` — predicts the next video frame from past observations and a perturbation `c` (text, action, camera pose). Two model families:

- **Cosmos-Predict1-Diffusion** (7B / 14B Text2World, Video2World): denoising-diffusion transformer in continuous latent space.
- **Cosmos-Predict1-Autoregressive** (4B / 5B-V2W / 12B / 13B-V2W): discrete-token next-token prediction, LLM-style.

Pre-trained on **~100 M curated clips / 20 M raw hours** (driving 11 %, hand-object manipulation 16 %, human motion 10 %, spatial navigation 16 %, ego POV 8 %, nature 20 %, etc.). Post-trained for camera control, robotic manipulation (Bridge), and autonomous driving.

## Why it's in the reading pile
Less directly relevant than SCOPE / V-JEPA / PI-JEPA, but **the canonical "world simulator" baseline** the brainstorm's encoder might be compared against. Also, Cosmos *generates* sim-style videos with physical structure — useful as a **data source** for the brainstorm's sim training.

## Components
1. **Video curator** — shot detection (TransNetV2/AutoShot), GPU transcoding, motion/quality/text/type filtering, VILA-13B captioning, semantic dedup (InternVideo2 embeddings + k-means).
2. **Cosmos Tokenizer** — causal continuous + discrete video tokenizers with multiple compression ratios. Joint image-video tokenization (T=0 ⇒ image). Beats existing tokenizers by +4 dB PSNR.
3. **Diffusion WFM** — DiT with 3D RoPE, T5-XXL text encoder via cross-attention, AdaLN modulation. Trained with EDM denoising on continuous tokens.
4. **Autoregressive WFM** — Cosmos-Tokenize1-DV8×16×16-720p produces FSQ-quantized vocab of 64,000. 3D RoPE + 3D APE, cross-attention for text, QK-norm and z-loss for stability. Two-stage training: pure next-token prediction → +Stage 2 text conditioning. Medusa speculative decoding for 2–6× inference speedup.
5. **Diffusion decoder** — used to sharpen AR outputs (trained by adapting Cosmos-Predict1-7B-Text2World as a conditional denoiser on AR's discrete tokens).
6. **Guardrails** — pre-guard input filter and post-guard generation filter.

## Post-training applications
- **Camera control** (Sec 6.1): Cosmos-Predict1-7B-V2W-Sample-CameraCond, trained on DL3DV-10K with GLOMAP-derived poses. Enables 3D-navigable scene generation.
- **Robotic manipulation** (Sec 6.2): action-conditioned and instruction-conditioned variants on Bridge data.
- **Autonomous driving** (Sec 6.3): multi-view, trajectory-conditioned variants for driving simulation.

## 3D consistency & physics alignment (Sec 5.3)
Two-axis evaluation:

| Metric                  | Cosmos-Predict1-7B-T2W | Cosmos-Predict1-7B-V2W | Real videos |
|-------------------------|------------------------|------------------------|-------------|
| Sampson error ↓         | 0.355                  | 0.473                  | 0.431       |
| Pose estimation succ. ↑ | 62.6 %                 | 68.4 %                 | 56.4 %      |
| PSNR ↑                  | 33.02                  | 30.66                  | 35.38       |

Physics alignment via **PhysX + Isaac Sim** scenarios (free-fall, slopes, stable/unstable stacks, dominoes, seesaw, gyroscope). 800 1080p videos rendered as ground-truth; Cosmos predictions compared via pixel (PSNR/SSIM), feature (DreamSim), object (SAMURAI-tracked IoU). Headline: **9-frame video conditioning gives ~0.59 IoU on tilted-plane rollouts**, but failure modes include object impermanence (spontaneous appearance / disappearance), deformation, gravity violations. WFMs do not yet pass physics.

## Take-aways for the brainstorm
1. **Cosmos is a generator, not a representation encoder.** The brainstorm's encoder would be a different artifact, but Cosmos could provide *training data* (sim-only physical scenes with paired ground-truth: depth, segmentation, contact via the underlying Isaac Sim render).
2. The platform openly admits **physics adherence is not solved** even at 13B parameters with 20 M hours of video. Same conclusion as IntPhys 2 — scaling video alone is not enough.
3. **Cross-attention on text tokens** is the design they chose for text conditioning in both diffusion and AR models — converges with SCOPE's choice for the brainstorm's fusion block.
4. **Causal video tokenizer** is a strong piece of infrastructure to borrow if the brainstorm processes video (not images).
5. The **PhysX + Isaac Sim evaluation protocol** (Sec 5.3.2) is a ready-made physics-alignment benchmark that pairs with IntPhys 2: it gives object-tracking IoU on physically-grounded simulated scenarios — much more diagnostic than mean-surprise.

## Released artifacts
- Open-weight via NVIDIA Open Model License: https://github.com/nvidia-cosmos/cosmos-predict1
- ShotBench (shot detection): https://github.com/NVlabs/ShotBench
- TokenBench (tokenizer eval): https://github.com/NVlabs/TokenBench
