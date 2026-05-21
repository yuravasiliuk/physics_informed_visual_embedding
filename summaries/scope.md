# SCOPE — Semantic Conditioning for Sim2Real Category-Level Object Pose Estimation

Hönig et al., 2025 (TU Wien). arXiv 2509.24572. [PDF](../papers/2509.24572v1.pdf).

## TL;DR
Train a U-Net denoising diffusion model to regress a NOCS map from `concat(RGB, surface-normals)`, while **conditioning the U-Net via cross-attention on frozen DINOv2 features** (CLS + patch tokens). Trained on synthetic data (CAMERA / CAMERA-BPR), evaluated zero-shot on real (REAL275, YCB-V, TYOL). Beats prior synthetic-only SOTA by 31.9 % on 5°/5 cm and grasps unseen objects on a real robot at 90–100 % success.

## Why this paper is the closest baseline for the brainstorm
The user's diagram is essentially SCOPE's data-flow:

| Brainstorm element                  | SCOPE element                                       |
|-------------------------------------|-----------------------------------------------------|
| Geometry encoder (conv) on RGB+aux  | U-Net conv encoder on `concat(RGB, normals)` crops  |
| Frozen semantic encoder (DINOv2)    | Frozen DINOv2 (CLS + patch tokens)                  |
| Cross-attention fusion              | Cross-attention layers at down-block 5 and up-block 2 |
| Supervision = simulator quantity    | NOCS map (rendered in BlenderProc); pose via TEASER++ |

SCOPE explicitly ablates **cross-attention vs. concatenation** on objects outside the training category distribution: ADD-S jumps from 59.5 → 76.0 (YCB-V) and 60.6 → 82.5 (TYOL). Concatenation requires a category label; cross-attention with continuous DINOv2 semantics does not. This is direct evidence for picking cross-attn over concat in the brainstorm fusion block.

## Method details
- **Conditional DDPM**, x₀ = NOCS image (H×W×3). Standard noise schedule, 5 denoising steps with DPM++ at inference (0.193 s total).
- **U-Net input**: 6-channel `concat(IRGB, Inormals)`, segmented crops from an off-the-shelf detector.
- **Semantic input S**: DINOv2 patch + CLS tokens. Q comes from U-Net feature map; K, V from S. Self-attention is run in parallel using only U-Net features.
- **Pose recovery**: predicted NOCS → TEASER++ on 500 sampled points → (R, t, scale).
- **Symmetry**: handled implicitly by the diffusion process (no symmetry-aware loss).
- **Sim2Real bridge**: photorealistic CAMERA-BPR with BlenderProc + domain randomization, Perlin noise on depth, dropout on normals.

## Useful empirical knobs
- DINOv2 ≫ CLIP-ViT/L14 as the semantic backbone (49.0 vs 28.1 @5°5cm). CLIP excels at text-image alignment, DINOv2 at spatial consistency — relevant when choosing the frozen branch.
- 5 denoising steps and 500 TEASER points are the sweet spot; refinement (multi-noise sampling) buys +2.4 % at 7.5× cost.
- The "Sim2Real gap" on YCB-V shrinks to 1.1 % on 5°/5 cm vs. methods that *do* see real data — a strong claim for the pure-sim training regime.

## Connection to brainstorm supervision heads
SCOPE's *single* supervision target is the NOCS map (a sim-only signal, since real objects rarely have ground-truth canonical coordinates). The brainstorm proposes many heads (depth, normals, segmentation as sim+real; center-of-mass, contact forces as sim-only). SCOPE shows the **architecture scales to at least one rich sim-only target**; it does not test multi-head supervision.

## Limitations to note for the brainstorm
- Single-task: only NOCS supervision; no multi-head ablation.
- Trained per-crop (segmented object), not on the whole frame — the brainstorm seems to operate on whole RGB.
- The "geometry" branch already eats normals (depth-derived). The brainstorm wants to *predict* normals/depth as supervision, which is a different setting — these channels can't be inputs and targets at the same time.

## Related ideas to chase from this paper
- DiffusionNOCS (Ikeda et al., 2024) — concatenation baseline that SCOPE beats; also worth reading the cross-attn ablation in its own right.
- SecondPose (Chen et al., 2024) — concatenates DINOv2 with Point-Pair Features (depth-derived); shows the value of explicit geometric features.

Code: https://github.com/hoenigpeter/scope
