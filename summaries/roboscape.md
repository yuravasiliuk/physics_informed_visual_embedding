# RoboScape — Physics-informed Embodied World Model

Shang, Zhang, Tang, Jin, Gao, Wu, Li. Tsinghua University / Manifold AI. arXiv 2506.23135v1, June 2025. [PDF](../papers/RoboScape_%20Physics-informed%20Embodied%20World%20Model%20-%202506.23135v1.pdf).

## TL;DR
A **masked-autoregressive video world model** for robot manipulation that augments standard next-frame RGB prediction with **two physics-informed auxiliary tasks**, trained jointly:

1. **Temporal depth prediction** — a parallel depth branch (dual-branch co-autoregressive Transformer, DCT) predicts the next frame's depth map and additively fuses its intermediate features into the RGB branch via a learned linear projection at every ST-block. Enforces 3D geometric consistency.
2. **Adaptive keypoint dynamics** — densely sample keypoints in the first frame ([SpatialTracker](https://github.com/) tracking), keep the top-*K* by motion magnitude, then enforce that the visual tokens at those keypoint locations remain *consistent* across frames (alignment to frame 1 in feature space). Implicitly encodes object material (rigidity / softness / deformation) without explicit material modeling.

A **keypoint-guided attention mask** boosts the cross-entropy weight on active-keypoint regions (γ = 5).

Trained on 50k AgiBotWorld-Beta clips → ~6.5 M 16-frame windows; 5 epochs × 32 A800-80GB. Beats IRASim, iVideoGPT, Genie, CogVideoX on appearance fidelity (LPIPS 0.126, PSNR 21.85), geometric accuracy (AbsRel 0.36, δ₁ 0.62, δ₂ 0.83) and action controllability (ΔPSNR 3.34).

## Why it's in the reading pile — directly maps to the brainstorm
This is the **closest concrete instantiation** of the brainstorm's idea in the literature, and the parallel is tight enough that the new theoretical paper ([plan](../plans/theory/theoretical-paper-plan.md)) needs to position against it explicitly.

| RoboScape                                              | Brainstorm's physics-informed embedding         |
|--------------------------------------------------------|--------------------------------------------------|
| Two auxiliary heads on top of RGB prediction           | Multiple auxiliary heads on top of an encoder    |
| Heads: depth + keypoint-token consistency              | Heads: normals, depth, segmentation, CoM, ω, contact, ground-distance |
| Labels from **pretrained perception models** (Video Depth Anything, SpatialTracker) | Labels from **simulator ground truth**     |
| Output is a *generative video model*                   | Output is a *representation encoder*             |
| Single domain: real robotic manipulation videos        | Cross-domain: sim + real, with sim-only labels   |
| Cross-branch feature fusion: additive linear projection | Cross-attention fusion (per SCOPE)              |
| All heads supervised on the same data                  | Sim+real heads on both domains, sim-only on sim only |

So RoboScape is the **same-shape architecture serving a different end product** (video generation vs encoder) with **different supervision sources** (off-the-shelf pretrained models vs simulator).

## Architectural details worth borrowing
- **Dual-branch ST-Transformer with causal temporal + bidirectional spatial attention** — same skeleton an encoder version could use, just lopping off the generative target.
- **Additive cross-branch fusion** (eq. 3): `h_RGB ← h_RGB + W·h_depth`. Simple and apparently sufficient — challenges SCOPE's claim that cross-attention is uniformly better. Worth re-checking in the brainstorm context.
- **Adaptive keypoint sampling by motion magnitude** (top-*K* over `Σ‖p_{t+1}−p_t‖`) avoids the need for segmentation masks and concentrates supervision where the physics is interesting. Directly transferable to the brainstorm's contact-force head (active-contact regions ≈ high-motion-with-impact regions).
- **Keypoint-guided attention mask** (γ-multiplier on the cross-entropy loss in keypoint cells, eq. 6–7) is a cheap loss-weighting trick to make sparse signals matter.
- **MAGVIT-2 tokenizer** for both RGB and depth — reuses one tokenizer family across modalities.

## What RoboScape does NOT do (gaps the brainstorm can claim)
1. **No simulator labels.** Depth comes from Video Depth Anything (a pretrained network), not from a renderer. So the depth labels carry the *bias* of the pretrained model, not the ground-truth structure of a simulator. The brainstorm replaces this with truly physical labels.
2. **No truly physical quantities.** Depth is geometry; keypoint consistency is a soft surrogate for material. The brainstorm's CoM, ω, contact forces are *forces and dynamics*, not derivable from any pretrained perception model.
3. **No domain-asymmetric supervision.** All labels available on all training data. The brainstorm's hardest design point (sim-only heads + the domain-marginalization regularizer in [the plan](../plans/theory/theoretical-paper-plan.md) §2) is untouched here.
4. **No equivariance.** Keypoint consistency is a coordinate-space alignment, not a group-equivariant constraint. Plan §3 still applies.
5. **No counterfactual / differential supervision** (plan §4) — RoboScape is supervised on observed videos only, no per-parameter perturbations.
6. **No conservation laws** (plan §5) — keypoint consistency is not the same as momentum or energy conservation; it's an L2 token alignment.
7. **Generator, not encoder.** The output is a video model used downstream as a (a) data generator for policy training, (b) policy evaluator (Pearson 0.953 with sim success rate). The brainstorm's output is a frozen-able encoder feature.

## Empirical findings to remember
- **Ablation** (Table 2): removing depth keeps LPIPS/PSNR roughly flat but hurts δ₁ (0.62 → 0.58); removing keypoint drops ΔPSNR (action controllability) from 3.34 → 2.95; removing both drops ΔPSNR to 1.99. **Keypoint loss is what gives action-conditioning teeth.** Implication for the brainstorm: a dynamics-flavored head (ω, contact-active points) is what will likely make the encoder action-discriminative, not the static heads (depth, normals).
- **Policy training with synthetic data** (Table 3): on LIBERO, 800 generated trajectories beat 200 real ones (79.1 % vs 65.2 % average). Generation is a viable data-augmentation source.
- **Policy evaluation**: RoboScape's success-rate prediction correlates r=0.953 with the ground-truth simulator across DP checkpoints; IRASim r=−0.13, iVideoGPT r=−0.20. A useful evaluation-mode role for the encoder if it ever feeds a world model.
- **Scaling**: monotonic improvement from 34M → 131M → 544M params (Fig. 6). Data scaling 1M → 6M clips improves visual + action metrics, but geometric metrics plateau or regress slightly — small datasets *overfit the conditioning frame*, inflating geometric scores artificially.

## Take-aways for the brainstorm and the planned paper
1. **Position RoboScape as the strongest direct competitor / closest baseline** alongside [SCOPE](scope.md) — same multi-head + cross-branch fusion architectural class, different objective.
2. **Adopt the adaptive-keypoint trick** as a candidate replacement for explicit segmentation/contact masks when defining the brainstorm's contact-force head.
3. **Argue for simulator labels over pretrained-perception labels.** RoboScape's depth supervision *inherits the failure modes of Video Depth Anything*. A simulator does not. This is a clean argument for plan §2 (domain-asymmetric supervision).
4. **Use RoboScape's ablation as motivation** for the brainstorm's mix of static (depth/normals/segmentation) and dynamic (ω/contact/CoM) heads: their ablation shows the dynamic-flavored head is what carries action-conditioning, which lines up with plan §5 (conservation laws / dynamical invariants).
5. **Generator vs encoder is not mutually exclusive** — if the brainstorm encoder ever needs to feed a world model, RoboScape's autoregressive backbone with depth-fed RGB is a ready recipe.

## Released artifacts
Code: https://github.com/tsinghua-fib-lab/RoboScape

## Not deeply read
- Section-level details of TransNetV2 shot detection / Intern-VL captioning in the data pipeline (Sec 2.2).
- Exact loss-coefficient sweeps beyond the headline λ₁=1, λ₂=0.01, λ₃=1, γ=5.
- Per-task breakdown of LIBERO results.
