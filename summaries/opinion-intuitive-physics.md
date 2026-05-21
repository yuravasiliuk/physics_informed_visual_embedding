# Opinion: Learning Intuitive Physics May Require More than Visual Data

Su, LeGris, Gureckis, Ren. NYU. NeurIPS 2025 Workshop "Embodied World Models for Decision Making." arXiv 2512.06232v1. [PDF](../papers/2512.06232v1.pdf).

## TL;DR
- Pretrained a V-JEPA base model (215 M params, ViT-L/16, 224 px) on **SAYCam** (472 h of egocentric infant head-cam) — i.e., a developmentally realistic, ecologically valid video dataset, ~0.01 % of V-JEPA-2's training volume.
- Evaluated on **IntPhys 2** alongside V-JEPA-2-H/VM22M, V-JEPA-L-HowTo100M-0.1%, and a VideoMAE trained on SAYCam.
- **All four models score near chance (0.50–0.54), while humans hit 0.96.**
- Conclusion: **scaling visual data alone is not the bottleneck**. Future work should add embodied signals (actions, ego-motion, accelerometer, contact, etc.).

## Why this paper is the most direct *external* support for the brainstorm
The brainstorm explicitly says "supervise from simulator" — i.e., add non-visual signals (CoM, contact forces, angular velocities) that aren't in any unlabeled video corpus. This paper is a clean negative result against the "more video = better physics" hypothesis. If true, it implies:

1. **Any vision-only encoder, including V-JEPA 2 / DINO / SigLip, is likely missing exactly the physical structure the brainstorm wants to inject.**
2. The marginal value of *physics-grounded* supervision (sim-only heads) is probably higher than the marginal value of *more visual data*.
3. SAYCam (≈0.01 % data) matches V-JEPA-2-H (full scale) on IntPhys 2 — strong evidence that **distribution doesn't help either**; the issue is the learning signal, not the data.

## Method
- V-JEPA base, ViT-L/16-224, 215 M params; recipe from Bardes et al. 2024 / Garrido et al. 2025.
- 40 epochs (vs. 200) on SAYCam, 80/20 split per child, 2× A100.
- Evaluated via **violation-of-expectation surprise metric**: `S_t = ||p_φ(f_θ(V_{t:t+C})) − g_ψ(V_{t:t+C+M})||_1`, averaged across frames; classify pair as "possible" if its mean surprise < its "impossible" counterpart's.
- Compared against: V-JEPA-2-H-VM22M (full-scale), V-JEPA-L-0.1%-HowTo100M, MAE-SAYCam.

## Results
| Model                       | Data        | Volume       | IntPhys 2 acc |
|-----------------------------|-------------|--------------|---------------|
| V-JEPA-B-SAYCam (ours)      | egocentric  | 472 h        | 0.50          |
| V-JEPA-2-H-VM22M            | mixed       | 1.73 M h     | 0.52          |
| MAE-SAYCam                  | egocentric  | 472 h        | 0.53          |
| V-JEPA-L-0.1%-HowTo100M     | exo tutorial | 128 h        | 0.54          |
| Human                       | —           | —            | **0.964**     |
| Chance                      | —           | —            | 0.50          |

The surprise metric is also highly noise-sensitive — the small differences above are within noise.

## Take-aways for the brainstorm
1. **The brainstorm's premise — that supervisory signals from a simulator are necessary — is consistent with the empirical literature, not just speculation.** Cite this paper to justify the approach.
2. The authors propose embodied / action-annotated / multimodal datasets (BabyView with accelerometers, etc.) as the natural next step. The brainstorm's sim-only heads (angular velocities, contact forces, distance-to-ground) are exactly this kind of signal, but rendered for free in simulation rather than collected from sensors.
3. **Evaluation hook**: the resulting embedding should be probe-tested on IntPhys 2. If sim-only physical supervision is doing anything useful, it should move the needle above the ~0.52 ceiling reported here.
4. The argument also doubles as a hedge against scaling-only paths (Cosmos-Predict, V-JEPA-2 with even more data) — the same authors of V-JEPA-2 sit on the acknowledgements of this paper.

## Limitations the authors flag
- SAYCam is still much smaller than the real visual data a child sees.
- SAYCam is purely visual — no proprioception, no contact, no self-motion. The right counterexample may need a fully embodied dataset.
- Surprise metric is fragile; the field needs better evaluation than mean-frame-surprise comparisons.

Code: https://github.com/eysu35/vjepa
HF: https://huggingface.co/eys8549/vjepa-saycam
