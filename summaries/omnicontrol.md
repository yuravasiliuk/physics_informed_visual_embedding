# OmniControl — Control Any Joint At Any Time for Human Motion Generation

Xie, Jampani, Zhong, Sun, Jiang. ICLR 2024. arXiv 2310.08580v2. [PDF](../papers/2310.08580v2.pdf).

> **Note**: only abstract / first page examined.

## TL;DR
A text-conditioned **diffusion model for human motion generation** that accepts flexible spatial control signals over *any joint at any time* (not just the pelvis, as in prior inpainting methods). Two complementary mechanisms:
- **Analytic spatial guidance** — ensures the generated motion tightly conforms to the input spatial constraints.
- **Realism guidance** — refines all joints together to avoid the per-joint artifacts that pure inpainting introduces.

State-of-the-art on HumanML3D and KIT-ML, both on pelvis control and on previously unsupported joints.

## Why it's in the reading pile
The connection to the brainstorm is **indirect**: OmniControl shows that a diffusion model can be *conditioned on sparse physical-quantity signals* (3D joint locations at specific timesteps) while still producing physically coherent global motion. The brainstorm's sim-only heads (CoM trajectory, angular velocity) are similar in flavor — sparse, physically meaningful conditioning over time.

## Possible relevance
- If the brainstorm's downstream use is **human/character motion synthesis** rather than perception, OmniControl is a candidate decoder/generator that consumes the kind of physical signals the encoder is being trained to predict.
- The **analytic + realism dual-guidance** design is a useful pattern for combining hard physical constraints with learned priors — applicable when a head produces a control signal that must be physically consistent (e.g. predicted CoM should be consistent with predicted contact forces).

## Not deeply read
Architecture details, guidance formulations, datasets, and the realism module are not summarized here. If the brainstorm pivots toward generation, return to this paper.

Project: https://neu-vi.github.io/omnicontrol/
