# Baseline Reading Notes — Physics-Informed Embedding Brainstorm

Working notes on the papers in [../papers](../papers), grouped by relevance to the proposed architecture in [../Physics informed embedding.pdf](../Physics%20informed%20embedding.pdf):

> RGB → frozen semantic encoder (JEPA/DINO/SigLip) + learned conv "geometry" encoder → cross-attention fusion → supervision heads. **Sim+real** heads: normal maps, segmentation, depth. **Sim-only** heads: angular velocities, center-of-mass, contact forces, distance to ground plane.

## Most directly relevant
- **[scope.md](scope.md)** — DINOv2 semantics + geometry (RGB+normals) fused via **cross-attention** to a U-Net diffusion model for Sim2Real 6D pose. Validates the exact fusion mechanism in the brainstorm diagram, and shows cross-attn ≫ concatenation.
- **[roboscape.md](roboscape.md)** — Dual-branch AR world model with **auxiliary depth + keypoint-dynamics heads** on top of RGB prediction. Same architectural class as the brainstorm but for generation, with labels from pretrained perception models (not a simulator). Closest direct competitor.
- **[why-auxiliary-tasks-jepa.md](why-auxiliary-tasks-jepa.md)** — Theoretical: the **auxiliary head determines which distinctions the encoder must preserve**. Picks of "center of mass / contact forces / depth" are not arbitrary — they define the equivalence relation the embedding will learn.
- **[pi-jepa.md](pi-jepa.md)** — PDE-domain JEPA with physics-residual regularizers. Different domain (subsurface flow) but the recipe — split predictor bank per physical sub-process, per-process residual loss — is transferable.
- **[crossjepa.md](crossjepa.md)** — Cross-modal JEPA (2D→3D) without masking. Inverted goal vs. the brainstorm, but its **latent-information sink** (feed nuisance factors like camera pose and color histogram into the *predictor*) is the most directly portable design idea for sim2real nuisance decoupling.

## Frozen-encoder candidates / world models
- **[v-jepa2.md](v-jepa2.md)** — V-JEPA 2 video encoder + action-conditioned predictor. Strong candidate for the "frozen semantic encoder" branch; also shows zero-shot robot planning from RGB alone.
- **[v-jepa2-1.md](v-jepa2-1.md)** — V-JEPA 2.1: diagnoses and fixes V-JEPA 2's weak *dense/local* features (context tokens were unsupervised, behaved like register tokens). Directly bears on the Rung-0 gating probe: a "no signal" result on vanilla V-JEPA 2 is now confounded with this fixable artifact. Ships a 300M distilled variant near 2B-teacher quality.
- **[dreamer-v3.md](dreamer-v3.md)** — Latent world model + actor-critic; general-purpose RL across 150+ tasks. Reference point for "learn a world model in latent space."
- **[cosmos-predict1.md](cosmos-predict1.md)** — NVIDIA Cosmos WFM platform (diffusion + autoregressive video world models, video tokenizers, post-training for camera/robot/driving).
- **[cosmos-predict2.5.md](cosmos-predict2.5.md)** — Cosmos-Predict2.5 / Cosmos-Transfer2.5: flow-matching WFMs and a control-net Sim2Real / Real2Real translator. The Transfer2.5 idea (sim → photorealistic real) is a direct **alternative** to the brainstorm direction (real & sim → shared embedding).

## Evidence problem & evaluation
- **[opinion-intuitive-physics.md](opinion-intuitive-physics.md)** — Position paper: SOTA video models (incl. V-JEPA 2) sit near chance on IntPhys 2. Argues **visual data alone is insufficient**; embodied / action-annotated / simulator-derived signal is needed. This is the strongest external argument *for* the brainstorm's "supervise from simulator" thesis.
- **[intphys2.md](intphys2.md)** — The benchmark itself: permanence / immutability / continuity / solidity, with photorealistic Unreal scenes, dynamic cameras, occlusions. Humans 96%, best model ~58%. Likely target benchmark.

## Tangential / context
- **[cosmos-policy.md](cosmos-policy.md)** — Fine-tunes Cosmos-Predict2 into a joint policy/world-model/value via *latent frame injection*. Different end product (policy, not encoder), no physics labels, but mild empirical support for multi-target supervision and a concrete contrast point for "video pretraining alone vs explicit physics supervision."
- **[omnicontrol.md](omnicontrol.md)** — Spatially-controllable human motion diffusion (any joint, any time). Not directly relevant; could inform "control signal as supervision" framing.
- **[rt-h.md](rt-h.md)** — RT-H: language-action hierarchy for robot policies. Tangential; useful only if the embedding feeds a VLA.

## Not summarized (skipped per request)
- `kimodo_tech_report.pdf` — not read.
