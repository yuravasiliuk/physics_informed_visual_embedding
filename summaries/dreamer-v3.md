# DreamerV3 — Mastering Diverse Control Tasks Through World Models

Hafner, Pasukonis, Ba, Lillicrap. Nature 640, 17 April 2025 (originally arXiv 2301.04104, 2023). [PDF](../papers/mastering_diverse_control_through_world_models%20.pdf).

> **Note**: only the abstract/intro of this paper were inspected for this summary. The detailed algorithm description (loss balancing, symlog transforms, world-model recipe) was not re-read here — refer directly to the PDF for those specifics.

## TL;DR
A general model-based RL algorithm with a **fixed hyperparameter configuration** that solves 150+ tasks across 8 domains (Atari, ProcGen, DMC, Crafter, BSuite, Minecraft, …) and is the **first algorithm to collect diamonds in Minecraft from scratch** without human data or curricula. Three networks:
1. **World model** — latent dynamics model (RSSM-style) that predicts the outcomes of actions.
2. **Critic** — value of imagined trajectories.
3. **Actor** — chooses actions that maximize critic-imagined return.

All training happens in the *imagination* of the world model after observation; the real environment is only used for new rollouts to refine the world model.

## Why it's in the reading pile
- Reference baseline for "**train a latent world model from observations, plan/act in latent space**." V-JEPA-2-AC inherits this philosophy in feature-prediction style; Dreamer does it in (decoded) pixel space with a discrete-token RSSM.
- Provides the **robustness tricks** (normalization, balancing, symlog/symexp value transforms) that make latent-world-model training stable across domains — relevant if the brainstorm's encoder is ever trained jointly with a downstream policy.

## Where it touches the brainstorm
- **Different problem statement.** Dreamer is RL with reward; the brainstorm is a representation-learning project with multi-head supervision. But:
  - The world model in Dreamer is essentially "predict the next latent state given action," matching V-JEPA-2-AC's role. If the brainstorm wants closed-loop simulator interaction, Dreamer's actor-critic loop on top of the embedding is the canonical recipe.
  - Dreamer's success on Minecraft demonstrates that **a learned latent model + planning is sufficient for long-horizon physical task execution** — provided the latent is well-structured. The brainstorm's bet is to *make* the latent well-structured via simulator supervision, not via further RL.
  - Dreamer's normalization tricks are likely transferable to JEPA-style heads (especially if scalar physical quantities like contact-force magnitude span many orders of magnitude — Dreamer's symlog transform handles this cleanly).

## Not directly examined in this reading
- Specific loss formulation, EMA tricks, return normalization.
- Minecraft curriculum details.
- Model sizing study.

If the brainstorm proceeds toward closed-loop control on top of the embedding, this paper deserves a deeper read.
