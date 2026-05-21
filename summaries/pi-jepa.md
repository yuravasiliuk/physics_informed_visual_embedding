# PI-JEPA — Physics-Informed JEPA for Multiphysics PDE Surrogates

Yee & Koh, 2026. arXiv 2604.01349v3. [PDF](../papers/pi-jepa.pdf).

## TL;DR
A JEPA variant for **subsurface flow PDE surrogates** that pre-trains on unlabeled coefficient fields (free to generate) and fine-tunes on a small number of expensive labelled simulations. Three ingredients:

1. **Masked latent prediction** in V-JEPA style on PDE solution fields.
2. **A bank of per-sub-operator predictors** aligned with the Lie–Trotter operator splitting of the governing PDEs — separate latent transformer module per physical sub-process (pressure, saturation, reaction).
3. **Per-sub-operator PDE residual loss** evaluated through a lightweight decoder (training-only).
4. **VICReg covariance regularizer** to prevent collapse.

Results: 1.8–2.7× lower error than FNO/PINO/DeepONet baselines in the low-data regime (≤100 labelled trajectories).

## Why it's in the reading pile (despite being a different domain)
PI-JEPA is the cleanest example in the literature of *baking physics-residual losses into a JEPA encoder*. The brainstorm proposes physically meaningful auxiliary heads (CoM, contact forces); PI-JEPA proposes physically meaningful auxiliary **residuals**. The ablation isolates how much each piece matters.

## Architecture summary
- **Context / Target encoder** `f_θ` / `f_ξ`: shared architecture (Fourier-enhanced conv + transformer). Target is EMA of context with τ ∈ [0.99, 0.999].
- **Predictor bank** `{g_φ1, ..., g_φK}`: one lightweight transformer per sub-operator. Chained: `z_c → g_φ1 → ẑ_t^(1) → g_φ2 → ẑ_t^(2) → ...`. Mirrors numerical operator splitting.
- **Auxiliary physics decoder** `d_ψ_k`: only used during pretraining; maps each predictor stage's latent back to physical space so the PDE residual `R_k` can be evaluated via autodiff.
- **Loss**: `L = L_pred + λ_p Σ_k L_phys^(k) + λ_r L_reg`. `L_pred` = L2 on predicted vs. EMA-target latent. `L_phys^(k)` = PDE residual at collocation points. `L_reg` = VICReg variance + covariance terms.
- **Spatiotemporal block masking**: context is a contiguous spatial block at time t; target is a *spatially displaced* block at t+Δt. Models must internalize advection/diffusion to predict.

## Ablation honesty (important)
Table 5 in the paper isolates contributions:
- **Operator splitting**: +16 % over a monolithic predictor.
- **VICReg regularization**: +17.7 %.
- **PDE residual loss itself**: **neutral** — no measurable benefit on top of the above.

The authors honestly flag this and point future work toward stronger physics-informed objectives. **Caveat for the brainstorm**: residual-style losses on RGB scenes (e.g. "predict force, check if Newton's law holds") may be similarly easy to satisfy in a trivial way unless the residual operator is non-trivially constraining.

## Transferable design ideas
1. **Operator splitting → multi-head splitting.** PI-JEPA's per-sub-operator predictor bank is the *structural* counterpart to the brainstorm's multi-head supervision. The lesson: keep predictor capacity *separated* by physical process, don't share one head across CoM + contact + segmentation.
2. **Auxiliary decoder only at training.** PI-JEPA decodes the latent back to physical fields only to compute the residual. The brainstorm can do the same: predict CoM / forces only from latent during training, drop the heads at inference.
3. **Spatiotemporal masking carries physics.** Drawing target patches at a *displaced* spatial location at t+Δt forces the encoder to internalize the dynamics — much stronger than naive future-frame masking.
4. **EMA target encoder is essential for stability.** Confirmed across the JEPA family.

## Relevance map
- Different domain (subsurface flow, not RGB scenes).
- Same overall recipe: masked latent prediction + per-process predictors + physics regularization.
- Strongest takeaway: **architecture (operator splitting) and collapse prevention (VICReg) dominate; physics-residual loss is hard to make load-bearing.** Plan ablations accordingly.
