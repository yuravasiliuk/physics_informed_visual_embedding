# IntPhys2 Option-A — zero-shot V-JEPA 2 surprise (baseline)

Status: **v0 implemented, unverified on GPU.** Code: [src/intphys/](../../src/intphys/).

## What this is
The **strong baseline** for the baseline-vs-physics-supervised comparison
(gate-b-probe.md sibling): a *frozen* V-JEPA 2 world model scored zero-shot on
IntPhys2 by its own pretraining objective used as a violation-of-expectation
signal. No training, no IntPhys2 labels at fit time.

## Recipe (faithful to app/vjepa/train.py loss)
```
h = layer_norm(target_encoder(clip))            # target reps, all tokens
z = predictor(encoder(clip, masks_enc), masks_enc, masks_pred)
surprise = mean |z - h[masks_pred]|             # L1 in representation space
```
Masks are **temporal**: context = first `t_ctx` token-frames of a window,
pred = next `t_pred`. Slide the window across the video; **max** over windows
localizes the violation (headline score), **mean** is the global number.

## Decisions pinned
- **Action-free predictor** (`vjepa2_vit_giant_384`), NOT the action-conditioned
  vjepa2-AC of `energy_landscape_example.ipynb` — that one needs 7-DoF
  actions/states and is for robot MPC, not passive VoE.
- **Local full checkpoint** `vjepa2/weights/vitg-384.pt` carries `encoder` +
  `predictor` + `target_encoder` → nothing downloads. Construction mirrors
  `src.hub.backbones._make_vjepa2_model(model_name="vit_giant_384")`,
  `pretrained=False`, then loads the three sub-state-dicts.
- **Metric = relative pairwise accuracy** (impossible more surprising than its
  matched possible twin), printed by `run_intphys2.py`. Threshold-free, no
  label leakage — the number V-JEPA 2 reports. The predictions CSV also carries
  a count-based within-SceneIndex `response` for parity with the official
  VLM-oriented scorer (`~/IntPhys2/ComputeScoreIntPhys2.ipynb`), but that path
  leaks the per-scene impossible count and is secondary.

## Data
`~/data/IntPhys2/{Debug,Main,HeldOut}/{Videos,metadata.csv}`. Develop on
**Debug** (small), report on **Main** (has `condition` ∈ permanence /
immutability / continuity / solidity and Easy/Medium/Hard env splits the
official scorer slices on). **HeldOut** has no labels (test server).

## Open / must-verify on GPU before trusting numbers
1. **Mask-index convention.** Token grid is flattened (T',H',W') row-major,
   S=576 spatial tokens/frame at 384²; `_temporal_masks` assumes token-frame f
   owns `[f·S,(f+1)·S)`. Validate against a real `encoder(clip, masks_enc)` /
   `predictor` forward (shapes, and that pred tokens align with `h[masks_pred]`).
2. **Window protocol.** `window/t_ctx/t_pred/stride` are untuned guesses; sweep
   on Debug and watch that the per-window curve spikes at the scripted violation
   frame before believing the aggregate.
3. **Frame sampling / clip length.** Pretrain was 64 frames; RoPE allows other
   lengths but very short windows may degrade. Tune `--fps`.
4. CPU is import-safe only; ViT-g surprise needs cuda + fp16.

## Next
Once Debug numbers look sane and beat chance (0.5 relative), run Main and slice
by `condition`. Then Option B (trained lightweight predictor + physics aux
heads) reuses this exact scorer for an apples-to-apples delta.
