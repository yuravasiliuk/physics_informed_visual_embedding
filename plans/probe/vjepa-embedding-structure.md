# V-JEPA 2 embeddings: feature volume, not an indivisible matrix

Working note for the dense-readout work (depth/normals from frozen features).
Captures the "is `(9216, 1408)` one holistic embedding?" discussion **as claims to
verify against the paper and the repo** — nothing here is settled until checked
against the boxes in §4.

## 1. The claim
A clip's frozen output `emb.shape == (9216, 1408)` is **not** a flat embedding
vector and **not** an indivisible blob. It is a structured **feature volume**:

```
9216 = T' × H' × W' = 16 × 24 × 24      (at 384², patch 16, tubelet 2, 32 input frames)
1408 = ViT-g embed dim
emb.reshape(16, 24, 24, 1408)           # (time, y, x, features)
```

Each cell is tied to a specific (time, y, x) location, **and** — because of
attention — carries whole-clip context. So it is *both* spatially/temporally
addressable *and* globally contextualized.

## 2. Two things that are independent (the crux)
- **(A) Each token is globally contextualized.** 40 layers of self-attention
  mix every token with every other across space *and* time, so a token's
  receptive field is the entire clip (richer than a CNN feature-map cell, whose
  receptive field is a local window). TRUE — this is what "end-to-end
  spatio-temporal encoding" means, and it genuinely helps depth/normals
  (temporal parallax resolves depth-scale; motion disambiguates occlusion).
- **(B) Therefore the decoder must consume all 9216×1408 jointly.** FALSE and
  unnecessary. The spatio-temporal integration happens *inside the frozen
  encoder*; by read-out time it is baked into each 1408-vector. A per-token head
  reads a *localized* output (depth at this t,y,x) from a *globally-informed*
  vector — it discards nothing temporal, because the temporal info is inside the
  token.

Conflating (A) and (B) is the error. (A) is about where mixing happens (the
encoder, already done, frozen); (B) wrongly demands the *head* redo it.

## 3. Consequence for the readout head
- A **per-token linear/shallow head with weight sharing** (like a 1×1 conv on a
  feature map) uses the structure via *layout* (each token → its pixel's depth)
  and *content* (each vector already global). It stays tiny (`1408→3` = 4227
  params) yet throws nothing away. → the clean **diagnostic**: any signal it
  recovers was put there by the encoder.
- A **token-mixing decoder** (DPT / attention across the grid) adds
  spatio-temporal reasoning *in the head*. Stronger, but now a good score may be
  the head's doing, not the representation's — same attribution confound as
  unfreezing. → a later step, not the baseline.

Mental model to keep: **a `(T'×H'×W'×D)` volume of globally-contextualized
tokens.** "Embedding matrix" is fine; "indivisible" is the word to drop.

## 4. Verify against paper + code before trusting the above
- [x] **Token count & factorization.** Confirmed directly in
      `vjepa2/src/models/utils/patch_embed.py` (`PatchEmbed3D.forward`): Conv3d
      output `(B,D,T',H',W')` → `.flatten(2).transpose(1,2)`. `9216 = 16·24·24`
      holds for 32 frames @ 384 (tubelet 2, patch 16); `encoders.py` hard-asserts
      this for every clip it runs.
- [x] **Flatten order is row-major (T', H', W').** Confirmed — PyTorch `flatten`
      is row-major/C-order, so token `i = t·H'·W' + h·W' + w`. Matches
      `encoders.py`'s assert and `surprise.py`'s `_temporal_masks` token-frame
      indexing (`S = tokens_per_tframe`, frame `f` owns `[f·S, (f+1)·S)`).
- [x] **No CLS token.** Confirmed in `vision_transformer.py:forward` — patch
      tokens flow straight into the blocks; nothing is concatenated.
- [x] **Global (not windowed) attention + 3D-RoPE.** Confirmed in
      `modules.py:RoPEAttention.forward` — `scaled_dot_product_attention` over
      *all* N tokens, `is_causal=False`, no windowing, every block; rotation is
      split into separate depth/height/width components (3D-RoPE) per
      `separate_positions`. Stronger than stated: the encoder is **acausal over
      the whole clip** (the `t=0` token has already attended to `t=31`).
- [ ] **Paper cross-check.** V-JEPA 2 paper (`papers/v-jepa2*.pdf`): tubelet
      2×16×16, ViT, representation-space prediction. Confirm tubelet/patch sizes
      and that the readout/attentive-probe evals decode from the token field
      (not a pooled vector) — i.e., that dense decoding from `(T',H',W',D)` is the
      intended use.
- [ ] **Temporal-helps-depth is an assumption, not shown here.** The claim that
      spatio-temporal beats per-frame for *our* depth/normals targets is testable:
      compare frozen V-JEPA vs a per-frame image encoder (DINOv2) on the same
      token-resolution depth/normals readout. Don't assert it until measured.
- [ ] **NEW — per-token content may not equal local content (§5).** V-JEPA 2.1
      shows vanilla V-JEPA 2's context tokens are register-like, not locally
      faithful, under the exact Arm-2a (linear, per-token) protocol. Run the
      label-free cosine-similarity diagnostic (§5) before trusting any Arm 2a
      null result on CoM-offset as "physics absent."

## 5. Verified against code (not just the paper) — and one real confound found

Checked directly against `~/vis_emb/vjepa2/src/models/{vision_transformer,utils/patch_embed,utils/modules}.py`:
- **Token factorization, row-major (T',H',W').** Confirmed: `PatchEmbed3D.forward`
  does `self.proj(x).flatten(2).transpose(1,2)` on a `(B,D,T',H',W')` conv output;
  PyTorch `flatten` is row-major, so token `i = t·H'·W' + h·W' + w`. Matches
  `encoders.py`'s hard assert and `surprise.py`'s `_temporal_masks` indexing.
- **No CLS token.** Confirmed — `forward()` never concatenates a class token;
  only patch tokens flow through the blocks.
- **Global, non-causal attention, every block.** Confirmed — `RoPEAttention.forward`
  runs `scaled_dot_product_attention` over *all* N tokens with `is_causal=False`,
  no windowing, at every one of the 40 blocks (unmasked inference). Stronger than
  §2 states: the encoder is **acausal over the whole clip** — even the `t=0` token
  has already attended to `t=31` before the readout head ever sees it.

So §1–§2's rejection of "indivisible" is right: `(9216,1408)` is a real,
code-verified `(T',H',W',D)` structured tensor, not an assertion.

**But §2's conclusion from (A)→¬(B) overclaims, and the literature now pins down
exactly how.** "Each token already carries whole-clip context, so a per-token
head discards nothing" conflates two different claims:
- *architectural capacity* — full attention *could* preserve local+global jointly
  in each token (true, verified above), with
- *trained behavior* — whether the **specific checkpoint's training objective**
  actually pressured context tokens to keep local content (a separate, empirical
  question).

[V-JEPA 2.1](../../summaries/v-jepa2-1.md) (read after this note was drafted) is a
direct falsification of the second claim for vanilla V-JEPA 2 — the exact
checkpoint `encoders.py` loads (`vitg-384.pt`). Their own diagnosis: under the
masked-only loss, context tokens have "no incentive to encode local information...
[and] devote this computation to aggregating global information... similarly to
register tokens" (Darcet et al. 2023). Their evidence is **literally Arm 2a's
methodology** — one linear layer per token — scoring 22.2 mIoU (ADE20K) / 0.682
RMSE (NYUv2) on vanilla V-JEPA 2, recovered to 47.9 / 0.307 only after a context
loss + deep self-supervision (V-JEPA 2.1). So Arm 2a is a clean diagnostic for
*attribution* (no head-capacity confound) but **not guaranteed lossless** — for
this checkpoint it is empirically lossy on fine local-geometry content.

**This likely doesn't hit the three Tier-1 targets evenly.** `v` is probably
safe (JEPA's predictive task structurally needs motion in context tokens to
predict future masked content — plausibly why even vanilla V-JEPA 2 keeps decent
SSv2/motion numbers despite the dense-feature collapse). `ω` is intermediate.
**CoM-offset is the most exposed** — it needs fine static-asymmetry detail, the
same class of signal (depth/normals/seg) the 2.1 paper shows is washed out of
vanilla V-JEPA 2's context tokens specifically.

**Consequence for GATE B:** a null Rung-0 result on CoM-offset is currently
ambiguous between "physics genuinely unrecoverable" and "known vanilla-V-JEPA-2
register-collapse, already fixed upstream." [gate-b-probe.md](gate-b-probe.md)'s
brutal-honesty mandate can't rule this out yet because the probe only runs
against vanilla V-JEPA 2. Two mitigations, cheapest first:
1. **Label-free diagnostic.** Per layer, cosine-similarity an object's
   masked-pooled feature against the whole-frame-pooled feature. Near-identical
   ⇒ register-collapse signature at that depth, checkable before spending any
   compute on the physics regression.
2. **Re-run Arm 2a against V-JEPA 2.1 weights**, same harness, swap checkpoint —
   if CoM-lift recovers under 2.1 but not vanilla 2, that confirms the confound
   rather than "no physics," and is a direct argument for adopting 2.1 as GATE
   B's frozen encoder.

**Also worth reconciling, lower stakes:**
- `encode_pooled` (`encoders.py`) masked-weighted-means over **both space and
  time** into one `(n_layers, D)` vector per body — this discards exactly the
  `(t,y,x)` addressability this note's mental model is built on. For a
  time-varying target (CoM during a tipping/contact event), whole-window pooling
  smears the dynamics signal R6's scenarios are designed to make visible. Not a
  bug — an implicit design choice worth deciding explicitly before Tier 2.
- `probe-walkthrough.md`'s diagram (`16 temporal × 24×24`, i.e. T'=16 → 32 raw
  frames) vs. BRAINSTORM/training-data.md's committed K=4–16 frames — confirm
  what `dataset.py` actually feeds the encoder for the live corpus.
- `encoders.py` (`num_frames=32`) vs. `surprise.py` (`num_frames=64`) seed the
  RoPE grid differently for the same model family. Harmless (`use_rope`
  re-derives T from the real input), but worth a one-line note so it doesn't
  read as load-bearing.

## Links
- Dense-readout plan / frozen-first: [gate-b-probe.md](gate-b-probe.md)
- Encoder wrapper that emits the per-token taps: [../../src/probe/encoders.py](../../src/probe/encoders.py)
- Mask-index convention that depends on the flatten order: [../../src/intphys/surprise.py](../../src/intphys/surprise.py)
- V-JEPA 2.1 paper notes (register-token diagnosis): [../../summaries/v-jepa2-1.md](../../summaries/v-jepa2-1.md)
