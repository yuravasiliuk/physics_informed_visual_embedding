# Physics-LoRA — implementation notes (V-JEPA hook points + code skeletons)

**Companion to [physics-lora-adapter.md](physics-lora-adapter.md)** (the *why*;
this file is the *where/how*). Status: notes + skeletons, nothing built.
Everything below was verified against the actual checkouts on 2026-07-19.

**Ground rule:** the `~/vis_emb/vjepa2` checkout is **read-only substrate**
(same spirit as the Isaac rule). All LoRA/adapter code lives in *this* repo as
new files (CLAUDE.md §Principles — backward compatibility), wrapping the
checkout's modules via import, exactly as [encoders.py](../../src/probe/encoders.py)
already does.

---

## 0. Glossary — `cut` vs `tap` (they are independent)

Figure: [`results/figures/lora_topology.pdf`](../../results/figures/lora_topology.pdf),
regenerate with `python3 scripts/viz_lora_topology.py`.

- **cut** — the boundary where *training* begins. Every block from the cut to the
  deepest tap carries adapters, **contiguously**. `cut=24, taps 28/30/32` ⇒ blocks
  24…32 are ALL adapted — **29 and 31 included**; they are simply not read.
- **tap** — a passive *read* of a block's output, through the frozen shared final
  LayerNorm (mirrors `vision_transformer.py:205`). A tap alters nothing: `x_out`
  continues into the next block unchanged. Blocks 28/30/32 run **exactly the same
  operation** as 29 — the only difference is that somebody takes a copy.
- **Design rule: leave adapted depth between the cut and the first tap.** With
  `cut=28, taps 28/30/32` the L28 tap sees one adapted block and is effectively
  frozen — and its mix weight becomes uninterpretable ("physics lives at 28" is
  indistinguishable from "28 is barely adapted"). Prefer `cut=24, taps 28/30/32`
  (keeps taps inside the GATE-B band); `cut=28` is defensible only with taps
  moved deeper (32/36/40), which bets that adaptation relocates the signal.
- **Truncate at the deepest tap.** Blocks above it feed nothing in the physics
  path: their adapters would get zero gradient (dead weights) and running them is
  wasted compute. V0 therefore adapts **blocks 24–32 = 9 blocks**.
- **Two forward configurations coexist** (they are not in conflict):

  | Path | Blocks run | Why |
  |---|---|---|
  | physics training | 24 → 32 (cache below) | deepest tap is 32 |
  | eval / canary / downstream | 0 → 39 + final norm + predictor | IntPhys2 reads final-depth features |

  The frozen tail still *propagates* the adapter's effect to the final layer —
  truncation is a training-time optimization, not a firewall. That propagation is
  exactly what the canary measures, and why §3.3 of the design doc adds a
  full-depth anchor.
- Exception: the V1 variant deliberately keeps a **final-block control tap**, which
  gives 33–39 a gradient path and makes their adapters live.
- Indexing: blocks are 0-based, so the last block is **index 39**; "L40" in probe
  notation means the 40th block = index 39.

## 1. Substrate map — where things are in the V-JEPA codebase

| What | Where (in `~/vis_emb/vjepa2`) | Key fact |
|---|---|---|
| Model constructor | `src/models/vision_transformer.py:435` `vit_giant_xformers_rope` | D=1408, depth=40, heads=22, `qkv_bias=True`, `use_rope=True` |
| Forward + tap hook | `src/models/vision_transformer.py:161-213` | `model.out_layers = [...]` → returns `self.norm(x)` **per tapped block** (final LayerNorm applied at every tap) |
| Block | `src/models/utils/modules.py:505` `Block` | pre-norm: `x + attn(norm1(x))`, `x + mlp(norm2(x))` |
| Attention | `src/models/utils/modules.py:266` `RoPEAttention` | **fused** `self.qkv = nn.Linear(1408, 3·1408)` (`:283`), `self.proj = nn.Linear(1408,1408)` (`:285`) |
| MLP | `src/models/utils/modules.py:72` `MLP` | plain `fc1/fc2` (GELU — giant does *not* use SwiGLU; hidden = 1408·48/11 = 6144) |
| Patch embed | `src/models/utils/patch_embed.py` `PatchEmbed3D` | tubelet 2 × patch 16 → token `(t,y,x)` row-major; 384² → 24×24/frame-pair |
| Our loader (reuse it) | [src/probe/encoders.py](../../src/probe/encoders.py) `VJEPA2Encoder` | checkpoint `weights/vitg-384.pt`, load recipe (strip `module.`/`backbone.`, `strict=False`), eval transform, seg-mask→token-weight pooling |

Facts that shape the implementation:

- **Fused qkv** ⇒ "LoRA on Q,V only" cannot target separate `q`/`v` Linears
  (they don't exist). We adapt slices of the fused output: Q = rows `[0:D]`,
  V = rows `[2D:3D]`; K stays bit-frozen. (Also why HF PEFT is a poor fit here
  — plus the model isn't an HF `transformers` class at all. Hand-rolled ~60
  lines, below, is the recommendation.)
- **LoRA sits upstream of RoPE.** Rotary embeddings are applied to q,k *after*
  the qkv projection (`modules.py:335ff`), so adapting `qkv` doesn't interact
  with position encoding. **RoPE is per-block, not an input-stage embedding:**
  with `use_rope=True` the forward *skips* the additive positional embedding
  entirely (`vision_transformer.py:183-188` adds `pos_embed` only in the
  `not use_rope` branch), so position enters ViT-g **only** via
  `RoPEAttention.forward` at all 40 blocks. Consequences: ΔQ from LoRA is rotated
  exactly like base Q (no special handling); cached lower-stack activations
  strand nothing positional; and `T/H_patches/W_patches` must be passed into every
  manually-run block.
- **Blocks are runnable standalone** but need `T, H_patches, W_patches` kwargs 
  (RoPE grid). `handle_nonsquare_inputs=True` by default, so the real values
  flow in normal forwards; our upper-stack must pass them explicitly.
- **Tap semantics:** probe features = `model.norm(block_i_output)`. Training
  taps must apply the same (frozen) final LayerNorm or the Gate-B baselines
  aren't comparable. Layer indices are **0-based block indices** — identical to
  the probe layer axis in `results/probe/*` (L28/30/32 = `blocks[28|30|32]`).
- Token count: 32-frame clip @384 → `16·24·24 = 9216` tokens
  ([vjepa-embedding-structure.md](vjepa-embedding-structure.md)); 64 frames →
  18432. `encoders.py:141` hard-asserts this factorization.

## 2. LoRA classes (skeletons)

### 2.1 Q,V-only adapter for the fused qkv

```python
class LoRAQKV(nn.Module):
    """Q,V-only LoRA around the fused qkv Linear (modules.py:283).

    Update writes only into the Q [0:D] and V [2D:3D] output slices; K is
    bit-frozen. Base Linear already has requires_grad=False (encoders.py:77)
    — the new A/B Parameters default to trainable, which is exactly right.
    """
    def __init__(self, qkv: nn.Linear, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base, self.D, self.scale = qkv, qkv.in_features, alpha / r
        D = self.D
        self.A_q = nn.Parameter(torch.empty(r, D))
        self.B_q = nn.Parameter(torch.zeros(D, r))   # B=0 ⇒ ΔW=0 at init:
        self.A_v = nn.Parameter(torch.empty(r, D))   # step 0 is exactly the
        self.B_v = nn.Parameter(torch.zeros(D, r))   # frozen encoder
        for A in (self.A_q, self.A_v):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)                              # (..., 3D)
        dq = (x @ self.A_q.T) @ self.B_q.T              # (..., D)
        dv = (x @ self.A_v.T) @ self.B_v.T
        zeros_k = dq.new_zeros(dq.shape)                # K slice untouched
        return out + self.scale * torch.cat([dq, zeros_k, dv], dim=-1)
```

### 2.2 Installation + (un)merge

```python
def install_lora(model, cut=24, r=16, alpha=16.0):
    """model = VJEPA2Encoder(...).model — adapt blocks[cut:] in place."""
    for blk in model.blocks[cut:]:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r=r, alpha=alpha)
    return [p for p in model.parameters() if p.requires_grad]  # optimizer set

def merge_lora(model):
    """Fold ΔW into the base weights → plain ViT, zero inference overhead.
    qkv.weight rows: Q=[0:D], K=[D:2D], V=[2D:3D]."""
    for blk in model.blocks:
        m = blk.attn.qkv
        if isinstance(m, LoRAQKV):
            with torch.no_grad():
                m.base.weight[: m.D] += m.scale * (m.B_q @ m.A_q)
                m.base.weight[2 * m.D :] += m.scale * (m.B_v @ m.A_v)
            blk.attn.qkv = m.base
```

α-scaling at inference (the frozen↔steered interpolation knob from the design
doc §1) = multiply `self.scale` before merging, or keep adapters unmerged and
scale their output.

Optional later targets (design doc grid): `blk.attn.proj`, `blk.mlp.fc1/.fc2`
— all plain `nn.Linear`, wrappable by a generic single-matrix `LoRALinear`
(same pattern minus the slice bookkeeping).

## 3. Split forward: cached lower stack + trainable upper stack

### 3.1 Lower stack (frozen, no-grad — run once per clip or per batch)

```python
@torch.no_grad()
def encode_lower(model, x, cut=24):
    """Mirror VisionTransformer.forward (vision_transformer.py:161-203)
    up to blocks[cut]. Returns the RAW residual stream (pre-norm) + RoPE grid.
    use_rope path ⇒ no absolute pos-embed is added (only patch_embed)."""
    _, _, T, H, W = x.shape
    T, Hp, Wp = T // model.tubelet_size, H // model.patch_size, W // model.patch_size
    x = model.patch_embed(x)
    for blk in model.blocks[:cut]:
        x = blk(x, mask=None, attn_mask=None, T=T, H_patches=Hp, W_patches=Wp)
    return x, (T, Hp, Wp)
```

**Cache vs. recompute** (decide per disk budget, R10): cached fp16 stream is
`N × 1408 × 2 B` ≈ **26 MB/clip** (32 fr) / 52 MB (64 fr) — ~78 GB for a 3k-clip
corpus. Recompute-on-the-fly costs ~60 % of a full forward per step but zero
disk and no backprop below the cut either way. Start with recompute; cache the
(smaller) twin corpus if GPU-bound.

### 3.2 Upper stack (the trainable unit)

```python
class UpperStack(nn.Module):
    TAPS = (28, 30, 32)          # 0-based block idx == probe layer axis (§1)

    def __init__(self, model, cut=24):
        super().__init__()
        self.cut = cut
        # TRUNCATE at the deepest tap (§0): blocks above it have no gradient
        # path — running them is waste and their adapters would be dead weights.
        self.blocks = model.blocks[cut:max(self.TAPS) + 1]  # frozen base W + LoRA
        self.norm = model.norm             # frozen final LN — tap semantics!
        self.mix = nn.Parameter(torch.zeros(len(self.TAPS)))  # learned scalars

    def forward(self, x, T, Hp, Wp):
        taps = {}
        for i, blk in enumerate(self.blocks, start=self.cut):
            x = blk(x, mask=None, attn_mask=None, T=T, H_patches=Hp, W_patches=Wp)
            if i in self.TAPS:
                taps[i] = self.norm(x)     # == encoders.py tap (per-block norm)
        w = torch.softmax(self.mix, dim=0)
        vision_emb = sum(wi * taps[i] for wi, i in zip(w, self.TAPS))
        return vision_emb                  # (B, N, 1408) token field
```

`w` after training is the free diagnostic from the design discussion: drift of
the mix weights = the adapter relocating physics across depth.

### 3.3 Projection + pooling (topology per design doc §2.1)

```python
proj = nn.Sequential(nn.Linear(1408, 512), nn.GELU(), nn.Linear(512, 256))
z_phys = proj(vision_emb)                  # token-wise → (B, N, 256)
```

- Per-body pooling for pair/material heads: **import, don't copy**
  (`from src.probe.encoders import VJEPA2Encoder`) — reuse
  `_mask_to_token_weights` (encoders.py:92) so mask→token geometry stays
  bit-identical to the probe path.
- Real-anchor and depth/normals heads attach to `vision_emb`
  (pre-projection); twin-contrastive + physics heads attach to `z_phys` —
  routing rules in design doc §2.1, not repeated here.

## 4. Architectural proposals (variants to sweep)

| | Variant | One-liner | Extra trainables |
|---|---|---|---|
| **V0** | Fixed taps 28/30/32 + learned scalar mix → shared projection | The default drawn above; baseline for everything | 3 scalars |
| V1 | Wide candidate taps (24,26,…,38) + L40 control tap, learned mix | Lets adaptation *relocate* the peak; mix weights = where physics lives; L40 weight tests "top layers re-specialize away" | ~9 scalars |
| V2 | Per-tap projections, summed in z-space (`Σ proj_i(tap_i)`) | Each depth gets its own physics view; heavier, interpretability drops | ×n_taps proj params |
| V3 | Concat taps (3×1408) → projection mixes | Most expressive mix, opaque; subsumes V0 at 3× first-layer cost | 3× proj first layer |

Parameter counts @ D=1408, Q,V-only, **9 adapted blocks (24–32, truncated at the
deepest tap — §0)**: 90,112 params/block ⇒ r=4 → **0.20 M** · r=16 → **0.81 M** ·
r=64 → **3.2 M**; projection ≈ 0.85 M; total V0 @ r=16 ≈ **1.7 M trainable
≈ 0.13 %** of ViT-g. Train-time forward touches 9 of 40 blocks (~22 %) with 0–23
cached. (V1, which keeps a final-block control tap, adapts 16 blocks instead:
r=16 → 1.44 M.) The full sweep grid
(cut ∈ {22,26} × modules ∈ {QV, QV+MLP} × r ∈ {4,16,64}) is design doc §2.

## 5. Open questions (blocking none of the skeletons)

1. **Tap/mix mechanism** — V0 vs V1 first? (User has an idea on layer capture
   — not yet recorded here.)
2. **Cache vs recompute** for the lower stack (§3.1) — disk-budget call.
3. Optimizer hygiene for mixed parts (LoRA lr vs projection/head lr — LoRA
   typically tolerates ~10× the head lr; sweep later, note now).
4. fp16 training of adapters on top of the fp16 frozen stack (encoders.py:55)
   — likely fine with fp32 master weights (standard AMP); verify no-grad
   lower stack matches cached activations bit-wise before trusting the cache.
