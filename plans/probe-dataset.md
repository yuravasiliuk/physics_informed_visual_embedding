# Physics-Probe Benchmark — Data Spec & Generation Plan

**Working name:** `RigidProbe` (rename before any external release).
**First use:** V-JEPA 2 linear probe on per-object (ω, v, CoM).
**Long-term use:** pretraining substrate for the sim+real encoder (extends `synthetic-data-collection.md`).

This plan defines **what** we record and **how** we store it. It is deliberately spec-first: schema and format are nailed down before the generator is touched.

---

## 1. Goals

1. A versioned, **reproducibly-generated** rigid-body video benchmark with per-body physics ground truth.
2. Stored in a **compact, streamable** format so a 4090-class workstation can train on it without disk pressure.
3. **Schema-stable across versions** — v1 adds bodies/lighting/cameras but does not change field names or units.
4. Single source of truth for the encoder pretraining substrate. No second dataset (Kubric, CLEVRER) for the probe.

## 2. Non-goals (v0)

- Real-world counterpart frames — ingested via a separate plan ([[training-data]]).
- Articulated bodies, fluids, deformables.
- Robot arms or task scenarios.
- Multi-camera, ego-motion, time-of-day variation.
- Public release. v0 is internal; release-readiness checks happen at v1.

## 3. Schema — per clip

A **clip** = K consecutive frames of a single rigid-body scene from a fixed camera, generated from one seed.

### 3.1 Dense modalities

| Member | Shape | Dtype | Frame? | Notes |
|---|---|---|---|---|
| `rgb` | (K, H, W, 3) | uint8 | per frame | Stored as H.264 mp4 inside the shard |
| `depth` | (K, H, W) | float16 | per frame | Distance to camera, meters |
| `normals` | (K, H, W, 3) | int8 | per frame | Quantized: 127↔+1, -127↔-1. Unit vectors in camera frame |
| `seg` | (K, H, W) | uint8 | per frame | Instance IDs. 0 = background/ground. IDs match `state.body_id` |

### 3.2 Per-body state — (K, N, …) packed arrays

Stored as `.npz`, one array per field. N = number of bodies in the clip (constant per clip; varies across clips).

| Field | Shape | Dtype | Units | Notes |
|---|---|---|---|---|
| `body_id` | (N,) | uint8 | — | Maps to seg IDs |
| `com_camera` | (K, N, 3) | float32 | m | Center of mass in camera frame |
| `lin_vel_camera` | (K, N, 3) | float32 | m/s | Linear velocity in camera frame |
| `ang_vel_camera` | (K, N, 3) | float32 | rad/s | Angular velocity in camera frame |
| `quat_camera_wxyz` | (K, N, 4) | float32 | — | Body→camera orientation |
| `mass` | (N,) | float32 | kg | Per body, constant in time |

**Contact fields intentionally omitted in v0.** `in_contact` and `contact_pairs` are not collected. They are cheap to add back as additive fields when a contact-prediction head is committed; until then they bloat the schema without consumers.

**All vectors in camera frame.** Labels become invariant to scene placement, and the encoder predicts directly in the frame it sees. `R_camera_world` and `camera_position_world` go in metadata if anyone needs world-frame.

### 3.3 Per-clip metadata (JSON)

```json
{
  "schema_version": "rigid-probe/v0",
  "clip_id": "shard00042_clip0007",
  "seed": 42007,
  "K": 16, "H": 320, "W": 320,
  "render_fps": 30, "physics_hz": 60,
  "camera": {
    "position_world": [1.2, 1.2, 0.8],
    "look_at_world": [0.0, 0.0, 0.1],
    "intrinsics": [fx, fy, cx, cy],
    "R_camera_world": [[...], [...], [...]]
  },
  "bodies": [
    {"id": 1, "shape": "cuboid", "scale": [0.1, 0.1, 0.1], "color_rgb": [0.8, 0.2, 0.2], "friction": 0.5, "restitution": 0.3},
    ...
  ],
  "scene": {"gravity": [0, 0, -9.81], "hdri": "studio_small_03", "hdri_rotation_deg": 137.0}
}
```

## 4. Storage format

**WebDataset `.tar` shards.** One clip = one key, multiple members under that key.

```
shards/
├── rigid-probe-v0-train-000000.tar
├── rigid-probe-v0-train-000001.tar
├── ...
└── rigid-probe-v0-val-000000.tar
```

Inside a shard:

```
0042_0000.rgb.mp4
0042_0000.depth.f16.zst
0042_0000.normals.i8.zst
0042_0000.seg.u8.zst
0042_0000.state.npz
0042_0000.meta.json
0042_0001.rgb.mp4
...
```

- **1000 clips per shard.** Target ~1.5 GB per shard so a shard fits comfortably in RAM.
- **`.f16.zst` / `.i8.zst` / `.u8.zst`** = raw little-endian numpy buffer, zstd-compressed (level 3). Shape stored in the sidecar `meta.json` under `tensor_shapes`.
- **`.npz`** for per-body state — small, structured, fast to load.
- **`.mp4`** for RGB — H.264, CRF 20, yuv420p, ffmpeg pipe encoded at write time.

**Why WebDataset, not HDF5 or per-clip dirs.** WebDataset streams from disk/HTTP/S3 with no random-access cost, plays cleanly with PyTorch `DataLoader` sharding, survives partial shards, and is the format `training-data.md` already commits to.

## 5. Compression decisions (with rationale)

| Modality | Format | Lossy? | Why this and not other options |
|---|---|---|---|
| RGB | H.264 mp4, CRF 20 | mildly lossy | Matches V-JEPA 2's training distribution; 30–50× compression over raw. Avoid HEVC (decoder availability), AV1 (slow encode), PNG (3–5× worse). |
| Depth | float16 + zstd | lossless | Float16 ≈ <1 mm error at 10 m, well below sensor noise floor. PNG-encoded depth is lossy in ways that hurt regression gradients. |
| Normals | int8 (quantized) + zstd | mildly lossy | ~1° quantization error, well below what a head can predict. Cuts size 4× vs float16. |
| Seg | uint8 + zstd | lossless | Must be exact — masks bleed kills the pool-mask trick. ≤255 bodies per clip enforced. |
| State | float32 in npz | lossless | Already tiny (~10 KB/clip); no point compressing further. |

## 6. Per-clip size budget

For K=32, H=W=320, ~5 bodies:

| Component | Raw | Compressed | Ratio |
|---|---|---|---|
| RGB | 9.8 MB | ~300 KB | 30× (mp4 CRF 20) |
| Depth float16 | 6.5 MB | ~1.2 MB | 5–6× (zstd) |
| Normals int8 | 9.8 MB | ~1.2 MB | 8× (zstd) |
| Seg uint8 | 3.3 MB | ~160 KB | 20× (zstd) |
| State | ~20 KB | ~20 KB | 1× |
| Meta | ~2 KB | ~2 KB | 1× |
| **Total / clip** | **~29.4 MB** | **~2.9 MB** | **~10×** |

| Corpus | Clips | Compressed size |
|---|---|---|
| v0 train | 2,000 | ~5.8 GB |
| v0 val/test | 400 | ~1.2 GB |
| **v0 total** | **2,400** | **~7.0 GB** |
| v1 train (target) | 50,000 | ~145 GB |

v0 fits on the local SSD with room to spare. v1 will want a dedicated drive. Shard size at 1000 clips/shard ≈ 2.9 GB/shard — fits RAM on the workstation.

## 7. Generation parameters — v0

| Parameter | Value | Rationale |
|---|---|---|
| Object count per scene | 3–8, uniform | Enough variety; segmentation IDs stay ≤8 |
| Object library | Cuboid, sphere, cylinder, capsule | Mass + inertia closed-form; defer YCB/GSO to v1 |
| Object scale | 0.05–0.15 m per axis | Smallish, fit ground patch |
| Material color | Random hue, sat ∈ [0.5, 1.0], val ∈ [0.4, 0.9] | Photometric variation without HDRI complexity |
| Friction | uniform [0.3, 0.8] | Plausible range |
| Restitution | uniform [0.0, 0.4] | Avoid super-bouncy chaos |
| Initial pose | Rejection-sample above ground, no overlap | Trivial 16-frame drops are not learnable for ω |
| Initial linear vel | 0 | Drops only; v1 adds throws |
| Initial angular vel | uniform [-2, 2] rad/s per axis | Enough rotation to make ω learnable from 16 frames at 30 fps |
| Gravity | [0, 0, -9.81] | Fixed |
| Camera | Fixed at (1.2, 1.2, 0.8) looking at origin | v0 only; randomized in v1 |
| Lighting | One HDRI, random rotation only | v0 only; intensity/HDRI swap in v1 |
| K (frames) | 32 | ~1 s of motion at 30 fps; gives ω estimation room and matches V-JEPA 2 ViT-L/G stride budgets |
| Render rate | 30 fps | Physics at 60 Hz, render every other step |
| Resolution | 320 × 320 | Allows 256-crop augmentation; square avoids aspect bookkeeping |

## 8. Splits

- **Deterministic by seed**: `seed = shard_index * 1000 + clip_index_in_shard`.
- Train / val / test = **80 / 10 / 10**, by clip ID.
- v0 has no OOD split. v1 adds: held-out object instances, held-out lighting, held-out camera poses.

## 9. Sanity checks (per clip, pre-write)

A clip is **rejected and re-seeded** if any of:
- Fewer than K frames captured (replicator drop).
- Any NaN or Inf in state.
- `seg` contains zero non-background pixels.
- For all bodies: peak |v| < 0.01 m/s AND peak |ω| < 0.1 rad/s (nothing moved → no signal).
- RGB per-pixel std < threshold (black-frame bug).
- Number of distinct seg IDs ≠ len(bodies) + 1.

Rejection log (clip_id, seed, reason) written to `shard_metadata.json` for forensics.

## 10. Versioning

| Version | Adds | Target clip count |
|---|---|---|
| **v0** | Spec in this doc. Primitives, fixed camera, fixed lighting. | 2,400 |
| v1 | YCB/GSO object library, camera randomization, lighting variation, initial linear vel | 20k–50k |
| v2 | Textured grounds, distractor objects, multi-camera | 100k+ |
| v∞ | Articulated bodies, deformables, robot arms | future |

**Schema invariant**: field names, units, frames, dtypes do not change across versions. New fields are additive. Old code reads new shards correctly.

## 11. Implementation order

1. **Refactor `collect_smoke.py` → `collect_clip.py`.** Pull scene-building, capture, state-logging into pure functions. Same output as today; no behavior change.
2. **Add randomization knobs** (object count, scale, color, friction, restitution, initial ω, pose rejection).
3. **Switch state logging to camera frame for all velocity vectors** (currently only CoM is in camera frame).
4. **Add quantization + compression encoders** (depth → f16.zst, normals → i8.zst, seg → u8.zst, rgb → mp4 via ffmpeg pipe).
5. **Add sanity-check pass.** Returns pass/fail + reason; loop in driver until pass or seed-budget exhausted.
6. **Add WebDataset shard writer** (`webdataset.ShardWriter`, one tar per 1000 clips).
7. **Driver `collect_shard.py`**: argparse over (shard_id, n_clips, output_dir), generate one shard end-to-end.
8. **Generate v0** (3 shards: 2000 train + 200 val + 200 test). Verify on-disk size matches budget, round-trip decode.
9. **Reader `probe_dataset.py`**: `webdataset.WebDataset` → decode → return dict per clip. Torch `IterableDataset`-compatible.
10. **Hand to probe pipeline.** (Out of scope for this plan.)

## 12. Acceptance criteria for v0

The v0 corpus is "done" when:
- 2,400 clips written across 3+ shards, all sanity-check passing.
- Round-trip decode of 10 random clips matches generation-time tensors within compression tolerance (rgb PSNR > 35 dB, normals angular error < 2°, depth and seg exact).
- A torch `DataLoader` can stream the corpus at ≥200 clips/sec on the workstation (sustained, not first-shard-warm).
- Total disk footprint < 4 GB.

## 13. Settled decisions

| Choice | Decision |
|---|---|
| Resolution | 320×320 |
| Clip length K | 32 frames (~1 s at 30 fps) |
| Object library v0 | Primitives only (cuboid, sphere, cylinder, capsule) |
| Working name | `RigidProbe` |
| Cameras per scene | 1, fixed |
| Contact fields | Omitted in v0; additive in a later version |
| Velocity frame | Camera frame only. World-frame velocities not stored — recoverable from `R_camera_world` in metadata if needed. Contract change from `collect_smoke.py`. |

---

## Relation to existing plans

- **[[synthetic-data-collection]]**: this plan is the v0 concretization of its data-generation milestone. The broader sim+real mixing strategy lives there.
- **[[training-data]]**: defines the *training* corpus mixture and supervision policy. This plan defines the *sim half's container format*. Same WebDataset choice.
- **[[domain-marginalization-regularizer]]**: consumes this corpus in B and F configs.
- **[[critique2]]**: motivated the probe-first sequencing that this dataset is sized for.
