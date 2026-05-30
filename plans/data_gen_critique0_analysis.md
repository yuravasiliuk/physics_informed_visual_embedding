# Data-Gen Critique 0 — Motivation Analysis

Companion to [data_gen_critique0.md](data_gen_critique0.md). For each of the 13
points: *what the reviewer is actually worried about*, *the mechanism behind it*,
*whether the worry holds in our setting*, and *what we can/can't do about it*.

**No code is changed by this document.** It is decision support, written before
touching the pipeline.

---

## 0. The disk reality the critique under-states

The critique treats disk as a per-issue footnote (#6 seg, ~1 TB). The measured
numbers say disk is the *dominant* constraint and reframes several points.

Measured, per clip (32 frames @ 1024×1024), from `~/data/sim_physics`:

| Channel        | dtype / format    | per clip | × 10k clips |
|----------------|-------------------|----------|-------------|
| `normals.npy`  | float16 (T,H,W,3) | **193 MB** | **1.93 TB** |
| `seg.npy`      | uint32 (T,H,W)    | **129 MB** | **1.29 TB** |
| `depth.npy`    | float16 (T,H,W)   | **65 MB**  | **0.65 TB** |
| `rgb.mp4`      | mp4v              | **0.6 MB** | **6 GB**    |
| state/meta     | json              | ~0.2 MB  | ~2 GB       |
| **total**      |                   | **~388 MB** | **~3.9 TB** |

Available disk: **255 GB free** of 916 GB (71 % used). At the current layout the
machine holds **~650 clips**, not 10k. *Every* large-scale plan in `plans/` is
disk-infeasible as written. This is the headline finding, and the critique
misses it because it never adds the channels together or checks `df`.

Two consequences:

1. **`normals.npy` (193 MB, 1.93 TB projected) is the single biggest consumer
   and the critique never mentions it.** It is 50 % larger than the uint32 seg
   waste the critique flags as "High". Any disk conversation that starts with
   RGB or seg is optimizing the wrong term.
2. **The lossy-RGB worry (#1) and the disk worry are in direct tension, and the
   resolution choice dominates both** (see §1 and §14).

---

## Critical issues

### #1 — RGB saved only as lossy mp4v

**What the reviewer fears:** mp4v (MPEG-4 Part 2) is an old, block-DCT,
4:2:0-chroma-subsampled codec. cv2's `VideoWriter` gives no bitrate/CRF control,
so quality is whatever the default rate-control picks. The fear is that
codec noise on the *encoder input* (RGB) corrupts linear-probe R² while the
*labels* (depth/normals/seg/state) come from exact simulation — so you measure
"encoder + codec" and blame the encoder. Worst case: weeks of GPU spent
concluding "V-JEPA 2 doesn't encode velocity" when the codec smeared the cue.

**Does the worry hold here?** Partially, and weaker than stated:

- **Downsampling launders most of the artifact.** V-JEPA 2's default checkpoint
  (`vjepa2-vith-fpc64-256`) ingests 256×256; the higher-res ViT-g variant
  (`vjepa2-vitg-fpc64-384`) is 384×384. Our frames are 1024. A ≥4× area
  downsample is a low-pass filter that
  removes exactly the high-frequency 8×8 block/ringing artifacts mp4v produces.
  Most codec noise never reaches the encoder.
- **Probe targets are low-spatial-frequency.** CoM position, linear/angular
  velocity, mass — these are global motion/geometry attributes. Inter-frame
  video coding preserves smooth rigid motion well; it sheds high-frequency
  *texture* detail, which we don't have much of (primitives, flat colors).
- **The real residual risk is chroma, not luma.** 4:2:0 subsampling halves
  color resolution. If a probe target depends on *color/material* (and one of
  the directions does — "does the encoder represent material?"), that signal is
  the one mp4v actually degrades. Worth isolating.
- **`cv2` does an `RGB2BGR`→YUV round-trip** with uncontrolled quantization. The
  artifact is real but, on flat synthetic scenes at low bitrate, mostly mild
  banding on gradients.

**The user's two objections, assessed:**

- *"Data is from primitives in simulation."* This cuts **both ways and is
  mostly a point in favor of lossless**, not against caring. Simple flat-shaded
  scenes (a) produce few codec artifacts *and* (b) compress losslessly extremely
  well — large constant regions are exactly what PNG/FFV1/zlib exploit. So the
  usual reason to accept lossy ("lossless is too big") is *weakest* precisely
  here. Lossless of a primitive scene is far below the 96 MB raw upper bound.
- *"Limited disk space."* True and binding — but see §0: `rgb.mp4` is 0.6 MB,
  the smallest channel by 100×. RGB is **not** where the disk problem lives.
  Storing RGB losslessly costs at most ~96 MB raw (uint8, 1024²), and far less
  compressed; meanwhile `normals.npy` is already eating 193 MB losslessly with
  no complaint. There is an **internal inconsistency** in the current pipeline:
  it preserves *normals* bit-exact at 193 MB but throws away the *primary
  encoder input* down to 0.6 MB. One of those two decisions is wrong.

**What we can do:**

- **Lossless that is cheap on synthetic scenes:** PNG sequence, FFV1, or H.264/5
  `-crf 0`. On flat primitive scenes expect roughly 5–30 MB/clip, not 96 MB.
- **`rgb.npy` uint8** — dead simple, 96 MB/clip raw (1024²) but `np.savez_compressed`
  brings it down a lot for flat scenes. No codec at all → zero ambiguity.
- **Resolution is the real knob (§14).** If RGB is stored at the encoder's native
  256×256, raw uint8 is **6 MB/clip** — lossless RGB becomes a non-issue and the
  whole debate dissolves. The lossy-vs-lossless argument is downstream of the
  resolution decision.
- **Keep both:** small mp4 for human scrubbing + a lossless source for training.

**What we can't do:** store raw 1024² RGB for 10k clips *and* everything else —
that path is part of the 3.9 TB that doesn't fit. Lossless RGB only works hand-
in-hand with the resolution / channel-precision decisions in §14.

**Verdict:** the reviewer is right that lossy-on-the-input is a real confound for
a probe study, and "it's primitives" is an argument for cheap losslessness, not
for shrugging. But the magnitude is smaller than billed (downsampling + simple
scenes), and the fix should be decided *jointly* with storage resolution, not as
an isolated "switch to lossless" that would blow the disk budget.

---

### #2 — Initial velocities not persisted

**Motivation:** `build_scene` returns `initial_velocities`, `simulate_clip`
applies them after `world.reset()`, then they vanish. `meta.json` records only
the sampling *range* (`scenario_specs`), not the realized draw. Frame 0 of
`state.json` is post-step (`v0 + g·dt` plus any contact impulse), so you cannot
cleanly recover the launch condition.

**Does it hold?** Yes, unambiguously. This is a cheap, correct catch. Two
distinct costs: (a) **reproducibility** — a clip is not reconstructable from
`meta.json` alone; (b) **labels** — any "predict initial condition" target is
impossible, and even "what did this start from?" debugging is blocked.

**Can/can't:** trivially fixable (store per-body `init_lin_vel_world` /
`init_ang_vel_world` in `body_specs`). No disk concern (a few floats). The only
judgment call is whether you *want* initial conditions as a target at all; even
if not, persisting them costs nothing and restores reproducibility.

---

### #3 — No retry on validation failure

**Motivation:** `main()` runs `collect_clip` once per seed and writes whatever
comes out, including `validation.ok=False` clips. With 7 scenarios and random
spawns, the reviewer estimates 5–20 % failures (`all_bodies_stationary`,
`empty_seg`, `rgb_low_variance`, …). Each failure burns an Isaac iteration and
writes a useless ~388 MB directory; downstream loaders must remember to filter.

**Does it hold?** The mechanism is real; the 5–20 % is an unverified guess. The
*disk* angle matters more than the compute angle given §0: a failed clip is
388 MB of garbage. The "must filter downstream" point is a genuine silent-
footgun.

**Can/can't:** easy to wrap in a bounded re-seed loop and to **not write failed
clips** (or route to `failed/`). One nuance the critique gets right: re-seed with
a large offset, not `seed+1`, so retries are independent draws. Defer-able for a
small first batch, required before any large run — but cheap enough to just do.

---

## High-priority issues

### #4 — Verify script only covers the trivial case

**Motivation:** `verify_collect_clip.py` checks a single sphere, ω=0, free
flight, spherical geometry (`local_com==0`). That exercises *none* of the risky
code paths: angular-velocity sign (`R_camera_world @ ang_vel_world`), quaternion
composition (`quat_wxyz_mul`), the `R_body_world @ com_local` asymmetric-CoM path
(dead code under a sphere), multi-body index alignment, and the *production*
camera pose (verify uses a different broadside camera).

**Does it hold?** This is the most important point in the document. The failure
mode is the worst kind: a sign flip in ω or a body-index swap passes verify
silently, you train probes for weeks, and you get random R² on angular velocity
with no error anywhere. The labels are the one thing you cannot sanity-check by
eyeballing — they *are* the ground truth. A narrow verify gives false
confidence precisely where confidence is load-bearing.

**Can/can't:** all proposed cases are constructible (spinning cuboid with known
constant ω; asymmetric-CoM body with analytic world→camera CoM; two-body
swap check; run at production camera pose). Medium effort, highest value-per-
hour in the list. Should precede any GPU spend.

### #5 — `seg_info` taken from frame 0 only

**Motivation:** `save_clip` reads `frames[0]["seg_info"]` for the id→class map.
If a body is occluded in frame 0 and Replicator reports per-frame *visible*
semantics (version-dependent), that body's id is missing from the map →
downstream per-body pooling mis-routes its pixels to "unknown" for some frames.

**Does it hold?** Conditionally — depends on Replicator's reporting semantics,
which the reviewer correctly flags as version-dependent and untested. Even if
current behavior reports all stage semantics, relying on it unstated is fragile.

**Can/can't:** merging `seg_info` across all frames is a 3-line, last-write-wins
fix (ids are stable) plus an empirical check that ids don't drift. Cheap
insurance; no downside.

### #6 — Seg stored as uint32

**Motivation:** ≤~10 semantic classes but stored uint32 = 129 MB/clip,
1.29 TB/10k. uint8 (≤255 ids) → 32 MB/clip, ~0.97 TB saved.

**Does it hold?** Yes, arithmetic is correct and verified against measured
129 MB. Trivial, safe (assert `max<256`, else uint16). **But** per §0 it's the
*second* biggest channel, not the first — fixing seg without also addressing
normals (193 MB) leaves the larger leak open. Do both.

---

## Medium-priority issues

### #7 — No mass/density randomization

**Motivation:** all primitives use default 1000 kg/m³, so mass = density ×
volume and volume is visually given by `shape`+`dims`. A "predict mass" probe
then just learns visible volume — it never tests whether the encoder represents
mass as a property *independent* of size. Real objects break this (steel vs.
ping-pong ball).

**Does it hold?** Yes, and it's the deepest *scientific* point in the document,
not a bug. It directly determines whether the dataset can answer a headline
research question (Direction A: "does V-JEPA 2 encode mass independent of
size?"). Without density variation, that probe is confounded and any positive
result is uninterpretable.

**Can/can't:** straightforward to add (`density_range` → `UsdPhysics.MassAPI` or
`set_mass`, log per body). The real question is a *design* one: is mass-as-
independent-quantity a target? If yes, this is mandatory and should arguably be
re-tiered to Critical. If no, skip. It interacts with #2/#8/#9 (all touch
`body_specs`). Note: mass affects *dynamics*, so you can't retrofit it onto
existing clips — it must be decided before generation.

### #8 — Mass logged every frame

**Motivation:** mass is constant but written per body per frame (8×32=256
redundant floats/clip, ~7 % of `state.json`). Belongs in `body_specs` once.

**Does it hold?** Yes but minor. `state.json` is ~0.2 MB, so the absolute disk
cost is ~2 GB/10k — negligible next to the TB-scale array channels. This is
cleanliness, not a disk lever. Worth doing only as a ride-along with #9/#2.

### #9 — Mass missing from `body_specs`

**Motivation:** forces a full `state.json` parse to get a per-body constant.
Pure ergonomics. Same fix as #8 (write mass once into `body_specs`). Real but
small.

### #10 — Frame 0 is post-step, not t=0

**Motivation:** the loop steps *then* captures, so frame 0 = state at
`t=render_dt` with gravity (and possibly contact) already applied. Verify
accounts for it; training code might wrongly assume `state[0]` = initial
conditions.

**Does it hold?** It's a documentation issue, explicitly "by design and
internally consistent" (RGB and state are co-stepped). For *current-frame*
physics probing it's harmless. Only matters if combined with #2 (initial-
condition targets). Fix = one sentence in the schema doc, or an optional true
t=0 capture before the loop.

---

## Design observations (watch-items)

### #11 — Replicator camera vs. hand-coded `world_to_camera_frame`

The hand-rolled look-at must match Replicator's internal extrinsics exactly
(both USD/OpenGL: +X right, +Y up, −Z forward). The verify ballistic roundtrip
is the regression test; risk surfaces only on an Isaac version bump. **Real but
latent** — the mitigation (run verify on every version bump) is correct and
free. Tighten by extending verify (#4) to the production camera.

### #12 — `world_to_camera_frame` degenerates top-down

If `forward ∥ world_up`, `cross(forward, world_up)=0` → `right/=norm` yields NaN
→ poisons every state record; only caught post-hoc by `nonfinite_state` after a
full clip's compute. Current cameras are safe; the trap springs the day someone
adds a top-down scenario. **Cheap defensive assert** (`|cross|>eps`) is pure
upside. Low effort, prevents a silent class of waste.

### #13 — Replicator RNG not seeded

`default_rng(seed)` covers scene sampling but not Replicator's RT/denoiser RNG,
so clips aren't bit-exact reproducible. **Correct and essentially unfixable**
through the public API. Fine for training data; only hurts "reproduce this one
interesting clip" debugging. Accept and document. (Note this also slightly
undercuts #2's "full reproducibility" claim — even with velocities persisted,
pixels aren't bit-exact.)

---

## §14 — What the critique missed (and the decision that dominates it)

1. **`normals.npy` is the biggest single channel (193 MB, 1.93 TB/10k) and is
   unmentioned.** Question worth asking before any storage fix: *do we even need
   per-pixel normals stored at full res?* If normals are a probe *target*,
   probes operate at token resolution (256/patch ≈ 16×16), so storing 1024²
   float16 normals is ~4000× over-resolved relative to what a probe consumes.
   Same argument applies to depth and seg.

2. **Label resolution ≠ render resolution.** RGB (encoder input) may want higher
   res for render quality; depth/normals/seg/state (probe *labels*) only need to
   exist at the resolution the probe reads them at. Decoupling these two is
   probably the single biggest disk win available and the critique never raises
   it.

3. **Render/store resolution is the master knob.** Per-pixel cost scales with
   area. Dropping 1024→256 is ÷16 on *every* array channel:

   | Channel (256², 10k) | size |
   |---------------------|------|
   | normals float16     | ~120 GB |
   | seg uint8           | ~20 GB |
   | depth float16       | ~40 GB |
   | rgb raw uint8       | ~60 GB |
   | **total**           | **~240 GB** (just fits in 255 GB) |

   At 256², lossless RGB (#1) costs 6 MB/clip and the entire RGB debate
   evaporates. So #1 should be decided *jointly* with resolution, not in
   isolation — switching RGB to lossless at 1024 without addressing the other
   channels would push us further past the disk ceiling, not toward a usable
   dataset.

   **Caveat — which V-JEPA 2 checkpoint?** 256² assumes the default
   `vjepa2-vith-fpc64-256`. The higher-res `vjepa2-vitg-fpc64-384` variant is
   384², which is **2.25× the per-pixel cost** → the stack above becomes
   ~540 GB and *does not fit*. If 384 is in scope, either (a) store labels at
   256 and accept a re-render if you later move to 384, or (b) pick a single
   target resolution now. This is a decision to make before generating, not
   after.

4. **Decisions that must precede generation vs. can be retrofitted.** Density
   (#7) and resolution change *dynamics/pixels* and cannot be retrofitted —
   decide first. Storage format/precision (#1, #6, normals), retry (#3),
   metadata layout (#2, #8, #9), verify (#4), seg-merge (#5) are all
   regenerate-or-rewrite and lower-stakes to defer.

---

## Suggested decision order (no code yet)

1. **Resolution + which channels at which resolution** (§14) — gates the disk
   budget and silently resolves #1. Biggest lever; decide first.
2. **#4 verify extension** — before any GPU spend; labels must be trusted.
3. **#7 density** — research-question gate; must precede generation if "mass ⟂
   size" is a target.
4. **#2 initial velocities** — free, restores reproducibility.
5. **#6 seg uint8 + normals precision/res** — the actual TB-scale disk wins.
6. **#5 seg-merge, #12 degenerate-camera assert** — cheap insurance.
7. **#3 retry** — before the large run.
8. **#8/#9/#10** — ride-along cleanup.
9. **#11/#13** — document and watch.
