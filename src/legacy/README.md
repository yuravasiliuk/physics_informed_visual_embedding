# src/legacy/ — retired code, kept readable

Policy (2026-07-27): **retire, don't delete.** Code whose question is settled
moves here instead of being removed, so the reasoning stays browsable without git
archaeology.

## The one invariant

**Nothing in `legacy/` may be imported by live code.** That is the definition of
retired — if something here is still imported, it is not legacy, it is a
dependency in the wrong folder. Check with:

```bash
grep -rn "from legacy\|import legacy" src/ scripts/ --include=*.py   # must be empty
```

There is deliberately no `__init__.py`: files here are read, not imported.

## Rules

- **Whole modules move.** Dead code *inside* a live module gets a `DEPRECATED:`
  marker with the reason and is dropped the next time that file is substantially
  edited — physically relocating a single function costs more than it saves.
- **Every entry gets a row below**, with what superseded it. Without that this
  becomes a junk drawer and the policy is worse than deleting.
- **A module with zero importers is not automatically legacy.** Tools that will
  be needed again (corpus acceptance, oracles) stay put.

## Contents

| module | retired | why | superseded by |
|---|---|---|---|
| `register_collapse.py` | 2026-07-27 | CoM-line diagnostic. Asked whether the CoM null came from register/CLS collapse in the frozen features — answered **no** (body↔frame cosine *falls* 0.96→0.56 with depth, `results/probe/register_collapse`). The question died with the CoM target (EXPERIMENTS.md Graveyard). | nothing — question closed |
| `verify_byte_split_depth.py` | 2026-07-27 | Verifier for the byte-split depth encoding, **KILLED** experiment: 2.4× *larger* than npz (25.6 MB) because the low byte is noise and won't compress. | `depth_map_factory.py` (npz / ffv1 / hevc12) |
| `collect_pairs.py` | 2026-07-28 | Controlled-intervention twin driver. Its **only** intervention is `com` (`base` vs `com<α>` re-authored CoM), so the whole module's question died with the CoM target. The twin *machinery* itself (same seed, one knob changed, frame-0-identical pixels) survives as a pattern — egomotion twins reuse it via `--camera_families` A/B, which needs no separate driver. | nothing — question closed; twin pattern lives on in the camera path families |
| `pair_scenarios.py` | 2026-07-28 | The two scenarios built **for** that pair corpus, and only for it: `weighted_roll` (offset CoM modulates rolling speed — the "loaded die" wobble) and `collide_static` (wall impulse acts off-CoM). Both are CoM-signature designs with no other purpose; imported solely by `collect_pairs.py`. | nothing — `scenarios.BUILDERS` is untouched, these were additive registrations |

## Explicitly NOT retired (checked 2026-07-27)

- **`src/probe/com_oracle.py`** — the CoM *target* is abandoned, but this module
  is a **live dependency**: `material_oracle` imports `_traj_features` and
  `_half_extents` from it, and the oracle *method* (target-agnostic, no-GPU
  scenario acceptance from `state.json`) is the durable asset from that line.
- **`src/probe/corpus_acceptance.py`** — zero importers today, but the corpus
  rebuild needs exactly this. Zero importers ≠ obsolete.
