# Manual video generation — scene renders, narration mux, deck assembly

How to (re)generate the explainer videos in `reports/video/`. Narration text and
per-scene cue sheets live in [reports/video/NARRATION.md](../../reports/video/NARRATION.md);
this file is only the *commands* layer.

**Requirements:** manim CE (0.20.x), ffmpeg/ffprobe on PATH. No Isaac Sim involved.

> **Always run from the repo root** — every scene script sets
> `media_dir="reports/video/_media"` *relative to the CWD*; run from anywhere
> else and the renders land in a stray tree.

## 1. The pieces

| piece | where |
|---|---|
| scene scripts (manim) | `scripts/video_scenes/scene01…scene05*.py` |
| paper-figure assets (scene 03b) | `reports/video/assets/*.png` (see §3) |
| raw renders | `reports/video/_media/videos/{480p15,1080p30}/<scene>.mp4` (no audio) |
| narration recordings | `reports/video/narration/sceneNN_*.wav` |
| final narrated chunks | `reports/video/<scene>.mp4` (480p canonical) · `reports/video/1080p/` |
| assembled deck (narration only) | `reports/video/deck_480p.mp4` · `reports/video/1080p/deck_1080p.mp4` |
| **shipped final** (music + logo) | `reports/video/1080p/deck_1080p_final.mp4` (see §7) |
| music bed + logo assets | `reports/video/assets/music.wav` · `reports/video/assets/logo_gl.png` |

Deck order: **01 → 02 → 03b → 04 → 05** (`scene03_vjepa_masked` is retired).
Scene 01's *module* is still `scene01_allocentric_worldlines.py` — the class and
output are "exocentric"; the filename stays because the sibling scenes import
the shared palette from it. Its wav also keeps the historical name; matching is
by the `sceneNN_` prefix, so nothing needs renaming.

## 2. Render a scene

```bash
python3 scripts/video_scenes/scene02_egocentric_representation.py          # 854×480@15 (-q l, default)
python3 scripts/video_scenes/scene04_linear_probe.py -q h                  # 1920×1080@30
```

Output: `reports/video/_media/videos/<quality>/<scene>.mp4` — video only.
All five scenes are *paced to their recordings* (beat times sit on the wav's
pause boundaries — 03b, 04 and 05 cue them on an absolute clock, the `until(t)`
helper in `construct()`), so after re-recording a wav, check the beat cues
before trusting the sync. Pause boundaries of a new wav:

```bash
ffmpeg -i reports/video/narration/sceneNN_*.wav \
    -af silencedetect=noise=-32dB:d=0.35 -f null - 2>&1 | grep silence_
```

## 3. Re-capturing the paper figures (scene 03b assets)

Scene 03b shows two paper figures as high-res crops **including the printed
figure caption** (the caption text explaining the figure is part of the image):

| asset | source |
|---|---|
| `reports/video/assets/vjepa2_fig1_caption.png` | `papers/v-jepa2.pdf` p. 2 — Fig. 1, system overview |
| `reports/video/assets/vjepa21_fig4_caption.png` | `papers/v-jepa2-1.pdf` p. 5 — Fig. 4, detailed architecture |

Captured at **400 dpi** (page → 3400×4400 px, crop ≈ 2720 px wide — ~2.3×
oversampled for the on-screen size at 1080p):

```bash
pdftoppm -f 2 -l 2 -r 400 -png papers/v-jepa2.pdf   /tmp/vjepa2    # → /tmp/vjepa2-02.png
pdftoppm -f 5 -l 5 -r 400 -png papers/v-jepa2-1.pdf /tmp/vjepa21   # → /tmp/vjepa21-05.png
python3 - <<'EOF'
from PIL import Image
Image.open('/tmp/vjepa2-02.png').crop((340, 320, 3060, 2100)) \
     .save('reports/video/assets/vjepa2_fig1_caption.png')
Image.open('/tmp/vjepa21-05.png').crop((340, 340, 3060, 2100)) \
     .save('reports/video/assets/vjepa21_fig4_caption.png')
EOF
```

If a crop box changes (e.g. a new paper revision reflows the page), the
highlight regions in scene 03b must be re-checked: each `_mark((fx0, fy0, fx1,
fy1), …)` is in *fractions of the cropped image* (x right, y down). Verify by
extracting frames at the beat times (§6) and eyeballing the boxes. The older
caption-less crops (`vjepa2_overview.png`, `vjepa21_architecture.png`) are
superseded but kept.

## 4. Attach narration + build the deck — `build_deck.py`

One script does both: any scene mp4 in `--src` that has **no audio track** but
has a matching `reports/video/narration/sceneNN_*.wav` gets the wav muxed in
(video's last frame is frozen if the audio runs longer — pad is computed from
the real durations, never hard-coded). Then all requested scenes are joined
with a 0.5 s background-colour gap (`--gap`), silence filling the un-narrated
ones, every segment normalized to the largest resolution / highest fps present.

```bash
# full 480p deck from the canonical chunks already in reports/video/
python3 scripts/video_scenes/build_deck.py

# subset, any order (bare stems, "name.mp4", or paths all accepted)
python3 scripts/video_scenes/build_deck.py \
    --scenes scene01_exocentric_worldlines scene02_egocentric_representation
```

Flags: `--src` (where the scene mp4s are), `--result_dir` (muxed chunks + deck
go here), `--scenes`, `--gap`, `--crf`, `--name` (default `deck_<height>p.mp4`),
`--narration_dir`.

## 5. Full 1080p workflow

```bash
# 1) render every scene at 1920×1080@30
for f in scene01_allocentric_worldlines scene02_egocentric_representation \
         scene03b_vjepa_architecture scene04_linear_probe scene05_oracle; do
  python3 scripts/video_scenes/$f.py -q h
done

# 2) mux narration + assemble → reports/video/1080p/deck_1080p.mp4
python3 scripts/video_scenes/build_deck.py \
    --src reports/video/_media/videos/1080p30 --result_dir reports/video/1080p
```

Add `--scenes …` to step 2 for a partial deck (e.g. only the narrated ones).
New wavs dropped into `reports/video/narration/` are picked up automatically on
the next build — no script changes needed.

A single 1080p narrated chunk (without building a deck) is the render from
step 1 plus a plain mux — the recorded scenes outlast their wavs by design, so
no last-frame freeze is needed:

```bash
ffmpeg -y -v error -i reports/video/_media/videos/1080p30/sceneNN_x.mp4 \
    -i reports/video/narration/sceneNN_x.wav -map 0:v -map 1:a \
    -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 160k \
    reports/video/1080p/sceneNN_x.mp4
```

## 6. Verifying a build

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 reports/video/1080p/deck_1080p.mp4
ffmpeg -y -ss 8 -i reports/video/1080p/deck_1080p.mp4 -frames:v 1 /tmp/frame.png   # eyeball a frame
ffmpeg -ss 1 -t 3 -i reports/video/1080p/deck_1080p.mp4 -af volumedetect -f null - 2>&1 | grep mean_volume
```

Expect: deck duration ≈ Σ chunk durations + 0.5 s × (n−1); narrated segments
around −20 dB mean volume, silent segments ≈ −91 dB; gap frames are the flat
palette background `#0B1020`.

## 7. The shipped final — chapter cards + music + logo

`deck_1080p.mp4` (§4/§5) is the **narration-only master**. The shipped final,
`reports/video/1080p/deck_1080p_final.mp4`, wraps it with three things: a **5 s
chapter card before each scene**, a **background music bed** that *swells over
each card* and ducks back under the narration, and the **logo** in the bottom-left
corner. Two scripts do it — the content scenes never change.

**a) Render the chapter cards** (`scene_chapter.py`, one per chapter — 5 s animated
title cards; edit the titles/subtitles here):

```bash
render() { python3 scripts/video_scenes/scene_chapter.py -q h \
             --index "$1" --title "$2" --subtitle "$3" --out "chapter$1"; }
render 1 "Exocentric world-lines"    "Physics as trajectories, seen from outside"
render 2 "Egocentric representation" "The same world, from the agent's view"
render 3 "Inside V-JEPA 2"           "Predicting masked video in latent space"
render 4 "What is a linear probe?"   "Reading physics out of frozen features"
render 5 "What is the oracle?"       "Certifying what's knowable before you probe"
```

Cards land in `reports/video/_media/videos/1080p30/chapter{1..5}.mp4` (silent).

**b) Assemble + score + brand** (`finalize_deck.py`). It concats
`card1·scene1·…·card5·scene5` (logo burned in one video encode), tiles `music.wav`
into a **crossfaded** bed (a hard loop pumps — the clip fades in, ends loud), bakes
a volume **envelope** into it (numpy), and mixes it under the narration (video is
copied — no second encode):

```bash
python3 scripts/video_scenes/finalize_deck.py       # all defaults → deck_1080p_final.mp4
```

The envelope: baseline `--music_volume` (0.23) under narrated scenes; over each card
the bed swells to `--music_volume × --transition_boost` (2×), ramping up
`--ramp_up` s (0.8) at the card start and ramping **down** `--ramp_down` s (1.5)
back to baseline once the next narrated scene begins; plus a global
`--fade_in`/`--fade_out`. `amix … normalize=0` keeps the voice at full level;
`alimiter` guards clips. Every knob is a flag:

| lever | flag (default) |
|---|---|
| bed loudness under narration | `--music_volume 0.23` |
| card swell factor | `--transition_boost 2.0` |
| swell-up / duck-down times | `--ramp_up 0.8` · `--ramp_down 1.5` |
| global fades | `--fade_in 1.5` · `--fade_out 3.0` |
| logo size / inset / opacity | `--logo_scale 150` · `--logo_margin 44` · `--logo_opacity 0.9` |

Intermediate `deck_1080p_chapters.mp4` (logo burned, narration only, no music) is
written alongside the final. Re-run **b** after any scene re-render or a title
edit; re-run **a** first only if the cards changed.

Verify: duration ≈ Σ scenes + 5 × 5 s cards (≈ 271 s); a bottom-left logo on a
frame grab; and, isolating audio in a silent inter-card moment, the bed reads
≈ −33 dB under scenes and ≈ 2× that (≈ −27 dB) across a chapter card.
