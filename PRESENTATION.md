# Physics-Informed Visual Embedding — 5-slide deck

*Audience: funding / leadership, with domain experts in the room. Star of the deck:
the **world-lines** representation. Every slide: a plain-language claim + one expert
line. Bookended with the payoff (slides 1 & 5); the idea at the apex (slide 3).*

---

## Slide 1 — The prize, and why it's still open

**Claim:** Robots and policies can be trained almost for free in simulation — but they
fail in the real world because the **visual encoder** doesn't transfer. Close that gap
once, at the embedding level, and *every* sim-trained policy deploys on real video.

- The prize: train entirely in sim → deploy on real, **no per-task real data**.
- The bottleneck is one component: a visual embedding that survives sim→real.
- **The gap is real and measured:** even the best video models (V-JEPA 2) score
  **~56% on intuitive physics — near chance — where humans hit ~96%** (IntPhys 2).
  Scaling video alone does **not** teach physics.

> *Expert line:* this is the latent-alignment route to sim2real, the alternative to
> pixel-translation (Cosmos-style) — cheaper and label-exact.

*Visual:* sim render → encoder → real render, with a broken bridge; the 56% vs 96% bar.

---

## Slide 2 — Why models fail where humans don't (the diagnosis)

**Claim:** Humans don't predict pixels — they parse the world into **persistent objects
with physical properties** and track them through occlusion and their own motion. That
*ontology* is exactly what video-trained models lack.

- The missing ingredient is **structure, not data** (a model on 0.01% of the data does
  as well as full-scale — the bottleneck is the learning signal).
- So: **inject physics from a simulator**, which knows mass, center-of-mass, spin exactly.
- The open question isn't *whether* to supervise physics — it's **how to represent it**
  so it's compact and transferable.

> *Expert line:* today's physics targets (per-object state, masked-pooled per frame) are
> a hand-engineered intermediate — the "anchor boxes" of physics. We can do better.

*Visual:* same clip, human view (object tokens persisting behind an occluder) vs model
view (surprise signal flat).

---

## Slide 3 — The idea: world-lines  ★

**Claim:** Represent a scene as a handful of **world-lines** — each object's trajectory
through space-time — in a frame anchored to gravity, plus the camera's own trajectory
(the *ego-line*). One compact, learnable description of the dynamics.

Three properties, and each fixes a specific failure:

- **Exocentric (world-, not camera-centred)** → an object's world-line *continues behind
  an occluder* → **object permanence** becomes expressible (the #1 model failure).
- **Camera factored out into the ego-line** → **robust to camera motion by construction**
  (the documented weak spot of every video model).
- **Dimensionless, gravity-natural units** → **camera- and scale-agnostic**; the targets
  are ratios (spin, mass ratio, mass offset) that mean the same thing in sim and real →
  built to transfer.

> *Expert line:* a similarity-invariant target — invariant intrinsics ⊕ covariant
> kinematics, set-predicted over object queries; gravity breaks the scale gauge, the
> ego-line removes the observer. Structurally it is multi-object tracking lifted from 2D
> pixels into an abstract, physics-bearing space.

*Visual:* one clip → 3 object world-lines + 1 ego-line in a gravity-aligned box.

---

## Slide 4 — Why it will work — and how we de-risk it cheaply

**Claim:** We don't bet big blind. Two cheap, **falsifiable** checkpoints gate every
dollar of scale, and the foundation is already built and verified.

- **GATE A** (~1 GPU-day): is dynamics learnable from *perfect* pixels at all? If not,
  stop.
- **GATE B**: is the physics *already inside* a frozen off-the-shelf encoder, and at what
  depth? We **measure** it with a probe — no expensive retraining assumed.
- **Already built & verified:** an Isaac-Sim data pipeline (real clips, physics labels
  checked against closed-form ground truth); a frozen-encoder probe harness; and the
  world-line encoder — which reconstructs **real** sim trajectories across all 9 scenarios
  to **~1–2% error**. The representation already holds on real dynamics.

> *Expert line:* frozen-first by design — trainable parameters never touch raw pixels, so
> sim→real stability holds by construction; the gates decide if the escape hatch is ever
> earned.

*Visual:* gates-as-checkpoints timeline; a real world-line overlaid on its reconstruction.

---

## Slide 5 — The path to impact, and the ask

**Claim:** A short, measurable path from here to the load-bearing result: a sim-trained
policy running on **real** video.

- **Roadmap:** gates → physics-supervised encoder (baseline vs full) → **sim→real policy
  transfer eval** (the real test, not a proxy).
- **The wedge:** latent-space alignment + a transferable physics target — distinct from,
  and cheaper than, the pixel-translation incumbents.
- **Honest risk, bounded:** the sim-only labels have no real-world ground truth (can't be
  directly checked) — which is *why* the targets are scale-free ratios and why we lean on
  measurable proxies + the transfer eval.
- **The ask:** [compute + N researchers + M months] to clear GATE A/B and run the first
  transfer eval.

*Visual:* roadmap arrow ending at "policy on real video → success rate"; the ask in one box.

---

### Notes on delivery
- Spend the most time on **Slide 3**; slides 1–2 set it up, 4–5 cash it in.
- Keep the IntPhys 2 number (56 vs 96) as the single statistic — repeat it once on slide 4
  framing GATE B ("we test whether we can beat 56%").
- The bracketed ask on slide 5 is the only thing to fill in before presenting.
