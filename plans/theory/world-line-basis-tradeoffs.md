# World-line descriptors: basis tradeoffs and the empirical degree sweep

Working note capturing the "what function family should encode a world-line"
discussion, grounded in a measured degree sweep over the real corpus. Companion
to [world-lines.md](world-lines.md) §4.4 (the compact-form choice) and §5
(contact). Code: [../../src/worldlines/degree_sweep.py](../../src/worldlines/degree_sweep.py).
Plot: `out/worldlines/degree_sweep.png`.

## 0. The question
A world-line is 3 scalar functions of time (+ an SO(3) rotation curve). What
basis encodes it *compactly and learnably*? The earlier costing said ~9 numbers
for a contact-free arc, blowing up at contact. This note measures the blow-up
and weighs the alternative descriptor families.

## 1. The degree sweep (110 sim_physics clips, gravity gauge, Chebyshev)
Error = decoded-vs-GT; position normalized by object bbox-diagonal (fraction of
object size), rotation as mean geodesic angle (rad). Distribution over all
bodies, since the bounce clips live in the tail (the mean hides them).

**Least degree meeting a "good" threshold:**

| criterion | median | p90 |
|---|---|---|
| position < 10% of object size | deg **8** | deg **20** |
| position < 5% | deg **13** | never (≥20) |
| position < 2% | never | never |
| rotation < 0.10 rad | **never** | **never** |

- **Position has no good global degree.** Median needs deg-13 for <5% size (=42
  numbers for position alone — not "compact"); **p90 never clears 5%** in the
  sweep. Decay is slow (~algebraic / 1/N), the signature of a real-axis
  singularity (§3).
- **Rotation is degree-INDEPENDENT** — see §2, the important finding.

## 2. Rotation: a parametrization flaw, not a degree problem
Rotation geodesic error **floors at ~0.6 rad median / ~1.5 rad p90 by degree-5
and stays flat to degree-20.** More coefficients do nothing. Cause:
`fit_rotations`/`reconstruct_rotations` fit one cumulative-rotvec curve and apply
a **single `exp()`**, exact only for a *fixed* rotation axis. A tumbling body
(changing axis) violates SO(3) commutativity, leaving an O(commutator) error no
basis degree removes. (synth.py used constant ω = fixed axis → round-tripped to
~0, which hid this until real tumbling data.) **Fix is orthogonal to degree:** a
Lie-group-correct curve — product of incremental exps, or a quaternion/SLERP/
SQUAD or Lie-group B-spline — not more coefficients. Tracked as a real bug.

## 3. Per-scenario: the singularity prediction, confirmed
Position median, deg-2 → deg-20:

| scenario | deg-2 | deg-20 | regime |
|---|---|---|---|
| topple | 0.068 | 0.005 | smooth pivot → fast |
| collide / push / roll | ~0.33 | ~0.02 | moderate |
| drop_on_platform / settle | 0.41 / 0.55 | 0.026 / 0.035 | contact |
| drop | 0.711 | 0.053 | bounce corner → slow |
| throw | 1.182 | 0.052 | parabola + landing corner |
| cluttered | 1.319 | 0.097 | many corners → slowest |

Smooth (topple) converges ~10× better than contact-rich (drop/throw/cluttered) —
exactly the complex-analysis prediction in §4.

## 4. Descriptor families weighed
- **Polynomial / Chebyshev (current).** Exact for ballistic (deg-2). Dies at
  contact: a velocity kink is a C⁰ corner = real-axis singularity; Chebyshev
  convergence rate is set by the nearest singularity's distance to the interval
  (Bernstein ellipse), so a corner forces slow ~1/N decay and Gibbs ringing that
  **pollutes the otherwise-smooth segments globally**. This is *why* §1 shows no
  good degree.
- **Fourier / complex exponentials.** No help: same Gibbs at corners, worse for
  non-periodic motion; natively 2D while trajectories are 3D. Complex-as-a-basis
  is a non-starter here. (Complex analysis is valuable only as the *explanatory
  lens* above, not as a basis.)
- **Padé / rational `P(t)/Q(t)`.** The denominator places **poles**, capturing
  sharp/near-singular features in far fewer terms than a polynomial. Defined over
  the reals (not "about" the complex plane — the poles merely *live* in ℂ, which
  is *why* it beats Taylor near singularities). Promising for compact *encoding*
  of a bounce; but nonlinear to fit and an unstable *regression target* for a
  learned head.
- **Laplace / Prony (exponential modes).** The Laplace transform is a compact
  descriptor only when the signal is a sum of `tⁿe^{λt}` (linear-ODE solutions),
  where the descriptor = **poles + residues** = time-domain **Prony /
  exponential fitting** — the same rational-pole family as Padé, not a new tool.
  Genuine niche win: **damped/settling motion** (`settle`, rolling-resistance),
  where motion decays ~exponentially — polynomials fit that terribly, a pole
  (= the friction/damping rate) fits it in 2–3 *physically meaningful* numbers.
  Does **not** dissolve the contact corner (a bounce is nonlinear switching, not
  an LTI mode; the shift theorem `e^{-sτ}` only handles a delayed *copy*, not a
  dynamics change).

## 5. Verdict
There is no satisfying global-basis sweet spot for contact-rich data. The two
problems are independent:
- **Position** → **piecewise / event (knot at contact)** is necessary, not just
  nicer; optionally **rational/Prony** for damped segments. The sim *knows* the
  contact times, so a GT-side piecewise fit is near-exact at low degree per
  segment — the hard part (detecting contact from video) only appears at the
  perception stage.
- **Rotation** → a **Lie-group spline** regardless of degree (§2).

Polynomial-degree was the wrong knob; *segmentation at events* + *the right
family per segment* (low-degree poly for arcs, exponential for decay, Lie-group
for rotation) is the real lever.
