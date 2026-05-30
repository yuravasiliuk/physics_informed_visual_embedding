# Observability-Weighted Loss for Physics Targets

*Date: 2026-05-30. Design note addressing Risk 6 of
[BRAINSTORM.md](BRAINSTORM.md)
("unlearnable-but-perfectly-labeled targets"). Companion to the CoM /
heteroscedastic-uncertainty discussion there (§4).*

## Problem

CoM offset and mass have **near-zero static visual signature** — they are
recoverable only from *dynamics*. If the dataset mixes informative and
uninformative samples, supervising every sample equally forces the model to
"predict" quantities that aren't present in the input, injecting label noise and
penalizing the model for failing the impossible.

**Idea under consideration:** weight each sample's physics loss by how much
dynamic evidence it actually contains, so visually static objects contribute
little. ("Self-compensating mechanism" — static objects don't participate much
in the loss.)

The mechanism is the right family. The open question is *which* variable to weight
by, and *how* the weighting should be produced.

## Correction: linear velocity is the wrong proxy

The first proposed proxy was linear-velocity magnitude. It is nearly orthogonal
to — possibly anti-correlated with — CoM/mass observability:

- **CoM offset is revealed by rotation, not translation.** In free flight the CoM
  follows a parabola *regardless of where it sits inside the object*; pure linear
  motion carries no information about the offset. What exposes it is tumbling
  *about* an off-center point and, above all, **contact** (tipping, settling). A
  fast non-spinning object reveals nothing about its CoM.
- **Mass is revealed only by forces.** Equivalence principle: any mass traces the
  same parabola. Mass appears only where momentum changes — **collisions /
  contact impulses**. Linear velocity in flight carries no mass information.

So "more linear velocity ⇒ more weight" up-weights the least-informative
(ballistic, non-rotating) samples and can down-weight a slow object tipping over,
which is maximally informative.

## Right proxies (and we have them exactly in sim)

| Target        | Observable via                              | Proxy signal (logged GT in sim)        |
|---------------|---------------------------------------------|----------------------------------------|
| CoM offset    | tumbling about off-center point; contact    | angular displacement over clip + contact events |
| Mass / inertia| force response at impact                    | contact impulse / Δmomentum            |
| ω (target)    | visual orientation-change rate × asymmetry  | (symmetric/untextured objects hide it) |

Because contact forces, ω, and angular displacement are exact sim ground truth,
a **physically-grounded per-sample observability score** can be computed
directly (e.g. a function of integrated angular displacement + total contact
impulse over the clip) — built from the *right* variables, not linear velocity.

## The self-compensating mechanism already exists: heteroscedastic σ

The "automatically don't penalize where the answer isn't recoverable" behavior is
exactly **Gaussian-NLL heteroscedastic regression**:

```
loss = ½ ( ‖μ − CoM‖² / σ² + log σ² )
```

When a sample carries no signal, the model widens σ and the `1/σ²` factor
**automatically down-weights its squared-error gradient**. This is the desired
"static objects don't participate much" effect — but *learned per-sample from the
data* rather than hand-prescribed from a (wrong) heuristic.

Why learned-σ beats any hand-picked weight (incl. a velocity gate):

1. **Keeps unobservable samples in, learning the correct response.** For a sample
   where the offset truly can't be seen, the Bayes-optimal output is "predict the
   prior (centroid / offset = 0) with high uncertainty." NLL teaches exactly that
   (μ → prior, σ → large). A hard gate that zeroes their loss never teaches a
   calibrated "I can't tell," so the model has no abstention behavior at test/real
   time.
2. **Uses the right axis implicitly.** No need to know whether rotation or contact
   is the relevant cue — σ widens whenever *whatever* signal is missing.

This ties directly to the earlier "sharp vs. wide Gaussian" framing: predictive
entropy becomes a readout of CoM identifiability from the observed dynamics
(use full covariance — identifiability is anisotropic).

## Caveats

- **Does not manufacture signal — R6 is still a dataset problem.** Weighting only
  stops the *unobservable subset* from injecting noise. If most data is static,
  the model learns "everything → prior + huge σ," and the weighting has merely
  *revealed* there aren't enough informative samples. Tumbling + contact-rich
  scenarios are still required. Weighting fixes the noise-injection half of R6,
  not the no-signal-in-data half.
- **σ as an escape hatch.** Pure NLL can inflate σ on hard-but-learnable samples
  to dodge the gradient. Mitigate: warm up with plain MSE before enabling the σ
  term; floor/clip σ; or anchor it (below).
- **Effective distribution shift.** Down-weighting low-observability samples
  trains the model mainly on the high-motion regime → possible **miscalibration
  on near-static objects at test/real time**. If real deployment has many static
  objects, the model could be confidently wrong there. Check calibration on the
  static slice explicitly.

## Strongest version: anchor σ with the sim observability score

Since sim gives exact dynamics, supervise the uncertainty itself: use the
physically-grounded observability score (angular displacement + contact impulse)
either (a) as a loss weight, or (b) — more interesting — as a *target that
anchors the σ head*, teaching the model what it *should* be uncertain about. This
turns predicted entropy into a **calibrated identifiability readout** rather than
an unconstrained free parameter.

Trade-off: this bakes in our own observability model, so keep a pure learned-σ
ablation alongside it to detect where the anchor disagrees with what the model
could actually infer.

## Takeaway

Keep the mechanism, change the variable: weight by **angular displacement +
contact** (exact in sim), not linear velocity; and prefer a **learned
(optionally physics-anchored) heteroscedastic σ** over a hand-set weight, because
it self-compensates *and* yields a calibrated "I can't tell" instead of a hard
mask. It mitigates the training-noise half of R6 but does not replace the need
for dynamically-rich scenarios.
