"""Shared CLI surface for the clip drivers — one declaration per knob.

`collect_clip.py` (single scenario, N rollouts) and `collect_corpus.py` (all
scenarios, one Isaac boot) expose the SAME set of ClipConfig overrides. Before
this module they each declared those flags and each applied them with a
hand-written `dataclasses.replace` block: 12 of 14 flags duplicated, two copies
of every help string, and two places to forget. That is exactly how the drift
below crept in, and how the retired `collect_pairs` driver ended up with its
"mirror the change here manually" note.

Scope is deliberately narrow: **only the knobs that write into ClipConfig live
here.** Genuinely driver-local arguments (`--result`, `--n_clips`, `--seed` vs
`--seed_stride`, `--scenario` vs `--scenarios`, `--environment`) stay in their
driver, because their semantics really do differ — pretending otherwise would
trade duplication for a worse abstraction.

Pure stdlib + config + camera_motion: no omni import, so a driver's argument
handling is testable without Isaac.

    import cli
    cli.add_config_args(parser, defaults={"n_cameras": 1})
    args = parser.parse_args()
    cli.validate_camera_args(args)
    config = cli.config_from_args(args)

★ KNOWN DIVERGENCE (surfaced 2026-07-28 by this extraction, NOT introduced by
it): `ClipConfig.n_cameras` is 7 while `collect_corpus` has always defaulted it
to 1 ("groups-not-views" — the probe splits on rollouts, not cameras, see
probe-corpus-spec.md). So the two drivers produce different corpus shapes by
default, and editing ClipConfig.n_cameras would move one and not the other.
Preserved verbatim here via each driver's `defaults=` so this extraction is a
no-op; it is a decision to be made, not smuggled into a refactor.
"""

import dataclasses

from config import ClipConfig


# (flag, ClipConfig field, argparse kwargs, help). One row per knob: adding a
# ClipConfig override is a one-line change that both drivers pick up.
_CONFIG_ARGS = (
    ("--n_cameras", "n_cameras", {"type": int},
     "cameras fanned out per rollout. 1 converts render budget into INDEPENDENT "
     "rollout groups (probe-corpus-spec.md: the probe splits on rollouts, not "
     "cameras)."),
    ("--depth_format", "depth_format", {"choices": ["npz", "ffv1", "hevc12"]},
     "depth storage backend (depth_map_factory.py). npz=uint16 npz (bit-exact); "
     "ffv1=lossless 16-bit video; hevc12=lossy 12-bit (smallest)."),
    ("--frame_count", "frame_count", {"type": int},
     "rendered frames per clip. A dynamic camera makes this load-bearing: path "
     "displacement is speed x frame_count/fps, so a short clip can make a "
     "moving camera look nearly static (dynamic-camera.md)."),
    ("--camera_families", "camera_path_families", {"nargs": "+"},
     "camera path families sampled per camera, spanning BOTH camera models: "
     "subject-anchored (static orbit arc dolly roll handheld gait) and free "
     "6-DoF (still dolly_in dolly_out truck pedestal steer handheld gait "
     "orbit_free survey). Names carried by both MUST be prefixed — "
     "'subject:gait' vs 'free:gait'. 'static' is the zero-coefficient case = "
     "legacy behaviour."),
    ("--camera_weights", "camera_path_weights", {"nargs": "+", "type": float},
     "sampling weights parallel to --camera_families (default uniform)."),
    ("--camera_roll_always", "camera_roll_always", {"action": "store_true"},
     "force a roll component on every path. REQUIRED for any batch that must "
     "carry the gravity-tilt label: only roll breaks g_cam.x==0 — orbit is pure "
     "yaw about the gravity axis and dolly is pure translation, and both leave "
     "tilt EXACTLY invariant (verify_camera_paths.py)."),
    ("--camera_speed_scale", "camera_speed_scale", {"type": float},
     "global time warp on every camera path, BOTH camera models: the same shot "
     "traversed this many times faster (2.0 = twice the travel over the same "
     "frame_count). Same seed = same shot at any speed, so a sweep is a "
     "controlled comparison. Raising it fights the egomotion-flow acceptance "
     "bound and costs along-path gate drops."),
    ("--camera_track_tau", "camera_track_tau", {"type": float},
     "free-6-DoF families only: per-frame blend toward re-aiming at the look-at "
     "point. 0 = dead reckoning (the subject leaves frame, which the along-path "
     "gate then drops); ~0.1-0.3 = loose follow; 1 = locked on."),
    ("--camera_path_gate_samples", "camera_path_gate_samples", {"type": int},
     "poses probed along each path for visibility gating; 1 disables the "
     "along-path gate (useful when you WANT to see a path leave frame)."),
    ("--log_contacts", "log_contacts", {"action": "store_true"},
     "log PhysX contact events (solver-exact impulses at physics rate + the "
     "event timestamps event-anchored targets are built on). REQUIRED for "
     "restitution and for mass-ratio-at-collisions — see contact_log.py."),
)

# Flags whose absence is indistinguishable from an explicit False, so they are
# only ever applied when true (never used to force a config field back to False).
_STORE_TRUE = {flag for flag, _, kw, _ in _CONFIG_ARGS if kw.get("action") == "store_true"}


def add_config_args(parser, *, defaults=None):
    """Register every ClipConfig-override flag on `parser`.

    `defaults` maps a ClipConfig field name to the value that flag should take
    when the user omits it. Omitted fields default to `None` = "leave whatever
    ClipConfig says", which keeps ClipConfig the single source of truth. Pass a
    value only where a driver deliberately overrides ClipConfig (see the
    n_cameras note in the module docstring).
    """
    defaults = defaults or {}
    for flag, field, kwargs, help_text in _CONFIG_ARGS:
        kw = dict(kwargs)
        if kwargs.get("action") != "store_true":
            kw["default"] = defaults.get(field, None)
        default_note = ("" if field not in defaults
                        else f" [driver default: {defaults[field]!r}]")
        parser.add_argument(flag, help=help_text + default_note, **kw)
    return parser


def validate_camera_args(args):
    """Reject unknown, ambiguous or UNIMPLEMENTED families, and bad weights.

    Runs at argument-parse time so a bad family costs a second rather than an
    Isaac boot plus a rollout. Three distinct failures, all caught here:
      * unknown name — a typo;
      * ambiguous name — `handheld`/`gait` exist in BOTH camera models and
        behave differently, so a bare name is a hard error, not a default;
      * resolves but is a STUB — `check_family_implemented`. This one was
        missed until 2026-07-28, when `free:steer` passed validation and then
        raised NotImplementedError ~12 s in, from inside `CameraRig.solve`
        with the stage already built. Name-resolution alone was never the
        question a driver needs answered.
    Raises SystemExit with the underlying message.
    """
    from camera_motion import (                   # deferred: keeps import cheap
        check_family_implemented, implemented_families)

    families = getattr(args, "camera_families", None)
    if families:
        for f in families:
            try:
                check_family_implemented(f)
            except ValueError as e:
                raise SystemExit(f"{e}\navailable now: {' '.join(implemented_families())}")
    weights = getattr(args, "camera_weights", None)
    if weights is not None and (not families or len(weights) != len(families)):
        raise SystemExit("--camera_weights must be parallel to --camera_families")


def config_from_args(args, base=None):
    """Apply every provided override to `base` and return the new ClipConfig.

    `None` means "not provided — keep the ClipConfig value"; store_true flags
    are applied only when true. Tuple-valued fields are normalized to tuples so
    the config stays hashable/comparable regardless of argparse's list output.
    """
    config = base if base is not None else ClipConfig()
    updates = {}
    for flag, field, kwargs, _ in _CONFIG_ARGS:
        value = getattr(args, flag.lstrip("-"), None)
        if flag in _STORE_TRUE:
            if value:
                updates[field] = True
            continue
        if value is None:
            continue
        current = getattr(config, field, None)
        updates[field] = tuple(value) if isinstance(current, tuple) or isinstance(value, list) else value
    # camera_path_weights must be None (not ()) when unset, or the sampler would
    # normalize an empty weight vector.
    if updates.get("camera_path_weights") == ():
        updates["camera_path_weights"] = None
    return dataclasses.replace(config, **updates) if updates else config
