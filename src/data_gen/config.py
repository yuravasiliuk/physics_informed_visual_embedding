"""Single source of truth for data-generation configuration.

Two kinds of config live here:

  * `ClipConfig` — the per-clip / per-run parameter bundle (frozen dataclass),
    overridable via `dataclasses.replace`. Defaults follow §7 of
    `plans/data-gen/probe-dataset.md`.
  * Module-level constants — scenario-builder tunables (geometry, spawn boxes,
    speed/angle ranges) and a couple of filesystem paths.

This module deliberately has NO `omni` / `isaac` imports, so it is safe to
import before a `SimulationApp` exists and creates no import cycles (every
other data_gen module may import it freely).
"""
import os
from dataclasses import dataclass


# ============================================================================
# Per-clip generation parameters
# ============================================================================

@dataclass(frozen=True)
class ClipConfig:
    """Per-clip generation parameters. Defaults follow §7 of plans/data-gen/probe-dataset.md."""

    # Capture
    frame_count: int = 64                     # K — plan §7
    resolution: tuple = (1024, 1024)            # (width, height) — render/capture res, plan §7
    save_resolution: tuple = (1024, 1024)         # (width, height) saved rgb/depth/normals are
                                                # downsampled to this; capture stays at `resolution`
    physics_dt: float = 1.0 / 60.0            # physics step — plan §7 (60 Hz)
    render_dt: float = 1.0 / 30.0             # render step — plan §7 (30 fps)
    camera_position: tuple = (2.2, 2.2, 1.8)  # world frame — single-camera fallback (verify, legacy)
    camera_look_at: tuple = (0.0, 0.0, 0.1)   # world frame — rig look-at base too

    # Multi-view capture (scene-diversification.md §10, axis G). One physics
    # rollout is rendered from up to `n_cameras` viewpoints fanned out into
    # INDEPENDENT clips (each its own folder, single-camera contract unchanged).
    # Physics steps once. Cameras are anchored on the body cloud (not the floor)
    # and render-probe-gated so each clip actually shows the action; a viewpoint
    # that stays occluded/empty after resampling is dropped, so a rollout may
    # emit FEWER than n_cameras clips (camera-framing-fix.md — quality > count).
    n_cameras: int = 7
    camera_elevation_range: tuple = (20.0, 55.0)   # deg, base band spread across the rig
    camera_azimuth_jitter_deg: float = 12.0        # per-camera ± on the evenly-spaced azimuth
    camera_elevation_jitter_deg: float = 8.0       # per-camera ± on the base elevation
    camera_elevation_clamp_deg: tuple = (12.0, 68.0)  # keep above floor / below near-top-down NaN
    camera_lookat_jitter: float = 0.08             # m, ± per-axis jitter on the look-at point
    # Body-cloud framing + visibility gating (camera-framing-fix.md §4–§5). The
    # rig distance is solved from the pre-rolled cloud radius so the action fills
    # ~frame_fill_frac of the half-frame, instead of a fixed far radius.
    frame_fill_frac: float = 0.4                   # target cloud fraction of the half-frame
    camera_radius_clamp: tuple = (0.8, 4.0)        # m, min/max solved framing distance
    min_body_coverage: float = 0.01               # reject a camera below this body-pixel fraction
    camera_resample_attempts: int = 8              # az/elev retries before dropping a camera

    # Scene randomization
    scenarios: tuple = (
        "drop", "roll", "throw",
        "drop_on_platform", "settle", "push",
        "cluttered", "topple", "collide",
    )                                                                         # uniform sampling per clip
    n_bodies_range: tuple = (3, 8)                                            # inclusive
    shape_choices: tuple = ("cuboid", "sphere", "cylinder", "capsule")
    scale_range: tuple = (0.05, 0.15)                                         # m; per-axis (cuboid) or radius/height
    # Bodies are scaled up when a large environment (warehouse etc.) is loaded
    # so they read at the env's scale instead of being ~1% of frame (Stage 3,
    # camera-framing-fix.md). build_scene swaps scale_range -> scale_range_env
    # when env_loaded. NOTE: topple's _sample_tall_dims does not read this.
    scale_range_env: tuple = (0.15, 0.45)                                     # m; body scale when an env USD is loaded
    color_hue_range: tuple = (0.0, 1.0)
    color_sat_range: tuple = (0.5, 1.0)
    color_val_range: tuple = (0.4, 0.9)
    static_friction_range: tuple = (0.3, 0.8)
    dynamic_friction_range: tuple = (0.3, 0.8)
    restitution_range: tuple = (0.0, 0.4)
    # Per-body density (kg/m^3), authored so mass is decoupled from visible
    # volume (critique #7 / physics-diversification.md W1). ~styrofoam(200) →
    # steel(8000). Without this, mass = 1000·volume and a "predict mass" probe
    # only ever learns visible size.
    density_range: tuple = (200.0, 8000.0)
    # Per-body CoM offset as a fraction of each half-extent (W2 / R7). Authored
    # via MassAPI so the CoM-offset label has real variance instead of being
    # identically zero. 0.3 keeps the CoM well inside the collision volume
    # (sphere worst case: 0.3*sqrt(3)*r ~= 0.52 r < r). Appearance gives no cue.
    com_offset_frac: float = 0.3
    # Composite glued bodies (physics-diversification.md W5). A fraction of
    # dynamic bodies are built as a uniform-colour cuboid internally split into
    # 2-3 contiguous density slabs, so the offset CoM *emerges* from a real mass
    # distribution PhysX integrates (mass=ΣρV, com=ΣρV·t/ΣρV — closed-form,
    # verified). Density asymmetry is invisible (one shared material) — the CoM
    # is recoverable only from geometry + dynamics, never pixels (R8). Per-slab
    # density reuses density_range above; the wide range creates the asymmetry.
    composite_frac: float = 0.3                  # P(a dynamic body is a composite)
    composite_n_parts_range: tuple = (2, 3)      # inclusive slab count along one axis
    composite_split_min_frac: float = 0.2        # min fraction of the split axis per slab
    init_xy_range: tuple = (-0.3, 0.3)
    init_height_range: tuple = (0.4, 1.2)
    init_ang_vel_range: tuple = (-2.0, 2.0)                                   # rad/s, uniform per axis
    pose_rejection_max_attempts: int = 100

    # PBR visual materials — Tier 3 (plan §5 / step 5.10).
    roughness_range: tuple = (0.2, 0.9)
    metallic_prob: float = 0.15
    metallic_values: tuple = (0.8, 1.0)

    # Ground colour — Tier 3. Greys with a slight tint (per RGB channel,
    # uniform in this range). Texture-based ground materials and "room"
    # enclosures (plan §5) deferred — they need external PBR assets.
    ground_color_range: tuple = (0.25, 0.55)

    # Ground grid texture (scene-diversification.md §10.4, axis D). A locally
    # generated grid-line PNG bound to the ground via OmniPBR gives a
    # translation/parallax cue the flat colour lacks. texture_scale (UV repeats)
    # is randomized per clip for line-spacing variety; suppressed when an
    # environment USD ships its own floor.
    ground_grid_enabled: bool = True
    ground_grid_tiling_range: tuple = (6.0, 16.0)  # UV repeats across the plane

    # Lighting (HDRI). Populate with file paths or nucleus URIs to enable HDRI
    # randomization; when empty, the procedural sky below is used instead.
    hdri_paths: tuple = ()
    hdri_rotation_range_deg: tuple = (0.0, 360.0)
    hdri_intensity: float = 1000.0

    # Procedural sky — active when `hdri_paths` is empty. A randomized DomeLight
    # (uniform sky colour) + a directional DistantLight ("sun") at a randomized
    # azimuth/elevation. Replaces the old "single DistantLight that looks like a
    # night" fallback so the procedural-ground branch gets real lighting variety
    # without external HDRI assets.
    procedural_sky_enabled: bool = True
    sky_dome_intensity_range: tuple = (400.0, 1200.0)
    sky_dome_hue_range: tuple = (0.0, 1.0)
    sky_dome_sat_range: tuple = (0.0, 0.25)        # pastel — sky tint, not saturated
    sky_dome_val_range: tuple = (0.75, 1.0)
    sun_intensity_range: tuple = (1500.0, 4500.0)
    sun_azimuth_range_deg: tuple = (0.0, 360.0)
    sun_elevation_range_deg: tuple = (20.0, 70.0)  # keep above horizon, below zenith
    sun_hue_range: tuple = (0.05, 0.15)            # warm range (subtle orange/yellow)
    sun_sat_range: tuple = (0.0, 0.2)              # near-white with slight warm tint
    sun_val_range: tuple = (0.85, 1.0)

    # Isaac environment backdrops (warehouse / simple_room). Enabled: each clip
    # samples uniformly over (discovered envs + "none"), mixing richly-textured
    # backdrops with the bare procedural ground + sky. The one bad asset that
    # motivated disabling this — office.usd, which rendered its floor textureless
    # — is excluded at the library level (environments.CANDIDATE_ENVIRONMENTS),
    # not by switching off every env. Set False to force procedural-only; the
    # `--environment` CLI override works either way.
    use_isaac_envs: bool = True


# ============================================================================
# Filesystem paths
# ============================================================================

DEFAULT_DATA_PATH = os.path.expanduser("~/data/sim_physics")              # collect_clip default --result
GRID_TEXTURE_CACHE = os.path.expanduser("~/.cache/sim_physics/grid_texture.png")  # generated ground grid PNG


# ============================================================================
# Scenario-builder tunables (scenarios.py)
# ============================================================================

# Body scale (scale_range[1]) the spawn footprints + procedural static geometry
# below were tuned for. scenarios._scene_scale() rescales every spawn box and
# procedural asset by scale_range[1] / BODY_SCALE_REFERENCE, so when bodies are
# enlarged for env scenes (scale_range_env) the apparatus keeps the same
# relative proportions instead of being overflowed (camera-framing-fix.md
# Stage 3). At the reference scale the factor is exactly 1.0 (no change).
BODY_SCALE_REFERENCE = 0.15

# --- roll -------------------------------------------------------------------
ROLL_RAMP_LENGTH = 0.6
ROLL_RAMP_WIDTH = 0.6
ROLL_ANGLE_RANGE_DEG = (15.0, 30.0)
ROLL_SPAWN_XY_RANGE = (-0.2, 0.2)
ROLL_SPAWN_HEIGHT_RANGE = (0.6, 1.0)

# --- throw ------------------------------------------------------------------
THROW_SPEED_RANGE = (0.5, 12.0)        # m/s, horizontal magnitude
THROW_UP_RANGE = (0.0, 10.5)           # m/s, +Z component

# --- drop_on_platform -------------------------------------------------------
PLATFORM_LENGTH = 0.4
PLATFORM_WIDTH = 0.4
PLATFORM_HEIGHT = 0.10
PLATFORM_SPAWN_XY_RANGE = (-0.25, 0.25)     # wider than platform half-width=0.2
PLATFORM_SPAWN_HEIGHT_RANGE = (0.5, 1.0)

# --- settle -----------------------------------------------------------------
BOWL_INNER_LENGTH = 0.9
BOWL_INNER_WIDTH = 0.9
BOWL_HEIGHT = 0.25
BOWL_WALL_THICKNESS = 0.02
SETTLE_SPAWN_XY_RANGE = (-0.27, 0.27)        # inside bowl, ~0.05 m wall clearance for max-radius bodies
SETTLE_SPAWN_HEIGHT_RANGE = (0.5, 1.5)       # 1.0 m vertical headroom for rejection sampling

# --- push -------------------------------------------------------------------
PUSH_SPEED_RANGE = (1.5, 3.5)                # m/s, pure horizontal
PUSH_SPAWN_XY_RANGE = (-0.3, 0.3)
PUSH_SPAWN_HEIGHT_RANGE = (0.15, 0.25)       # half-extent ≲0.075 → no spawn-time ground penetration

# --- cluttered --------------------------------------------------------------
CLUTTERED_N_STATIC_RANGE = (2, 5)                # inclusive
CLUTTERED_STATIC_XY_RANGE = (-0.5, 0.5)          # wider than the body footprint
CLUTTERED_STATIC_RADIUS_MARGIN = 0.05            # extra gap between circumscribed circles, m
CLUTTERED_DYNAMIC_XY_RANGE = (-0.4, 0.4)
CLUTTERED_DYNAMIC_HEIGHT_RANGE = (1.0, 1.5)      # well above typical asset heights (~0.5 m)

# --- topple -----------------------------------------------------------------
# A tall, small-footprint body is spawned leaning *just past* its critical tip
# angle so gravity topples it (W3a). A small kick in the tip direction assists;
# it is kept well below the energy barrier so it never swamps the CoM signal.
TOPPLE_N_RANGE = (1, 2)                      # inclusive — keep it clean/observable
TOPPLE_FOOTPRINT_RANGE = (0.05, 0.09)        # small base side / diameter, m
TOPPLE_HEIGHT_RANGE = (0.25, 0.45)           # tall, m — aspect ratio ~3-9:1
TOPPLE_KICK_ANGVEL_RANGE = (0.3, 1.0)        # rad/s, small assist in the tip direction (gravity does the work)
TOPPLE_LEAN_MARGIN_DEG = (3.0, 8.0)          # lean this far PAST theta_crit so it always tips
TOPPLE_TIP_JITTER_DEG = 15.0                 # ± jitter on a cuboid's tip-over direction
TOPPLE_XY_RANGE = (-0.30, 0.30)
TOPPLE_GROUND_CLEARANCE = 0.005              # spawn just clear of ground (no t=0 penetration)
TOPPLE_SHAPES = ("cuboid", "cylinder")       # only shapes with a flat base to stand on

# --- collide ----------------------------------------------------------------
# Two bodies on a shared horizontal axis aimed at each other; momentum exchange
# makes the mass ratio observable (mass decoupled from volume by W1).
COLLIDE_HALF_SEP_RANGE = (0.30, 0.42)        # m, each body's distance from origin
COLLIDE_SPEED_RANGE = (1.5, 6.0)            # m/s, horizontal closing speed per body
COLLIDE_HEIGHT_RANGE = (0.12, 0.22)          # m, shared spawn height of both centers
COLLIDE_AIM_JITTER_DEG = 8.0                 # max ± aim jitter so it's not perfectly 1D
