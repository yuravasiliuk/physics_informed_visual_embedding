"""Sky lighting for the procedural-ground scenes (step 5.4 of
plans/scene-diversification.md).

`setup_random_sky` picks one of three paths:

  * **HDRI dome** — when `config.hdri_paths` is non-empty: a DomeLight with a
    randomly-rotated HDRI texture (file path or nucleus URI).
  * **Procedural sky** — when `hdri_paths` is empty and
    `config.procedural_sky_enabled` is True (default): a randomized DomeLight
    (uniform sky colour + intensity) plus a directional DistantLight ("sun") at
    a randomized azimuth/elevation/colour. No external HDRI assets needed —
    every parameter is sampled from `rng` and captured in the returned dict so
    the lighting is reproducible from the seed.
  * **Fallback** — both disabled: a single DistantLight, matching the
    pre-procedural-sky behaviour ("looks like a night").

The HDRI vendoring vs runtime-download question (plan §9 open question 1) is
deferred to the caller: this module just consumes whatever path strings it's
given.
"""
import colorsys

import omni.usd
from pxr import Gf, UsdGeom, UsdLux


DOME_LIGHT_PRIM_PATH = "/World/env_dome"
SUN_LIGHT_PRIM_PATH = "/World/env_sun"
FALLBACK_LIGHT_PRIM_PATH = "/World/default_distant_light"
FALLBACK_LIGHT_INTENSITY = 3000.0
# Defensive intensity when an environment USD is already loaded: half the
# no-env value so authored env lights stay dominant when their textures
# resolve, but the scene is still visible when they don't (some Isaac envs
# reference HDRIs under /NVIDIA/Assets/Skies/ which may not be synced).
FALLBACK_LIGHT_INTENSITY_WITH_ENV = 1500.0


def install_fallback_distant_light(intensity: float = FALLBACK_LIGHT_INTENSITY):
    """Author a single DistantLight at FALLBACK_LIGHT_PRIM_PATH.

    Idempotent — repeated calls overwrite the intensity on the same prim
    rather than stacking lights.
    """
    stage = omni.usd.get_context().get_stage()
    fallback = UsdLux.DistantLight.Define(stage, FALLBACK_LIGHT_PRIM_PATH)
    fallback.CreateIntensityAttr().Set(float(intensity))
    return fallback


def _hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
    return float(r), float(g), float(b)


def setup_random_hdri(rng, config):
    """Install a DomeLight with an HDRI chosen uniformly from config.hdri_paths.

    Returns a meta dict for the "hdri" branch, or None if config.hdri_paths is
    empty. The caller (`setup_random_sky`) dispatches when this returns None.
    Rotation is uniform around world Z (Isaac is Z-up).
    """
    if not config.hdri_paths:
        return None

    hdri_path = str(rng.choice(config.hdri_paths))
    rotation_deg = float(rng.uniform(*config.hdri_rotation_range_deg))
    intensity = float(config.hdri_intensity)

    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
    dome.CreateTextureFileAttr().Set(hdri_path)
    dome.CreateIntensityAttr().Set(intensity)

    xform = UsdGeom.Xformable(dome.GetPrim())
    xform.AddRotateZOp().Set(rotation_deg)

    return {
        "type": "hdri",
        "path": hdri_path,
        "rotation_deg": rotation_deg,
        "intensity": intensity,
    }


def setup_procedural_sky(rng, config):
    """Author a randomized DomeLight (uniform sky colour) + DistantLight ("sun").

    No external assets required — every parameter is sampled from `rng` against
    ranges in `config`. Returns a meta dict capturing the sampled dome + sun
    parameters so the lighting is reproducible and inspectable from disk.

    Sun direction convention: elevation = angle above horizon (90 = zenith),
    azimuth = direction in horizontal plane (0 = +X, 90 = +Y). A DistantLight
    emits along its local -Z by default; we rotate Y by `(90 - elevation)` to
    pitch the beam down from vertical, then rotate Z by `azimuth` to spin it.
    """
    stage = omni.usd.get_context().get_stage()

    # --- dome (uniform sky colour) ---
    dome_r, dome_g, dome_b = _hsv_to_rgb(
        rng.uniform(*config.sky_dome_hue_range),
        rng.uniform(*config.sky_dome_sat_range),
        rng.uniform(*config.sky_dome_val_range),
    )
    dome_intensity = float(rng.uniform(*config.sky_dome_intensity_range))
    dome = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
    dome.CreateColorAttr().Set(Gf.Vec3f(dome_r, dome_g, dome_b))
    dome.CreateIntensityAttr().Set(dome_intensity)

    # --- sun (directional light) ---
    sun_r, sun_g, sun_b = _hsv_to_rgb(
        rng.uniform(*config.sun_hue_range),
        rng.uniform(*config.sun_sat_range),
        rng.uniform(*config.sun_val_range),
    )
    sun_intensity = float(rng.uniform(*config.sun_intensity_range))
    sun_az = float(rng.uniform(*config.sun_azimuth_range_deg))
    sun_el = float(rng.uniform(*config.sun_elevation_range_deg))
    sun = UsdLux.DistantLight.Define(stage, SUN_LIGHT_PRIM_PATH)
    sun.CreateColorAttr().Set(Gf.Vec3f(sun_r, sun_g, sun_b))
    sun.CreateIntensityAttr().Set(sun_intensity)
    xform = UsdGeom.Xformable(sun.GetPrim())
    xform.AddRotateYOp().Set(90.0 - sun_el)
    xform.AddRotateZOp().Set(sun_az)

    return {
        "type": "procedural",
        "dome": {
            "color_rgb": [dome_r, dome_g, dome_b],
            "intensity": dome_intensity,
        },
        "sun": {
            "color_rgb": [sun_r, sun_g, sun_b],
            "intensity": sun_intensity,
            "azimuth_deg": sun_az,
            "elevation_deg": sun_el,
        },
    }


def setup_random_sky(rng, config):
    """Top-level lighting dispatcher for the no-env path.

    Tries HDRI first (if `config.hdri_paths` is populated), falls back to the
    procedural sky (`config.procedural_sky_enabled`, default True), and finally
    to a bare DistantLight if both are disabled. Returns the meta dict for
    `meta.json["scene"]["sky"]`, or None for the bare fallback (matches the
    previous "no HDRI metadata to record" signal).
    """
    spec = setup_random_hdri(rng, config)
    if spec is not None:
        return spec
    if getattr(config, "procedural_sky_enabled", False):
        return setup_procedural_sky(rng, config)
    install_fallback_distant_light()
    return None
