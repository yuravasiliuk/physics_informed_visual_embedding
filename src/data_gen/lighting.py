"""HDRI environment lighting (step 5.4 of plans/scene-diversification.md).

Sets up a USD DomeLight with a randomly-rotated HDRI texture. If
`config.hdri_paths` is empty (the default), a fallback DistantLight is
installed instead so the scene is still lit — matching the "single distant
light, looks like a night" behavior we used to get from Isaac's
`default_environment.usd` before `build_scene` switched to a procedural
GroundPlane (which carries no embedded lights). Populate `config.hdri_paths`
with file paths or Isaac nucleus URIs (for example
"omniverse://localhost/NVIDIA/Assets/Skies/Clear/sunset_4k.hdr") to enable
lighting randomization.

The HDRI vendoring vs runtime-download question (plan §9 open question 1) is
deferred to the caller: this module just consumes whatever path strings it's
given.
"""
import omni.usd
from pxr import UsdGeom, UsdLux


DOME_LIGHT_PRIM_PATH = "/World/env_dome"
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


def setup_random_hdri(rng, config):
    """Install a DomeLight with an HDRI chosen uniformly from config.hdri_paths.

    Returns a dict for meta.json["scene"]["hdri"], or None if config.hdri_paths
    is empty — in which case a fallback DistantLight is authored so the scene
    is still lit. The None return just signals "no HDRI metadata to record."
    Rotation is uniform in config.hdri_rotation_range_deg around the world
    Z-axis (azimuthal — Isaac is Z-up).
    """
    if not config.hdri_paths:
        install_fallback_distant_light()
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
        "path": hdri_path,
        "rotation_deg": rotation_deg,
        "intensity": intensity,
    }
