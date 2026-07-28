"""Run: python -m worldlines.demo   (from src/). Synth -> encode -> verify round-trip."""
import numpy as np
from .synth import make_synth_clip
from .encode import encode_clip, EncodeConfig
from .project import reproject_error


def run(scheme="gravity", degree=4, moving_camera=True):
    raw = make_synth_clip(n_bodies=3, moving_camera=moving_camera, seed=0)
    wl = encode_clip(raw, EncodeConfig(scheme=scheme, degree=degree))
    err = reproject_error(wl, raw)
    print(f"scheme={scheme} degree={degree} moving_camera={moving_camera} ego.static={wl.ego.static}")
    for ln in wl.lines:
        print(f"  {ln.name}: pos_err={ln.pos_err:.2e} rot_err={ln.rot_err:.2e} "
              f"reproj={err[ln.name]:.2e} mass_ratio={ln.intrinsics['mass_ratio']:.2f} "
              f"com/size={ln.intrinsics['com_offset_over_size']:.3f}")
    return wl, err


if __name__ == "__main__":
    # ballistic motion is exactly degree-2 in position: deg>=2 must round-trip to ~0.
    run("gravity", 4, True)
    run("metric", 2, False)
