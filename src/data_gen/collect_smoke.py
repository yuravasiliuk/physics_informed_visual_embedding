"""Phase-0 smoke test for Isaac Sim data collection.

Renders one 16-frame free-object drop scene with three primitives, captures
RGB / depth / normals / instance-seg per frame, and logs per-body state to JSON.

Step 1 of the implementation order in plans/synthetic-data-collection.md.
Deliberately minimal: no asset library, no domain randomization, no Parquet,
no WebDataset shards. The point is to verify the pipeline end-to-end on
the simplest scene that exercises every output modality.

Run from the Isaac Sim Python environment, e.g.:
    ${ISAAC_SIM}/python.sh src/data_gen/collect_smoke.py --out out/clip_0000
"""
from isaacsim import SimulationApp

# SimulationApp must be constructed before any other omni / isaac import.
simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import argparse
import json
from pathlib import Path

import os
import cv2
import numpy as np

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, DynamicSphere

FRAME_COUNT = 50
RESOLUTION = (640, 480)
CAMERA_POSITION = (1.2, 1.2, 0.8)
CAMERA_LOOK_AT = (0.0, 0.0, 0.1)
WRT_CAMERA_FRAME = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, default=Path("result/clip_0000"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_scene(world, rng):
    world.scene.add_default_ground_plane()

    def spawn_pos():
        return rng.uniform([-0.2, -0.2, 0.5], [0.2, 0.2, 0.9])

    bodies = [
        world.scene.add(DynamicCuboid(
            prim_path="/World/cube", name="cube",
            position=spawn_pos(), scale=[0.1, 0.1, 0.1],
            color=np.array([0.8, 0.2, 0.2]),
        )),
        world.scene.add(DynamicSphere(
            prim_path="/World/sphere", name="sphere",
            position=spawn_pos(), radius=0.06,
            color=np.array([0.2, 0.8, 0.2]),
        )),
        world.scene.add(DynamicCylinder(
            prim_path="/World/cylinder", name="cylinder",
            position=spawn_pos(), radius=0.05, height=0.12,
            color=np.array([0.2, 0.2, 0.8]),
        )),
    ]
    return bodies


def make_annotators():
    camera = rep.create.camera(position=CAMERA_POSITION, look_at=CAMERA_LOOK_AT)
    rp = rep.create.render_product(camera, RESOLUTION)
    annots = {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
        "depth": rep.AnnotatorRegistry.get_annotator("distance_to_camera"),
        "normals": rep.AnnotatorRegistry.get_annotator("normals"),
        "seg": rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast"),
    }
    for a in annots.values():
        a.attach(rp)
    return annots


def capture(annots):
    """Pull one frame from every annotator and return it as plain arrays.

    Aggregation (stack frames, encode video, write .npy) is the caller's job.
    """
    rgb = annots["rgb"].get_data()           # (H, W, 4) uint8
    depth = annots["depth"].get_data()       # (H, W) float32
    normals = annots["normals"].get_data()   # (H, W, 4) float32 in [-1, 1]
    seg = annots["seg"].get_data()           # {"data": uint32 (H,W), "info": {...}} or ndarray

    seg_array = seg["data"] if isinstance(seg, dict) else seg
    seg_info = seg["info"] if isinstance(seg, dict) else None

    return {
        "rgb": rgb[..., :3].astype(np.uint8),
        "depth": depth.astype(np.float16),
        "normals": normals[..., :3].astype(np.float16),
        "seg": seg_array.astype(np.uint32),
        "seg_info": seg_info,
    }


def world_to_camera_frame(camera_pos, look_at, world_up=(0.0, 0.0, 1.0)):
    """Return (R_camera_world, camera_pos_world) for a USD/OpenGL camera.

    USD camera convention: +X right, +Y up, -Z forward (camera looks down -Z).
    R_camera_world rotates a *vector* expressed in world coords into camera coords.
    A *point* transforms as p_cam = R_camera_world @ (p_world - camera_pos_world).
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward = look_at - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Columns are the camera basis expressed in world coords (X right, Y up, Z back).
    R_world_camera = np.stack([right, up, -forward], axis=1)
    R_camera_world = R_world_camera.T
    return R_camera_world, camera_pos


def com_world_to_camera(com_world, R_camera_world, camera_pos_world):
    """Convert a CoM 3D point from world frame to camera frame."""
    return R_camera_world @ (np.asarray(com_world, dtype=np.float64) - camera_pos_world)


def log_state(bodies, frame_idx, records, R_camera_world=None, camera_pos_world=None):
    for body in bodies:
        pos, quat = body.get_world_pose()
        com_world = np.asarray(pos)  # for uniform-density primitives, prim origin == CoM
        entry = {
            "frame": frame_idx,
            "body": body.name,
            "quat_world_wxyz": np.asarray(quat).tolist(),
            "lin_vel_world": np.asarray(body.get_linear_velocity()).tolist(),
            "ang_vel_world": np.asarray(body.get_angular_velocity()).tolist(),
            "mass": float(body.get_mass()),
        }
        if WRT_CAMERA_FRAME:
            entry["com_camera"] = com_world_to_camera(
                com_world, R_camera_world, camera_pos_world
            ).tolist()
        else:
            entry["com_world"] = com_world.tolist()
        records.append(entry)


def main():
    args = parse_args()

    save_dir = os.path.abspath(args.result)
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    world = World(stage_units_in_meters=1.0)
    bodies = build_scene(world, rng)
    annots = make_annotators()

    R_camera_world, camera_pos_world = world_to_camera_frame(CAMERA_POSITION, CAMERA_LOOK_AT)

    world.reset()
    for b in bodies:
        b.set_angular_velocity(rng.uniform(-1.0, 1.0, size=3))

    state_records = []
    frames = []
    for frame_idx in range(FRAME_COUNT):
        world.step(render=True)
        rep.orchestrator.step()  # flush annotator outputs for this frame
        frames.append(capture(annots))
        log_state(bodies, frame_idx, state_records, R_camera_world, camera_pos_world)

    rgb_stack = np.stack([f["rgb"] for f in frames])           # (T, H, W, 3) uint8
    depth_stack = np.stack([f["depth"] for f in frames])       # (T, H, W) float16
    normals_stack = np.stack([f["normals"] for f in frames])   # (T, H, W, 3) float16
    seg_stack = np.stack([f["seg"] for f in frames])           # (T, H, W) uint32

    height, width = rgb_stack.shape[1:3]

    video_file = os.path.join(save_dir, 'rgb.mp4')
    writer = cv2.VideoWriter(
        video_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (width, height),
    )
    for frame in rgb_stack:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    np.save(os.path.join(save_dir, "depth.npy"), depth_stack)
    np.save(os.path.join(save_dir, "normals.npy"), normals_stack)
    np.save(os.path.join(save_dir, "seg.npy"), seg_stack)
    if frames[0]["seg_info"] is not None:
        with open(os.path.join(save_dir, "seg_id_to_prim.json"), "w") as f:
            json.dump(frames[0]["seg_info"], f, indent=2, default=str)

    with open(os.path.join(save_dir, "state.json"), "w") as f:
        json.dump(state_records, f, indent=2)

    meta = {
        "frame_count": FRAME_COUNT,
        "resolution": list(RESOLUTION),
        "seed": args.seed,
        "bodies": [b.name for b in bodies],
        "camera_position_world": list(CAMERA_POSITION),
        "camera_look_at_world": list(CAMERA_LOOK_AT),
        "wrt_camera_frame": WRT_CAMERA_FRAME,
        "R_camera_world": R_camera_world.tolist(),
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    simulation_app.close()


if __name__ == "__main__":
    main()
