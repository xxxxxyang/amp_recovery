"""Play an AMP motion NPZ in the native MuJoCo viewer."""

from pathlib import Path
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


def main(
    motion_file: str = "src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz",
    speed: float = 1.0,
    loop: bool = False,
) -> None:
    """Play the recorded root and joint poses in a MuJoCo window."""
    path = Path(motion_file).expanduser().resolve()
    if speed <= 0:
        raise ValueError("--speed must be greater than zero")

    with np.load(path, allow_pickle=False) as motion:
        joint_pos = motion["joint_pos"]
        body_pos = motion["body_pos_w"]
        body_quat = motion["body_quat_w"]
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])

    if joint_pos.ndim != 2 or body_pos.ndim != 3 or body_quat.ndim != 3:
        raise ValueError(f"Unexpected motion array shapes in {path}")
    if fps <= 0:
        raise ValueError(f"Invalid fps value {fps} in {path}")

    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device="cpu")
    model = scene.compile()
    sim = Simulation(
        num_envs=1,
        cfg=SimulationCfg(),
        model=model,
        device="cpu",
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)

    mujoco_model = sim.mj_model
    data = sim.mj_data
    free_joint_ids = np.flatnonzero(mujoco_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joint_ids) != 1:
        raise RuntimeError(f"Expected one free root joint, found {len(free_joint_ids)}")
    root_qpos = int(mujoco_model.jnt_qposadr[free_joint_ids[0]])
    hinge_ids = np.flatnonzero(mujoco_model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    joint_qpos = mujoco_model.jnt_qposadr[hinge_ids]
    if len(joint_qpos) != joint_pos.shape[1]:
        raise ValueError(
            f"Motion has {joint_pos.shape[1]} joints, model has {len(joint_qpos)} hinges"
        )
    if body_pos.shape[1] == 0 or body_quat.shape[1] == 0:
        raise ValueError("Motion contains no body poses")

    frame_count = min(len(joint_pos), len(body_pos), len(body_quat))
    frame_period = 1.0 / (fps * speed)
    print(f"Playing {path} ({frame_count} frames at {fps:g} fps)")
    with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
        while viewer.is_running():
            for frame in range(frame_count):
                if not viewer.is_running():
                    break
                start = time.perf_counter()
                data.qpos[root_qpos : root_qpos + 3] = body_pos[frame, 0]
                data.qpos[root_qpos + 3 : root_qpos + 7] = body_quat[frame, 0]
                data.qpos[joint_qpos] = joint_pos[frame]
                mujoco.mj_forward(mujoco_model, data)
                viewer.sync()
                remaining = frame_period - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
            if not loop:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "motion_file",
        nargs="?",
        default="src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz",
        help="motion NPZ file to play",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument("--loop", action="store_true", help="loop playback")
    args = parser.parse_args()
    main(args.motion_file, speed=args.speed, loop=args.loop)
