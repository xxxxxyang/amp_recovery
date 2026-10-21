"""Play AMP NPZ motions and raw G1 motion CSVs in a Viser viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.viewer.viser.scene import ViserMujocoScene


def _collect(paths: list[str]) -> list[Path]:
  files: list[Path] = []
  for item in paths:
    path = Path(item).expanduser().resolve()
    if path.is_dir():
      files.extend(path.rglob("*.npz"))
      files.extend(path.rglob("*.csv"))
    elif path.is_file() and path.suffix.lower() in {".npz", ".csv"}:
      files.append(path)
    else:
      raise FileNotFoundError(f"Motion file or directory not found: {path}")
  unique = sorted(set(files))
  if not unique:
    raise FileNotFoundError("No .npz or .csv motions found")
  return unique


def _load(path: Path, csv_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  if path.suffix.lower() == ".csv":
    # Retargeted G1 CSV format: root xyz, quaternion xyzw, then 29 joint angles.
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
      raw = raw[None, :]
    if raw.shape[1] != 36:
      raise ValueError(f"{path}: expected 36 CSV columns, got {raw.shape[1]}")
    root_pos = raw[:, :3]
    root_quat = raw[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    joint_pos = raw[:, 7:]
    return root_pos, root_quat, joint_pos, csv_fps

  with np.load(path, allow_pickle=False) as motion:
    joint_pos = np.asarray(motion["joint_pos"])
    body_pos = np.asarray(motion["body_pos_w"])
    body_quat = np.asarray(motion["body_quat_w"])
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
  if joint_pos.ndim != 2 or body_pos.ndim != 3 or body_quat.ndim != 3:
    raise ValueError(f"{path}: unexpected AMP NPZ array shapes")
  if body_pos.shape[1] == 0 or body_quat.shape[1] == 0 or fps <= 0:
    raise ValueError(f"{path}: missing root poses or has invalid fps={fps}")
  count = min(len(joint_pos), len(body_pos), len(body_quat))
  return body_pos[:count, 0], body_quat[:count, 0], joint_pos[:count], fps


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("motions", nargs="+", help="CSV/NPZ files and/or directories")
  parser.add_argument("--csv-fps", type=float, default=30.0, help="source CSV frame rate")
  parser.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
  args = parser.parse_args()
  if args.csv_fps <= 0 or args.speed <= 0:
    parser.error("--csv-fps and --speed must be greater than zero")

  paths = _collect(args.motions)
  clips = [_load(path, args.csv_fps) for path in paths]
  print(f"Loaded {len(paths)} motions:")
  for path, (root_pos, _, joint_pos, fps) in zip(paths, clips, strict=True):
    print(f"  {path.name}: {len(joint_pos)} frames, {fps:g} fps")

  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device="cpu")
  sim = Simulation(
    num_envs=1,
    cfg=SimulationCfg(),
    model=scene.compile(),
    device="cpu",
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  model = sim.mj_model
  data = sim.mj_data
  free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  hinge_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
  if len(free_ids) != 1:
    raise RuntimeError(f"Expected one free root joint, found {len(free_ids)}")
  root_qpos = int(model.jnt_qposadr[free_ids[0]])
  joint_qpos = model.jnt_qposadr[hinge_ids]
  if any(joint_pos.shape[1] != len(joint_qpos) for _, _, joint_pos, _ in clips):
    raise ValueError(f"Motion joint count must match model hinge count ({len(joint_qpos)})")

  server = viser.ViserServer(label="AMP / f2s2 motion playback")
  viewer = ViserMujocoScene.create(server, model, num_envs=1)
  viewer.debug_visualization_enabled = True
  selected = {"index": 0, "frame": 0, "last_tick": time.perf_counter()}
  playing = {"value": True}
  speed = {"value": args.speed}

  with server.gui.add_folder("Motion playback", expand_by_default=True):
    clip_select = server.gui.add_dropdown(
      "Motion", options=[p.name for p in paths], initial_value=paths[0].name
    )
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=len(clips[0][2]) - 1, step=1, initial_value=0
    )
    play_button = server.gui.add_button("Pause" if playing["value"] else "Play")
    speed_slider = server.gui.add_slider(
      "Speed", min=0.1, max=3.0, step=0.1, initial_value=args.speed
    )
    status = server.gui.add_markdown("")

  @clip_select.on_update
  def _(_event) -> None:
    selected["index"] = [p.name for p in paths].index(clip_select.value)
    selected["frame"] = 0
    frame_slider.max = len(clips[selected["index"]][2]) - 1
    frame_slider.value = 0
    selected["last_tick"] = time.perf_counter()

  @frame_slider.on_update
  def _(_event) -> None:
    selected["frame"] = int(frame_slider.value)

  @play_button.on_click
  def _(_event) -> None:
    playing["value"] = not playing["value"]
    play_button.label = "Pause" if playing["value"] else "Play"
    selected["last_tick"] = time.perf_counter()

  @speed_slider.on_update
  def _(_event) -> None:
    speed["value"] = float(speed_slider.value)

  print("Viser server running. Open the printed URL in a browser.")
  try:
    while True:
      index = selected["index"]
      root_pos, root_quat, joint_pos, fps = clips[index]
      frame = min(selected["frame"], len(joint_pos) - 1)
      data.qpos[root_qpos : root_qpos + 3] = root_pos[frame]
      data.qpos[root_qpos + 3 : root_qpos + 7] = root_quat[frame]
      data.qpos[joint_qpos] = joint_pos[frame]
      mujoco.mj_forward(model, data)
      viewer.update_from_mjdata(data)
      viewer.refresh_visualization()
      status.content = (
        f"**{paths[index].name}** · frame {frame + 1}/{len(joint_pos)} · "
        f"{fps:g} fps · {len(joint_pos) / fps:.1f} s"
      )

      now = time.perf_counter()
      if playing["value"] and now - selected["last_tick"] >= 1.0 / (fps * speed["value"]):
        selected["frame"] = (frame + 1) % len(joint_pos)
        frame_slider.value = selected["frame"]
        selected["last_tick"] = now
      time.sleep(1.0 / 60.0)
  except KeyboardInterrupt:
    print("Shutting down Viser playback.")


if __name__ == "__main__":
  main()
