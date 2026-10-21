"""Unitree G1 AMP Locomotion environment configurations."""

import os

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_constants import get_spec as _get_g1_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.amp_loco import mdp
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from src.tasks.amp_loco.mdp.unified_command import RecoveryLocomotionCommandCfg

def g1_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  # Keep CCD high enough for stability but avoid Warp OOM from excessive EPA buffers.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  body_names = ("pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link",)
  anchor_name = "torso_link"
  root_name = "pelvis"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Configure motion reset to sample from the entire motion with a delay.
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  # Set motion data path for startup loader and reset.
  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "g1", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  _recovery_dir = os.path.abspath(os.path.join(_motion_base, "Recovery"))

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
 
  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

  

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def g1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = g1_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

  return cfg


def g1_amp_flat_recovery_control_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat AMP baseline with current Recovery data and reset sampling."""
  cfg = g1_amp_flat_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = 4096

  recovery_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "assets", "motions", "g1",
    "amp_unified_v1", "Recovery",
  ))
  cfg.events["init_motion_loader"].params["recovery_dir"] = recovery_dir
  cfg.events["reset_from_motion"].params["recovery_sampling_mode"] = "low_non_upright"
  return cfg


def _get_g1_spec_with_head_site():
  """Return the stock G1 spec with the head site used by SMP success logic."""
  spec = _get_g1_spec()
  torso = spec.body("torso_link")
  if not any(site.name == "head" for site in torso.sites):
    torso.add_site(name="head", pos=(0.0, 0.0, 0.43))
  return spec


def g1_amp_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat G1 AMP get-up task with random Recovery-frame resets."""
  cfg = g1_amp_flat_env_cfg(play=play)

  # The get-up task uses a head site for the same stable-standing condition as SMP.
  cfg.scene.entities["robot"].spec_fn = _get_g1_spec_with_head_site

  # Recovery data is both the reset pool and the expert motion prior.
  motion_dir = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      "..", "..", "..", "..", "assets", "motions", "g1", "amp", "Recovery",
    )
  )
  cfg.events["init_motion_loader"].params.update(
    motion_dir=motion_dir,
    recovery_dir=None,
    delay_reset_env_ratio=0.0,
    max_delay_steps=0,
  )
  cfg.events["reset_from_motion"].params["motion_dir"] = motion_dir
  cfg.events["reset_from_motion"].params.update(
    pose_mode="random",
    frame_index=None,
    default_pose_reset_ratio=0.0 if play else 0.10,
    pose_filters={
      "max_root_height": 0.30,
      "min_abs_forward_z": 0.75,
      "max_root_linear_speed": 0.50,
      "max_root_angular_speed": 1.00,
    },
  )
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_getup_stand_counter,
    mode="reset",
  )

  # Get-up has no velocity command. Keep mild startup domain randomization, but
  # disable locomotion pushes and terrain resampling for this first baseline.
  cfg.commands = {}
  cfg.observations["actor"].terms.pop("command", None)
  cfg.observations["critic"].terms.pop("command", None)
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)
  cfg.curriculum = {}

  # Preserve AMP's height objective and regularization terms without delay masks.
  cfg.rewards = {
    "amp_getup_root_height": RewardTermCfg(
      func=mdp.amp_getup_root_height,
      weight=3.5,
      params={"std": 0.3},
    ),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
  }

  # Both success and the 5-second episode limit are truncations, so the value
  # function bootstraps from the final standing or timeout state.
  cfg.terminations = {
    "stood_up": TerminationTermCfg(
      func=mdp.stood_up,
      time_out=True,
      params={"head_height": 1.2, "max_speed": 0.5, "hold_steps": 25},
    ),
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  cfg.episode_length_s = 5.0

  cfg.metrics = {
    "getup/root_height": MetricsTermCfg(func=mdp.getup_root_height),
    "getup/head_height": MetricsTermCfg(func=mdp.getup_head_height),
    "getup/stood_up_episodes": MetricsTermCfg(
      func=mdp.getup_termination_event,
      params={"term_name": "stood_up"},
    ),
    "getup/time_out_episodes": MetricsTermCfg(
      func=mdp.getup_termination_event,
      params={"term_name": "time_out"},
    ),
  }

  # The flat play config normally uses a very long horizon. Keep the training
  # timeout in play mode so success/timeout behavior can be inspected directly.
  if play:
    cfg.episode_length_s = 5.0

  return cfg


def g1_amp_recovery_loco_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """First unified recovery plus locomotion task, trained on flat ground."""
  cfg = g1_amp_flat_env_cfg(play=play)
  # A single training environment rounds the fixed 6:1:3 cohorts to Recovery
  # only. Match the environment count used by this task's existing runs.
  if not play:
    cfg.scene.num_envs = 4096
  cfg.scene.entities["robot"].spec_fn = _get_g1_spec_with_head_site
  asset_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "g1", "amp_unified_csv2"
  ))
  walk_dir = os.path.join(asset_root, "WalkandRun")
  recovery_dir = os.path.join(asset_root, "Recovery")
  cfg.events["init_motion_loader"].params.update(
    motion_dir=walk_dir,
    recovery_dir=recovery_dir,
    delay_reset_env_ratio=0.6,
    max_delay_steps=250,
    unified_recovery_ratio=0.6,
    unified_stand_ratio=0.1,
    unified_play_phase="random",
  )
  reset = cfg.events["reset_from_motion"]
  reset.func = mdp.reset_unified_recovery_locomotion
  reset.params.update(motion_dir=walk_dir, play_phase="random")
  reset.params.pop("pose_mode", None)
  reset.params.pop("frame_index", None)
  reset.params.pop("pose_filters", None)
  reset.params.pop("default_pose_reset_ratio", None)

  old = cfg.commands["twist"]
  cfg.commands["twist"] = RecoveryLocomotionCommandCfg(
    entity_name=old.entity_name,
    resampling_time_range=old.resampling_time_range,
    rel_standing_envs=old.rel_standing_envs,
    rel_heading_envs=old.rel_heading_envs,
    heading_command=old.heading_command,
    heading_control_stiffness=old.heading_control_stiffness,
    debug_vis=bool(play),
    ranges=old.ranges,
    walkandrun_velocity_stages=None if play else (
      {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
      {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
    ),
  )

  # Keep the original locomotion perturbation for WalkandRun and static
  # environments. Recovery environments receive it only after returning to
  # stable, commanded locomotion so those falls become get-up training samples.
  if play:
    cfg.events.pop("push_robot", None)
  else:
    base_push_cfg = cfg.events["push_robot"]
    cfg.events["push_robot"] = EventTermCfg(
      func=mdp.push_unified_robot,
      mode="interval",
      interval_range_s=base_push_cfg.interval_range_s,
      params={
        **base_push_cfg.params,
        "recovery_push_probability": 0.25,
        "recovery_min_walk_s": 1.0,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )
  # This inherited term only rewards delayed-reset environments; unified
  # recovery uses the unmasked get-up height objective below.
  cfg.rewards.pop("track_root_height", None)
  cfg.rewards["amp_getup_root_height"] = RewardTermCfg(
    # Keep the get-up objective at the same strength as the standalone task.
    func=mdp.amp_getup_root_height, weight=3.5, params={"std": 0.3}
  )
  cfg.rewards["unified_stand_still"] = RewardTermCfg(
    func=mdp.unified_stand_still, weight=1.0, params={"std": 0.25}
  )
  cfg.rewards["default_pose_stand"] = RewardTermCfg(
    func=mdp.unified_default_pose_stand,
    weight=0.5,
    params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["unified_stand_foot_slip"] = RewardTermCfg(
    func=mdp.unified_stand_foot_slip,
    weight=-0.25,
    params={"sensor_name": "feet_ground_contact", "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
  )
  cfg.metrics["unified_recovery_phase_fraction"] = MetricsTermCfg(
    func=mdp.UnifiedRewardEpisodeLogger
  )
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(func=mdp.unified_bad_orientation),
    "bad_base_height": TerminationTermCfg(func=mdp.unified_bad_base_height),
    "recovery_fall": TerminationTermCfg(func=mdp.unified_recovery_fall),
    "recovery_timeout": TerminationTermCfg(
      func=mdp.unified_recovery_timeout, time_out=True, params={"max_recovery_s": 5.0}
    ),
  }
  cfg.episode_length_s = 20.0
  cfg.curriculum = {}
  if play:
    cfg.episode_length_s = 20.0
  return cfg
