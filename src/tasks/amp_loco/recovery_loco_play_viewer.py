"""Viser controls for evaluating unified recovery and locomotion policies."""

from __future__ import annotations

import math
from typing import Any

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.viewer.viser.viewer import ViserPlayViewer


_UI_TO_PHASE = {
  "Recovery (起身后走跑)": "recovery",
  "Locomotion (走跑)": "walk",
  "Standstill (静止)": "stand",
}
_PHASE_TO_UI = {phase: label for label, phase in _UI_TO_PHASE.items()}
_POST_RECOVERY_OPTIONS = {
  "Hold (持续零命令)": "hold",
  "Locomotion (渐进速度命令)": "locomotion",
}
_POST_RECOVERY_LABELS = {value: label for label, value in _POST_RECOVERY_OPTIONS.items()}
_PHASE_NAMES = {
  "recover": "起身恢复",
  "settle": "起身后站稳",
  "ramp": "起身后渐进走路",
  "hold": "起身后静止保持",
  "stand": "静止",
  "track": "走跑命令跟踪",
}


class RecoveryLocoViserPlayViewer(ViserPlayViewer):
  """Add reset, phase status, and manual disturbance controls."""

  def setup(self) -> None:
    super().setup()
    env = self.env.unwrapped
    groups = getattr(env, "_amp_unified_reset_groups", None)
    if groups is None:
      raise RuntimeError("Unified reset groups were not initialized before play.")

    current_group = int(groups[min(self._scene.env_idx, groups.numel() - 1)].item())
    initial_phase = getattr(env, "_amp_unified_play_phase", "random")
    if initial_phase == "random":
      initial_phase = {0: "recovery", 1: "stand", 2: "walk"}[current_group]

    with self._server.gui.add_folder("Recovery / locomotion", expand_by_default=True):
      self._phase_selector = self._server.gui.add_dropdown(
        "Reset phase",
        options=list(_UI_TO_PHASE),
        initial_value=_PHASE_TO_UI.get(initial_phase, _PHASE_TO_UI["recovery"]),
        hint="Choose the state used by the next reset. Recovery continues into its configured post-get-up behavior.",
      )
      command = env.command_manager.get_term("twist")
      selected_env_idx = min(self._scene.env_idx, env.num_envs - 1)
      initial_post_mode = env._amp_unified_post_recovery_modes[selected_env_idx]
      initial_post_recovery = (
        "Locomotion (渐进速度命令)"
        if int(initial_post_mode.item()) == command.RAMP
        else "Hold (持续零命令)"
      )
      self._post_recovery_selector = self._server.gui.add_dropdown(
        "After get-up",
        options=list(_POST_RECOVERY_OPTIONS),
        initial_value=initial_post_recovery,
        hint="Used when the reset phase is Recovery. Hold keeps the command at zero; Locomotion ramps into velocity tracking.",
      )
      reset_button = self._server.gui.add_button("Reset to selected phase")
      self._push_strength_slider = self._server.gui.add_slider(
        "Push strength × training push",
        min=0.2,
        max=4.0,
        step=0.1,
        initial_value=1.5,
        hint="Scales the velocity and rotation ranges used by the manual push.",
      )
      push_button = self._server.gui.add_button("Push robot (selected environment)")
      self._phase_status = self._server.gui.add_markdown("")
      self._disturbance_status = self._server.gui.add_markdown("")
      self._live_phase_status = self._server.gui.add_markdown("")

      @self._phase_selector.on_update
      def _(_: Any) -> None:
        self._phase_status.content = (
          f"Selected reset: **{self._phase_selector.value}**. "
          f"After get-up: **{self._post_recovery_selector.value}**. "
          "Click the button to apply it."
        )

      @self._post_recovery_selector.on_update
      def _(_) -> None:
        self._phase_status.content = (
          f"Selected reset: **{self._phase_selector.value}**. "
          f"After get-up: **{self._post_recovery_selector.value}**. "
          "Click the button to apply it."
        )

      @reset_button.on_click
      def _(_) -> None:
        self.request_action(
          "CUSTOM",
          (
            "RESET_UNIFIED_PHASE",
            (
              _UI_TO_PHASE[self._phase_selector.value],
              _POST_RECOVERY_OPTIONS[self._post_recovery_selector.value],
            ),
          ),
        )

      @push_button.on_click
      def _(_) -> None:
        self.request_action(
          "CUSTOM", ("APPLY_DISTURBANCE", float(self._push_strength_slider.value))
        )

      self._phase_status.content = (
        f"Selected reset: **{_PHASE_TO_UI[initial_phase]}**. "
        f"After get-up: **{initial_post_recovery}**."
      )
    self._disturbance_status.content = (
      "Set the slider, then click to apply one push_robot disturbance."
    )
    self._update_live_phase_status()

  def _handle_custom_action(self, action: Any, payload: Any) -> bool:
    if action.name != "CUSTOM":
      return super()._handle_custom_action(action, payload)

    if isinstance(payload, tuple):
      action_name, action_value = payload
    else:
      action_name, action_value = None, None
    if action_name == "APPLY_DISTURBANCE":
      env = self.env.unwrapped
      env_index = min(self._scene.env_idx, env.num_envs - 1)
      env_ids = torch.tensor([env_index], dtype=torch.long, device=env.device)
      strength = float(action_value)
      envs_mdp.push_by_setting_velocity(
        env,
        env_ids,
        velocity_range={
          "x": (-1.0 * strength, 1.0 * strength),
          "y": (-0.5 * strength, 0.5 * strength),
          "z": (-0.4 * strength, 0.4 * strength),
          "roll": (-0.52 * strength, 0.52 * strength),
          "pitch": (-0.52 * strength, 0.52 * strength),
          "yaw": (-0.78 * strength, 0.78 * strength),
        },
        asset_cfg=SceneEntityCfg("robot"),
      )
      self._disturbance_status.content = (
        f"Applied push to environment **{env_index}** at "
        f"**{strength:.1f}×** training push strength."
      )
      return True
    if action_name != "RESET_UNIFIED_PHASE":
      return super()._handle_custom_action(action, payload)

    env = self.env.unwrapped
    if isinstance(action_value, tuple):
      phase, post_recovery = action_value
    else:
      phase, post_recovery = action_value, "locomotion"
    if phase not in {"recovery", "walk", "stand"}:
      self._phase_status.content = f"Reset failed: unsupported phase `{phase}`"
      return True
    if post_recovery not in _POST_RECOVERY_OPTIONS.values():
      self._phase_status.content = (
        f"Reset failed: unsupported after-get-up behavior `{post_recovery}`"
      )
      return True

    groups = env._amp_unified_reset_groups
    post_modes = env._amp_unified_post_recovery_modes
    command = env.command_manager.get_term("twist")
    group_id = {"recovery": 0, "stand": 1, "walk": 2}[phase]
    groups.fill_(group_id)
    post_modes.fill_(
      command.RAMP if phase == "recovery" and post_recovery == "locomotion"
      else command.HOLD
    )
    env._amp_unified_play_phase = phase

    # Keep delayed termination aligned with the selected evaluation cohort.
    termination_manager = env.termination_manager
    delay_mask = getattr(termination_manager, "_delay_env_mask", None)
    if delay_mask is not None:
      delay_mask.copy_(groups == 0)
      delay_counters = getattr(termination_manager, "_delay_counters", None)
      if delay_counters is not None:
        delay_counters.zero_()

    # A locomotion evaluation must receive a moving command, not the
    # low-probability standing command used during training.
    if phase == "walk":
      command.cfg.rel_standing_envs = 0.0

    try:
      self.reset_environment()
      self.pause()
      self._phase_status.content = (
        f"Reset complete: **{_PHASE_TO_UI[phase]}**; after get-up: "
        f"**{_POST_RECOVERY_LABELS[post_recovery]}**. Press Play to run the policy."
      )
      self._update_live_phase_status()
      self._sync_ui_state()
    except (IndexError, ValueError, RuntimeError) as error:
      self._phase_status.content = f"Reset failed: `{error}`"
    return True

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    self._update_live_phase_status()

  def _update_live_phase_status(self) -> None:
    if not hasattr(self, "_live_phase_status"):
      return
    env = self.env.unwrapped
    command = env.command_manager.get_term("twist")
    env_idx = min(self._scene.env_idx, command.phase.numel() - 1)
    phase_code = int(command.phase[env_idx].item())
    phase_name = next(
      (
        name
        for name in ("recover", "settle", "ramp", "hold", "stand", "track")
        if getattr(command, name.upper(), None) == phase_code
      ),
      "unknown",
    )
    vx, vy, wz = command.vel_command_b[env_idx].detach().cpu().tolist()
    details = []
    if phase_code == command.RECOVER:
      head_height = float(command._head_height()[env_idx].item())
      root_height = float(command.robot.data.root_link_pos_w[env_idx, 2].item())
      gravity_up = -command.robot.data.projected_gravity_b[env_idx, 2]
      tilt_deg = float(torch.rad2deg(torch.acos(gravity_up.clamp(-1.0, 1.0))).item())
      root_speed = float(
        torch.linalg.vector_norm(
          command.robot.data.root_link_lin_vel_w[env_idx]
        ).item()
      )
      stand_count = int(command.stand_count[env_idx].item())
      details.append(
        f"Get-up check: head `{head_height:.2f}/{command.cfg.getup_head_height:.2f} m`, "
        f"root `{root_height:.2f}/{command.cfg.getup_root_height:.2f} m`, "
        f"tilt `{tilt_deg:.0f}/≤{math.degrees(command.cfg.getup_max_tilt):.0f}°`, "
        f"speed `{root_speed:.2f}/<{command.cfg.getup_max_speed:.2f} m/s`, "
        f"stable `{stand_count}/{command.cfg.getup_hold_steps} steps`"
      )
    elif phase_code in (command.SETTLE, command.RAMP):
      phase_duration = 0.75
      details.append(
        f"Phase timer: `{float(command.timer[env_idx].item()):.2f}/{phase_duration:.2f} s`"
      )
    self._live_phase_status.content = "  \n".join((
      f"Environment **{env_idx}** — policy phase: **{_PHASE_NAMES.get(phase_name, phase_name)}**",
      f"Velocity command (x, y, yaw): `{vx:+.2f}, {vy:+.2f}, {wz:+.2f}` m/s, m/s, rad/s",
      *details,
    ))
