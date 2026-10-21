"""Viser controls for evaluating unified recovery and locomotion policies."""

from __future__ import annotations

from typing import Any

import torch

from mjlab.viewer.viser.viewer import ViserPlayViewer


_UI_TO_PHASE = {
  "Recovery (起身后走跑)": "recovery",
  "Locomotion (走跑)": "walk",
  "Standstill (静止)": "stand",
}
_PHASE_TO_UI = {phase: label for label, phase in _UI_TO_PHASE.items()}
_PHASE_NAMES = {
  "recovery": "起身恢复",
  "settle": "起身后站稳",
  "ramp": "起身后渐进走路",
  "hold": "起身后静止保持",
  "stand": "静止",
  "track": "走跑命令跟踪",
}


class RecoveryLocoViserPlayViewer(ViserPlayViewer):
  """Add a phase selector and reset button for the unified task."""

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
      reset_button = self._server.gui.add_button("Reset to selected phase")
      self._phase_status = self._server.gui.add_markdown("")
      self._live_phase_status = self._server.gui.add_markdown("")

      @self._phase_selector.on_update
      def _(_: Any) -> None:
        self._phase_status.content = (
          f"Selected reset: **{self._phase_selector.value}**. "
          "Click the button to apply it."
        )

      @reset_button.on_click
      def _(_) -> None:
        self.request_action(
          "CUSTOM", ("RESET_UNIFIED_PHASE", _UI_TO_PHASE[self._phase_selector.value])
        )

    self._phase_status.content = f"Selected reset: **{_PHASE_TO_UI[initial_phase]}**"
    self._update_live_phase_status()

  def _handle_custom_action(self, action: Any, payload: Any) -> bool:
    if action.name != "CUSTOM":
      return super()._handle_custom_action(action, payload)

    if isinstance(payload, tuple):
      action_name, phase = payload
    else:
      action_name, phase = None, None
    if action_name != "RESET_UNIFIED_PHASE":
      return super()._handle_custom_action(action, payload)

    env = self.env.unwrapped
    if phase not in {"recovery", "walk", "stand"}:
      self._phase_status.content = f"Reset failed: unsupported phase `{phase}`"
      return True

    groups = env._amp_unified_reset_groups
    post_modes = env._amp_unified_post_recovery_modes
    command = env.command_manager.get_term("twist")
    group_id = {"recovery": 0, "stand": 1, "walk": 2}[phase]
    groups.fill_(group_id)
    post_modes.fill_(command.RAMP if phase == "recovery" else command.HOLD)
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
        f"Reset complete: **{_PHASE_TO_UI[phase]}**. Press Play to run the policy."
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
    self._live_phase_status.content = (
      f"Current policy phase: **{_PHASE_NAMES.get(phase_name, phase_name)}**  \n"
      f"Velocity command (x, y, yaw): `{vx:+.2f}, {vy:+.2f}, {wz:+.2f}` m/s, m/s, rad/s"
    )
