"""Getup-specific Viser controls for selecting and resetting the start pose."""

from __future__ import annotations

from typing import Any

from mjlab.viewer.viser.viewer import ViserPlayViewer


_UI_TO_MODE = {
  "Random": "random",
  "Supine": "supine",
  "Prone": "prone",
  "Default-pose supine": "default_supine",
  "Default-pose prone": "default_prone",
  "Fixed frame": "frame",
}
_MODE_TO_UI = {value: key for key, value in _UI_TO_MODE.items()}


class GetupViserPlayViewer(ViserPlayViewer):
  """Viser play viewer with a Getup reset-pose selector."""

  def setup(self) -> None:
    super().setup()
    env = self.env.unwrapped

    with self._server.gui.add_folder("Getup start pose", expand_by_default=True):
      pose_mode = getattr(env, "_amp_getup_reset_pose_mode", "random")
      frame_index = getattr(env, "_amp_getup_reset_frame", None)
      frame_count = getattr(env, "_amp_getup_reset_frame_count", 1)

      self._reset_pose_selector = self._server.gui.add_dropdown(
        "Reset pose",
        options=list(_UI_TO_MODE),
        initial_value=_MODE_TO_UI[pose_mode],
        hint="Applied to the next reset and later automatic resets.",
      )
      self._reset_frame_input = self._server.gui.add_number(
        "Reference frame",
        initial_value=0 if frame_index is None else frame_index,
        min=0,
        max=max(frame_count - 1, 0),
        step=1,
        disabled=pose_mode != "frame",
        hint="Index in the flattened, sorted reference frame pool.",
      )
      self._reset_pose_status = self._server.gui.add_markdown(
        f"Current reset mode: **{pose_mode}**{self._last_reset_frame_text(env)}"
      )
      reset_button = self._server.gui.add_button("Reset to selected pose")

      @self._reset_pose_selector.on_update
      def _(_: Any) -> None:
        mode = _UI_TO_MODE[self._reset_pose_selector.value]
        self._reset_frame_input.disabled = mode != "frame"
        self.request_action("CUSTOM", ("SET_GETUP_RESET_MODE", mode))

      @self._reset_frame_input.on_update
      def _(_: Any) -> None:
        self.request_action(
          "CUSTOM",
          ("SET_GETUP_RESET_FRAME", int(round(self._reset_frame_input.value))),
        )

      @reset_button.on_click
      def _(_: Any) -> None:
        self.request_action(
          "CUSTOM",
          (
            "RESET_GETUP_TO_SELECTION",
            {
              "mode": _UI_TO_MODE[self._reset_pose_selector.value],
              "frame_index": int(round(self._reset_frame_input.value)),
            },
          ),
        )

  def _handle_custom_action(self, action: Any, payload: Any) -> bool:
    if action.name == "CUSTOM" and isinstance(payload, tuple):
      action_name, action_data = payload
    elif action.name == "CUSTOM":
      action_name, action_data = None, payload
    else:
      return super()._handle_custom_action(action, payload)

    env = self.env.unwrapped
    if action_name == "SET_GETUP_RESET_MODE":
      env._amp_getup_reset_pose_mode = action_data
      self._reset_pose_status.content = f"Selected reset mode: **{action_data}**"
      return True
    if action_name == "SET_GETUP_RESET_FRAME":
      env._amp_getup_reset_frame = action_data
      if getattr(env, "_amp_getup_reset_pose_mode", "random") == "frame":
        self._reset_pose_status.content = f"Selected fixed frame: **{action_data}**"
      return True
    if action_name != "RESET_GETUP_TO_SELECTION":
      return super()._handle_custom_action(action, payload)

    try:
      env._amp_getup_reset_pose_mode = action_data["mode"]
      env._amp_getup_reset_frame = action_data["frame_index"]
      self.reset_environment()
      self.pause()
      self._reset_pose_status.content = (
        f"Reset complete: **{action_data['mode']}**"
        + (
          f", frame **{action_data['frame_index']}**"
          if action_data["mode"] == "frame"
          else self._last_reset_frame_text(env)
        )
      )
      self._sync_ui_state()
    except (IndexError, ValueError) as error:
      self._reset_pose_status.content = f"Reset failed: `{error}`"
    return True

  def _last_reset_frame_text(self, env: Any) -> str:
    frames = getattr(env, "_amp_getup_last_reset_frames", None)
    if frames is None or frames.numel() == 0:
      return ""
    env_index = min(self._scene.env_idx, frames.numel() - 1)
    frame_index = int(frames[env_index].item())
    return f", sampled frame **{frame_index}**" if frame_index >= 0 else ""
