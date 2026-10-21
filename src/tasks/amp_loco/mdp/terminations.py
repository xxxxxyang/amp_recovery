from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.termination_manager import TerminationManager
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class DelayedTerminationManager(TerminationManager):
    """TerminationManager subclass that delays reset for a subset of envs.

    For delay envs, when a termination is triggered the reset signal is
    suppressed and a counter starts incrementing. Once the counter reaches
    ``max_delay_steps``, the reset signal is released and the counter resets.
    """

    def __init__(
        self,
        base: TerminationManager,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        # Steal all internal state from the base manager (avoid re-init).
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask          # (num_envs,) bool
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = max_delay_steps

    def compute(self) -> torch.Tensor:
        super().compute()  # fills _truncated_buf, _terminated_buf

        if self._max_delay_steps <= 0:
            return self._truncated_buf | self._terminated_buf

        # Only consecutive terminations (currently locomotion falls) consume
        # the delay budget. Time-limit truncations still reset immediately.
        delay_and_terminated = self._delay_env_mask & self._terminated_buf
        delay_recovered = self._delay_env_mask & ~self._terminated_buf
        self._delay_counters[delay_and_terminated] += 1

        # Delay envs whose counter hasn't reached threshold: suppress reset.
        not_ready = delay_and_terminated & (self._delay_counters < self._max_delay_steps)
        self._terminated_buf[not_ready] = False

        # Delay envs whose counter reached threshold: allow reset, clear counter.
        ready = delay_and_terminated & (self._delay_counters >= self._max_delay_steps)
        self._delay_counters[ready] = 0

        # Clear the grace-period counter as soon as the robot is no longer
        # triggering a termination. This also clears it on truncation; reset()
        # below provides a second guard for all episode resets.
        self._delay_counters[delay_recovered] = 0

        return self._truncated_buf | self._terminated_buf

    def reset(self, env_ids=None):
        """Clear delay counters for environments starting a new episode."""
        if env_ids is None:
            self._delay_counters.zero_()
        else:
            self._delay_counters[env_ids] = 0
        return super().reset(env_ids)


def stood_up(
    env: ManagerBasedRlEnv,
    head_height: float = 1.2,
    max_speed: float = 0.5,
    hold_steps: int = 25,
) -> torch.Tensor:
    """Succeed after standing height and low root speed hold for a short time."""
    robot = env.scene["robot"]
    head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
    z = robot.data.site_pos_w[:, head_idx, 2]
    speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=-1)
    standing = (z >= head_height) & (speed < max_speed)

    count = getattr(env, "_amp_getup_stand_count", None)
    if count is None:
        count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    count = torch.where(standing, count + 1, torch.zeros_like(count))
    env._amp_getup_stand_count = count  # type: ignore[attr-defined]
    return count >= hold_steps


def unified_locomotion_fall(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Report falls; DelayedTerminationManager grants grace only to Recovery envs."""
    cmd = env.command_manager.get_term("twist")
    return cmd.fallen_mask()


def unified_recovery_timeout(env: ManagerBasedRlEnv, max_recovery_s: float = 5.0) -> torch.Tensor:
    """End an episode if one recovery attempt exceeds its five-second budget."""
    cmd = env.command_manager.get_term("twist")
    return (cmd.phase == cmd.RECOVER) & (cmd.timer >= max_recovery_s)
