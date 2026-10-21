from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def getup_root_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current root height in world coordinates."""
    return env.scene["robot"].data.body_link_pos_w[:, 0, 2]


def getup_head_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current height of the head site used by the success condition."""
    robot = env.scene["robot"]
    head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
    return robot.data.site_pos_w[:, head_idx, 2]


def getup_termination_event(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
    """Expose a terminal event as an episode fraction on old/new mjlab APIs.

    MetricsManager averages per-step values over the episode. Scaling the one-step
    event by the current episode length makes that average 1 for the event and 0
    otherwise, which reports a success/timeout fraction without API-specific
    reduction settings.
    """
    if term_name not in env.termination_manager.active_terms:
        return torch.zeros(env.num_envs, device=env.device)
    event = env.termination_manager.get_term(term_name).float()
    return event * env.episode_length_buf.float()


def mean_delay_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean number of delay steps for environments in delayed termination.

  This metric is useful for monitoring the delay duration when using delayed termination.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  tm = env.termination_manager
  delay_counters = getattr(tm, "_delay_counters", None)
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  
  if delay_env_mask is not None and delay_counters is not None:
    total_delay_steps = torch.sum(delay_counters.float())
    total_delay_envs = torch.sum(delay_env_mask.float())
    mean_delay = total_delay_steps / (total_delay_envs + 1e-8)  # Avoid division by zero.
    return mean_delay.expand(env.num_envs)  # Return same mean for all envs for easier logging.
  else:
    return torch.zeros(env.num_envs, device=env.device)


class UnifiedRewardEpisodeLogger:
    """Log environment task returns by the phase where each reward accrued.

    RewardManager still owns and optimizes the original reward terms. This
    observer only sums its already-computed per-term reward rates and writes
    episode totals to the log, so it cannot change the policy reward.
    """

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._env = env
        self._episode_totals = {
            "getup": torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
            "walkandrun": torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
        }

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        command = env.command_manager.get_term("twist")
        phase = command.phase
        step_task_reward = env.reward_manager._step_reward.sum(dim=-1) * env.step_dt
        masks = {
            "getup": phase == command.RECOVER,
            # Settling, zero-speed holds, and the dedicated stand cohort are
            # all part of the locomotion task's zero-velocity target.
            "walkandrun": phase != command.RECOVER,
        }
        for name, mask in masks.items():
            self._episode_totals[name] += step_task_reward * mask
        return masks["getup"].float()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        else:
            ids = env_ids.long()
        if ids.numel() == 0:
            return
        log = self._env.extras.setdefault("log", {})
        log["Episode_reward_getup"] = self._episode_totals["getup"][ids].mean().item()
        log["Episode_reward_walkandrun"] = self._episode_totals["walkandrun"][ids].mean().item()
        for totals in self._episode_totals.values():
            totals[ids] = 0.0
