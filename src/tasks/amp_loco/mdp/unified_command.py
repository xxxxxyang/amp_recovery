from __future__ import annotations

import math

import torch

from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


class RecoveryLocomotionCommand(UniformVelocityCommand):
    """Velocity command with recovery, settling, standing, and locomotion phases."""

    RECOVER, SETTLE, HOLD, RAMP, TRACK, STAND = range(6)

    def __init__(self, cfg: "RecoveryLocomotionCommandCfg", env):
        super().__init__(cfg, env)
        self.env = env
        self.phase = torch.full((self.num_envs,), self.STAND, dtype=torch.long, device=self.device)
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.post_recovery_mode = torch.full_like(self.phase, self.HOLD)
        self.stand_count = torch.zeros_like(self.phase)
        self.ramp_target = torch.zeros_like(self.vel_command_b)

    def _head_height(self):
        robot = self.robot
        idx = robot.find_sites(["head"], preserve_order=True)[0][0]
        return robot.data.site_pos_w[:, idx, 2]

    def fallen_mask(self) -> torch.Tensor:
        """Use the original locomotion fall thresholds: tilt >70 deg or root <0.5m."""
        gravity_up = -self.robot.data.projected_gravity_b[:, 2]
        tilt = torch.acos(gravity_up.clamp(-1.0, 1.0))
        root_height = self.robot.data.root_link_pos_w[:, 2]
        return (root_height < 0.5) | (tilt > math.radians(70.0))

    def reset_phases(self, env_ids: torch.Tensor, phases: torch.Tensor, post_mode: torch.Tensor | None = None):
        self.phase[env_ids] = phases
        self.timer[env_ids] = 0.0
        self.stand_count[env_ids] = 0
        if post_mode is not None:
            self.post_recovery_mode[env_ids] = post_mode
        self.vel_command_b[env_ids] = 0.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        zero = (self.phase[env_ids] == self.RECOVER) | (self.phase[env_ids] == self.SETTLE) | (self.phase[env_ids] == self.HOLD) | (self.phase[env_ids] == self.STAND)
        self.vel_command_b[env_ids[zero]] = 0.0

    def compute(self, dt: float) -> None:
        super().compute(dt)
        phase = self.phase
        self.timer += dt

        # Recovery-cohort environments keep a delayed-reset policy even after
        # the first successful get-up. If they fall again while holding or
        # locomoting, start another recovery attempt instead of applying the
        # ordinary locomotion fall termination. The recovery timeout below is
        # the final reset condition for that attempt.
        reset_groups = getattr(self.env, "_amp_unified_reset_groups", None)
        if reset_groups is not None and reset_groups.numel() == self.num_envs:
            recovery_cohort = reset_groups == 0
            fallen_again = recovery_cohort & (phase != self.RECOVER) & self.fallen_mask()
            self.phase[fallen_again] = self.RECOVER
            self.timer[fallen_again] = 0.0
            self.stand_count[fallen_again] = 0
            self.vel_command_b[fallen_again] = 0.0

        head = self._head_height()
        speed = torch.linalg.vector_norm(self.robot.data.root_link_lin_vel_w, dim=-1)

        recovering = phase == self.RECOVER
        stable = (head >= 1.2) & (speed < 0.5)
        self.stand_count = torch.where(recovering & stable, self.stand_count + 1, torch.where(recovering, 0, self.stand_count))
        success = recovering & (self.stand_count >= 25)
        self.phase[success] = self.SETTLE
        self.timer[success] = 0.0

        settling_done = (phase == self.SETTLE) & (self.timer >= 0.75)
        ramp_start = settling_done & (self.post_recovery_mode == self.RAMP)
        ramp_ids = ramp_start.nonzero(as_tuple=False).flatten()
        if ramp_ids.numel():
            # Sample while marked as RAMP so the recovery/settling zero-command
            # gate does not erase the target. Use the normal resample helper to
            # refresh the command timer as well.
            self.phase[ramp_ids] = self.RAMP
            self._resample(ramp_ids)
            # The RAMP branch must lead into locomotion, not the optional
            # standing-command subpopulation used by ordinary locomotion.
            self.is_standing_env[ramp_ids] = False
            self.ramp_target[ramp_ids] = self.vel_command_b[ramp_ids]
        self.phase[settling_done] = self.post_recovery_mode[settling_done]
        self.timer[settling_done] = 0.0

        ramp = self.phase == self.RAMP
        ramp_done = ramp & (self.timer >= 0.75)
        zero = (self.phase == self.RECOVER) | (self.phase == self.SETTLE) | (self.phase == self.HOLD) | (self.phase == self.STAND) | ramp
        self.vel_command_b[zero] = 0.0
        # Ramp the already sampled velocity command from zero after settling.
        self.vel_command_b[ramp] = self.ramp_target[ramp] * torch.clamp(self.timer[ramp] / 0.75, 0.0, 1.0).unsqueeze(-1)
        self.phase[ramp_done] = self.TRACK
        self.timer[ramp_done] = 0.0


class RecoveryLocomotionCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return RecoveryLocomotionCommand(self, env)
