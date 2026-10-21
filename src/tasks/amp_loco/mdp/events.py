from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager
from src.tasks.amp_loco.mdp.unified_command import RecoveryLocomotionCommand

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _initialize_unified_reset_groups(
    env: ManagerBasedRlEnv,
    recovery_ratio: float,
    stand_ratio: float,
    play_phase: str = "random",
) -> torch.Tensor:
    """Create persistent task cohorts and their post-recovery branches once."""
    if recovery_ratio < 0 or stand_ratio < 0 or recovery_ratio + stand_ratio > 1:
        raise ValueError("Reset group ratios must be nonnegative and sum to at most one")
    if play_phase not in {"random", "recovery", "stand", "walk"}:
        raise ValueError(f"Unsupported unified play phase: {play_phase}")

    cmd = env.command_manager.get_term("twist")
    if not isinstance(cmd, RecoveryLocomotionCommand):
        raise TypeError("Unified reset requires RecoveryLocomotionCommand for 'twist'")

    groups = torch.full((env.num_envs,), 2, dtype=torch.long, device=env.device)
    order = torch.randperm(env.num_envs, device=env.device)
    n_recovery = round(env.num_envs * recovery_ratio)
    n_stand = round(env.num_envs * stand_ratio)
    recovery_env_ids = order[:n_recovery]
    stand_env_ids = order[n_recovery : n_recovery + n_stand]
    groups[recovery_env_ids] = 0
    groups[stand_env_ids] = 1

    if play_phase != "random":
        forced_group = {"recovery": 0, "stand": 1, "walk": 2}[play_phase]
        groups.fill_(forced_group)
        recovery_env_ids = (groups == 0).nonzero(as_tuple=False).flatten()

    post_modes = torch.full((env.num_envs,), cmd.HOLD, dtype=torch.long, device=env.device)
    if recovery_env_ids.numel():
        recovery_branch = recovery_env_ids[
            torch.randperm(recovery_env_ids.numel(), device=env.device)
        ]
        post_modes[recovery_branch[recovery_env_ids.numel() // 2 :]] = cmd.RAMP

    env._amp_unified_reset_groups = groups
    env._amp_unified_post_recovery_modes = post_modes
    env._amp_unified_play_phase = play_phase
    return groups


class MotionResetManager:
    """Manages motion frame data and delayed-reset logic for AMP environments."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}
        self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}
        self.pose_frame_indices: dict[tuple, torch.Tensor] = {}
        self.default_lie_root_heights: dict[tuple[int, str], float] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(
        self,
        env: ManagerBasedRlEnv,
        motion_dir: str,
        recovery_dir: str | None = None,
    ) -> None:
        if motion_dir in self.walk_run_frames:
            return

        loader = MotionLoader(
            motion_dir=motion_dir,
            tgt_body_indexes=[],
            tgt_anchor_indexes=0,
            feet_indexes=0,
            device=str(env.device),
            recovery_dir=recovery_dir,
        )

        self.walk_run_frames[motion_dir] = self._concat_frames(loader.motion_data)
        motion_count = self.walk_run_frames[motion_dir]["root_pos"].shape[0]
        print(f"[MotionResetManager] Loaded {len(loader.motion_data)} clips, {motion_count} frames from {motion_dir}")

        if loader.motion_data_recovery:
            self.recovery_frames[motion_dir] = self._concat_frames(loader.motion_data_recovery)
            recovery_count = self.recovery_frames[motion_dir]["root_pos"].shape[0]
            print(f"[MotionResetManager] Loaded {len(loader.motion_data_recovery)} recovery clips, {recovery_count} frames from {recovery_dir}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        pose_mode: str = "random",
        frame_index: int | None = None,
        pose_filters: dict[str, float] | None = None,
        default_pose_reset_ratio: float = 0.0,
        recovery_sampling_mode: str = "random",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        # Play can override these values at runtime from the Viser controls.
        pose_mode = getattr(env, "_amp_getup_reset_pose_mode", pose_mode)
        frame_index = getattr(env, "_amp_getup_reset_frame", frame_index)
        pose_filters = pose_filters or {}

        # Synthetic default-pose resets do not use the motion reference data.
        # They are useful for inspecting recovery from a simple, repeatable pose.
        if pose_mode in {"default_supine", "default_prone"}:
            self._write_default_lie_state(env, env_ids, pose_mode, asset_cfg)
            return

        if not 0.0 <= default_pose_reset_ratio <= 1.0:
            raise ValueError(
                "default_pose_reset_ratio must be between 0 and 1, "
                f"got {default_pose_reset_ratio}"
            )

        # During training, mix a small fraction of synthetic flat default-pose
        # starts into the Recovery-frame reset distribution.
        if pose_mode == "random" and default_pose_reset_ratio > 0.0:
            use_default = (
                torch.rand(len(env_ids), device=env.device) < default_pose_reset_ratio
            )
            default_ids = env_ids[use_default]
            env_ids = env_ids[~use_default]
            if len(default_ids) > 0:
                use_supine = torch.rand(len(default_ids), device=env.device) < 0.5
                supine_ids = default_ids[use_supine]
                prone_ids = default_ids[~use_supine]
                if len(supine_ids) > 0:
                    self._write_default_lie_state(
                        env, supine_ids, "default_supine", asset_cfg
                    )
                if len(prone_ids) > 0:
                    self._write_default_lie_state(
                        env, prone_ids, "default_prone", asset_cfg
                    )
            if len(env_ids) == 0:
                return

        # Split remaining reference-frame envs into delay and normal groups.
        delay_mask = self._get_delay_env_mask(env)
        if delay_mask is not None:
            is_delay = delay_mask[env_ids]
            delay_ids = env_ids[is_delay]
            normal_ids = env_ids[~is_delay]
        else:
            delay_ids = env_ids[:0]  # empty
            normal_ids = env_ids

        # Reset normal envs with the selected reference frame pool.
        if len(normal_ids) > 0:
            frames = self.walk_run_frames[motion_dir]
            selected = self._sample_frame_indices(
                frames, len(normal_ids), pose_mode, frame_index, pose_filters, motion_dir
            )
            self._remember_reset_frames(env, normal_ids, selected)
            self._write_reset_state(env, normal_ids, frames, asset_cfg, selected)

        # Reset delay envs with recovery data (fallback to walk/run if unavailable).
        if len(delay_ids) > 0:
            recovery = self.recovery_frames.get(motion_dir)
            frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
            if recovery is not None and recovery_sampling_mode == "low_non_upright":
                selected = self._sample_recovery_candidate_frames(frames, len(delay_ids))
            elif recovery_sampling_mode == "random":
                selected = self._sample_frame_indices(
                    frames, len(delay_ids), pose_mode, frame_index, pose_filters, motion_dir
                )
            else:
                raise ValueError(
                    f"Unsupported Recovery reset sampling mode: {recovery_sampling_mode}"
                )
            self._remember_reset_frames(env, delay_ids, selected)
            self._write_reset_state(env, delay_ids, frames, asset_cfg, selected)

    @staticmethod
    def _sample_recovery_candidate_frames(
        frames: dict[str, torch.Tensor], count: int
    ) -> torch.Tensor:
        """Sample low, non-upright Recovery frames using unified-task rules."""
        up_b = torch.zeros_like(frames["root_pos"])
        up_b[:, 2] = 1.0
        torso_up = quat_apply(frames["root_quat"], up_b)[:, 2]
        lin_speed = torch.linalg.vector_norm(frames["root_lin_vel"], dim=-1)
        ang_speed = torch.linalg.vector_norm(frames["root_ang_vel"], dim=-1)
        candidates = (
            (frames["root_pos"][:, 2] < 0.70)
            & (torso_up.abs() < 0.78)
            & (lin_speed < 1.0)
            & (ang_speed < 1.5)
        ).nonzero(as_tuple=False).flatten()
        if not candidates.numel():
            candidates = (frames["root_pos"][:, 2] < 0.70).nonzero(
                as_tuple=False
            ).flatten()
        if not candidates.numel():
            candidates = torch.arange(
                frames["root_pos"].shape[0], device=frames["root_pos"].device
            )
        selected = candidates[
            torch.randint(candidates.numel(), (count,), device=candidates.device)
        ]
        return selected

    @staticmethod
    def _remember_reset_frames(
        env: ManagerBasedRlEnv, env_ids: torch.Tensor, frame_indices: torch.Tensor
    ) -> None:
        last_frames = getattr(env, "_amp_getup_last_reset_frames", None)
        if last_frames is None or last_frames.numel() != env.num_envs:
            last_frames = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            )
            env._amp_getup_last_reset_frames = last_frames
        last_frames[env_ids.long()] = frame_indices

    def frame_count(self, motion_dir: str) -> int:
        """Return the number of flattened reference frames loaded for a source."""
        return int(self.walk_run_frames[motion_dir]["root_pos"].shape[0])

    def _sample_frame_indices(
        self,
        frames: dict[str, torch.Tensor],
        count: int,
        pose_mode: str,
        frame_index: int | None,
        pose_filters: dict[str, float],
        motion_dir: str,
    ) -> torch.Tensor:
        total_frames = frames["root_pos"].shape[0]
        if pose_mode == "random":
            return torch.randint(total_frames, (count,), device=frames["root_pos"].device)
        if pose_mode == "frame":
            if frame_index is None or not 0 <= frame_index < total_frames:
                raise ValueError(
                    f"reset frame must be in [0, {total_frames - 1}], got {frame_index}"
                )
            return torch.full(
                (count,), frame_index, dtype=torch.long, device=frames["root_pos"].device
            )
        if pose_mode not in {"supine", "prone"}:
            raise ValueError(f"Unsupported Getup reset pose: {pose_mode}")

        defaults = {
            "max_root_height": 0.30,
            "min_abs_forward_z": 0.75,
            "max_root_linear_speed": 0.50,
            "max_root_angular_speed": 1.00,
        }
        filters = {**defaults, **pose_filters}
        cache_key = (
            motion_dir,
            pose_mode,
            tuple(sorted(filters.items())),
            str(frames["root_pos"].device),
            frames["root_pos"].data_ptr(),
        )
        candidates = self.pose_frame_indices.get(cache_key)
        if candidates is None:
            root_quat = frames["root_quat"]
            forward_b = torch.zeros_like(frames["root_pos"])
            forward_b[:, 0] = 1.0
            forward_z = quat_apply(root_quat, forward_b)[:, 2]
            forward_mask = (
                forward_z >= filters["min_abs_forward_z"]
                if pose_mode == "supine"
                else forward_z <= -filters["min_abs_forward_z"]
            )
            root_linear_speed = torch.linalg.vector_norm(frames["root_lin_vel"], dim=-1)
            root_angular_speed = torch.linalg.vector_norm(frames["root_ang_vel"], dim=-1)
            mask = (
                (frames["root_pos"][:, 2] <= filters["max_root_height"])
                & forward_mask
                & (root_linear_speed <= filters["max_root_linear_speed"])
                & (root_angular_speed <= filters["max_root_angular_speed"])
            )
            candidates = mask.nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise ValueError(
                    f"No reference frames match reset pose '{pose_mode}'. "
                    "Adjust the pose filters for this motion set."
                )
            self.pose_frame_indices[cache_key] = candidates

        selected = torch.randint(candidates.numel(), (count,), device=candidates.device)
        return candidates[selected]

    def _get_delay_env_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
        """Get delay env mask from DelayedTerminationManager if installed."""
        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            return tm._delay_env_mask
        return None

    def _write_reset_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        frames: dict[str, torch.Tensor],
        asset_cfg: SceneEntityCfg,
        frame_indices: torch.Tensor | None = None,
    ) -> None:
        num_reset = len(env_ids)
        idx = frame_indices
        if idx is None:
            total_frames = frames["root_pos"].shape[0]
            idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

        asset: Entity = env.scene[asset_cfg.name]

        # --- Root pose ---
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        positions = env.scene.env_origins[env_ids].clone()

        # --- Key Fix for terrain ---
        terrain_z = positions[:, 2].clone()
        positions[:, 2] = terrain_z + root_pos[:, 2]

        root_pose = torch.cat([positions, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # --- Root velocity ---
        root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        # --- Joint state ---
        joint_pos = frames["joint_pos"][idx]
        joint_vel = frames["joint_vel"][idx]

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
            joint_pos_limits[..., 0], joint_pos_limits[..., 1]
        )

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        asset.write_joint_state_to_sim(
            joint_pos_clamped,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    def _write_default_lie_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        pose_mode: str,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Reset to default joint angles, rotated flat onto the back or stomach."""
        last_frames = getattr(env, "_amp_getup_last_reset_frames", None)
        if last_frames is not None:
            last_frames[env_ids.long()] = -1

        asset: Entity = env.scene[asset_cfg.name]
        root_default = asset.data.default_root_state[env_ids]
        joint_default = asset.data.default_joint_pos[env_ids]
        joint_velocity_default = asset.data.default_joint_vel[env_ids]

        # The robot's forward axis is +X. Rotating it toward +Z makes the face
        # point upward (supine); toward -Z makes the face point toward the floor.
        half_angle = math.pi / 4.0
        sign = -1.0 if pose_mode == "default_supine" else 1.0
        lie_quat = torch.tensor(
            [math.cos(half_angle), 0.0, sign * math.sin(half_angle), 0.0],
            dtype=root_default.dtype,
            device=env.device,
        ).expand(len(env_ids), -1)
        root_quat = quat_mul(root_default[:, 3:7], lie_quat)

        cache_key = (id(asset), pose_mode)
        root_height = self.default_lie_root_heights.get(cache_key)
        if root_height is None:
            root_height = self._measure_default_lie_root_height(
                env, asset, asset_cfg, env_ids[0], lie_quat[0]
            )
            self.default_lie_root_heights[cache_key] = root_height

        root_pos = env.scene.env_origins[env_ids].clone()
        root_pos[:, 2] += root_height
        asset.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
        asset.write_root_link_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=env.device, dtype=root_pos.dtype), env_ids
        )
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)
        asset.write_joint_state_to_sim(
            joint_default[:, joint_ids],
            joint_velocity_default[:, joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    def _measure_default_lie_root_height(
        self,
        env: ManagerBasedRlEnv,
        asset: Entity,
        asset_cfg: SceneEntityCfg,
        probe_env_id: torch.Tensor,
        lie_quat: torch.Tensor,
    ) -> float:
        """Find a collision-geometry based root height for a flat default pose."""
        probe_id = int(probe_env_id.item())
        qpos_ids = torch.cat(
            [asset.indexing.free_joint_q_adr, asset.indexing.joint_q_adr]
        )
        qvel_ids = torch.cat(
            [asset.indexing.free_joint_v_adr, asset.indexing.joint_v_adr]
        )
        old_qpos = asset.data.data.qpos[probe_id, qpos_ids].clone()
        old_qvel = asset.data.data.qvel[probe_id, qvel_ids].clone()

        origin = env.scene.env_origins[probe_id : probe_id + 1].clone()
        default_root = asset.data.default_root_state[probe_id : probe_id + 1]
        probe_pos = origin.clone()
        probe_pose = torch.cat(
            [probe_pos, quat_mul(default_root[:, 3:7], lie_quat.unsqueeze(0))], dim=-1
        )
        probe_ids = torch.tensor([probe_id], device=env.device)
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)
        asset.write_root_link_pose_to_sim(probe_pose, probe_ids)
        asset.write_root_link_velocity_to_sim(
            torch.zeros((1, 6), device=env.device, dtype=probe_pos.dtype),
            probe_ids,
        )
        asset.write_joint_state_to_sim(
            asset.data.default_joint_pos[probe_id : probe_id + 1, joint_ids],
            asset.data.default_joint_vel[probe_id : probe_id + 1, joint_ids],
            env_ids=probe_ids,
            joint_ids=joint_ids,
        )
        env.sim.forward()

        collision_local_ids = [
            index
            for index, geom in enumerate(asset.indexing.geoms)
            if geom.name is not None and "collision" in geom.name
        ]
        if not collision_local_ids:
            raise ValueError("Cannot place default lie pose: robot has no collision geoms.")
        local_ids = torch.tensor(collision_local_ids, dtype=torch.long, device=env.device)
        geom_ids = asset.indexing.geom_ids[local_ids].long()
        geom_pos = asset.data.geom_pos_w[probe_id, local_ids]
        geom_xmat = asset.data.data.geom_xmat[probe_id, geom_ids].reshape(-1, 3, 3)
        geom_aabb = asset.data.model.geom_aabb[probe_id, geom_ids]
        half_extents = (geom_aabb[:, 1] - geom_aabb[:, 0]) * 0.5
        vertical_radius = (geom_xmat[:, 2, :].abs() * half_extents).sum(dim=-1)
        min_geom_z = (geom_pos[:, 2] - vertical_radius).min()
        root_height = max(0.0, float((origin[0, 2] - min_geom_z).item())) + 0.015

        asset.data.data.qpos[probe_id, qpos_ids] = old_qpos
        asset.data.data.qvel[probe_id, qvel_ids] = old_qvel
        env.sim.forward()
        return root_height

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
        root_pos_list = []
        root_quat_list = []
        root_lin_vel_list = []
        root_ang_vel_list = []
        joint_pos_list = []
        joint_vel_list = []
        for motion in motions:
            root_pos_list.append(motion["body_pos_w"][:, 0, :])
            root_quat_list.append(motion["body_quat_w"][:, 0, :])
            root_lin_vel_list.append(motion["body_lin_vel_w"][:, 0, :])
            root_ang_vel_list.append(motion["body_ang_vel_w"][:, 0, :])
            joint_pos_list.append(motion["dof_pos"])
            joint_vel_list.append(motion["dof_vel"])
        return {
            "root_pos": torch.cat(root_pos_list, dim=0),
            "root_quat": torch.cat(root_quat_list, dim=0),
            "root_lin_vel": torch.cat(root_lin_vel_list, dim=0),
            "root_ang_vel": torch.cat(root_ang_vel_list, dim=0),
            "joint_pos": torch.cat(joint_pos_list, dim=0),
            "joint_vel": torch.cat(joint_vel_list, dim=0),
        }


# ------------------------------------------------------------------
# Event callback wrappers (thin delegates to singleton)
# ------------------------------------------------------------------

def init_motion_loader(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
    unified_recovery_ratio: float | None = None,
    unified_stand_ratio: float = 0.0,
    unified_play_phase: str = "random",
) -> None:
    """Startup event: load motion data and optionally install delayed termination."""
    MotionResetManager.get().init(
        env=env,
        motion_dir=motion_dir,
        recovery_dir=recovery_dir,
    )

    # Unified tasks use the same fixed Recovery cohort for reset sampling and
    # delayed termination. Other AMP tasks retain the legacy random delay mask.
    if unified_recovery_ratio is not None:
        groups = _initialize_unified_reset_groups(
            env,
            recovery_ratio=unified_recovery_ratio,
            stand_ratio=unified_stand_ratio,
            play_phase=unified_play_phase,
        )
        delay_mask = groups == 0
        num_delay = int(delay_mask.sum().item())
    else:
        num_delay = int(env.num_envs * delay_reset_env_ratio)
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if num_delay > 0:
            delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
            delay_mask[delay_indices] = True

    if num_delay > 0 and max_delay_steps > 0:
        env.termination_manager = DelayedTerminationManager(
            base=env.termination_manager,
            delay_env_mask=delay_mask,
            max_delay_steps=max_delay_steps,
        )
        print(
            "[init_motion_loader] DelayedTerminationManager installed: "
            f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
        )


def push_unified_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    recovery_push_probability: float = 0.25,
    recovery_min_walk_s: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Push ordinary loco/stand envs and selectively push Recovery envs in TRACK."""
    groups = getattr(env, "_amp_unified_reset_groups", None)
    if groups is None:
        raise RuntimeError("Unified push event requires initialized reset cohorts")
    if not 0.0 <= recovery_push_probability <= 1.0:
        raise ValueError("recovery_push_probability must be between zero and one")
    if env_ids is None or env_ids.numel() == 0:
        return

    cmd = env.command_manager.get_term("twist")
    if not isinstance(cmd, RecoveryLocomotionCommand):
        raise TypeError("Unified push event requires RecoveryLocomotionCommand")

    groups_for_ids = groups[env_ids]
    phases = cmd.phase[env_ids]
    ordinary = (groups_for_ids != 0) & (
        (phases == cmd.TRACK) | (phases == cmd.STAND)
    )
    command_moving = torch.linalg.vector_norm(cmd.vel_command_b[env_ids], dim=-1) > 0.1
    recovery_walking = (
        (groups_for_ids == 0)
        & (phases == cmd.TRACK)
        & (cmd.timer[env_ids] >= recovery_min_walk_s)
        & command_moving
    )
    recovery_ids = env_ids[recovery_walking]
    if recovery_ids.numel() and recovery_push_probability < 1.0:
        keep = torch.rand(recovery_ids.numel(), device=env.device) < recovery_push_probability
        recovery_ids = recovery_ids[keep]

    push_ids = torch.cat((env_ids[ordinary], recovery_ids))
    if push_ids.numel():
        envs_mdp.push_by_setting_velocity(
            env, push_ids, velocity_range=velocity_range, asset_cfg=asset_cfg
        )


def reset_from_motion_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    pose_mode: str = "random",
    frame_index: int | None = None,
    pose_filters: dict[str, float] | None = None,
    default_pose_reset_ratio: float = 0.0,
    recovery_sampling_mode: str = "random",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset event: reset envs from random motion frames, with delay support."""
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        pose_mode=pose_mode,
        frame_index=frame_index,
        pose_filters=pose_filters,
        default_pose_reset_ratio=default_pose_reset_ratio,
        recovery_sampling_mode=recovery_sampling_mode,
        asset_cfg=asset_cfg,
    )


def reset_getup_stand_counter(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
    """Clear the get-up success hold counter for reset environments."""
    count = getattr(env, "_amp_getup_stand_count", None)
    if count is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    count[env_ids] = 0


def reset_unified_recovery_locomotion(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    play_phase: str = "random",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset environments using their fixed recovery/stand/walk cohort."""
    env._amp_unified_task = True
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env_ids = env_ids.long()
    if not env_ids.numel():
        return
    # The Viser play controls can change the selected cohort between resets.
    # Training keeps the startup value, while play reads this mutable value.
    play_phase = getattr(env, "_amp_unified_play_phase", play_phase)
    if play_phase not in {"random", "recovery", "stand", "walk"}:
        raise ValueError(f"Unsupported unified play phase: {play_phase}")

    manager = MotionResetManager.get()
    recovery = manager.recovery_frames.get(motion_dir)
    if recovery is None:
        raise RuntimeError(f"No Recovery motions loaded for unified reset: {motion_dir}")
    locomotion = manager.walk_run_frames[motion_dir]
    cmd = env.command_manager.get_term("twist")
    if not isinstance(cmd, RecoveryLocomotionCommand):
        raise TypeError("Unified reset requires RecoveryLocomotionCommand for 'twist'")

    # Cohorts and delayed-reset masks are assigned together at startup. Reset
    # must preserve those IDs so async dones cannot drift the task mixture.
    groups = getattr(env, "_amp_unified_reset_groups", None)
    post_modes = getattr(env, "_amp_unified_post_recovery_modes", None)
    if (
        groups is None or groups.numel() != env.num_envs
        or str(groups.device) != str(env.device)
        or post_modes is None or post_modes.numel() != env.num_envs
        or str(post_modes.device) != str(env.device)
    ):
        raise RuntimeError(
            "Unified reset cohorts must be initialized by init_motion_loader before reset"
        )

    group_ids = groups[env_ids]
    if play_phase == "random":
        is_recovery = group_ids == 0
        is_stand = group_ids == 1
    else:
        is_recovery = torch.full((len(env_ids),), play_phase == "recovery", device=env.device, dtype=torch.bool)
        is_stand = torch.full((len(env_ids),), play_phase == "stand", device=env.device, dtype=torch.bool)

    recovery_ids = env_ids[is_recovery]
    stand_ids = env_ids[is_stand]
    walk_ids = env_ids[~(is_recovery | is_stand)]
    phases = torch.full((len(env_ids),), cmd.TRACK, dtype=torch.long, device=env.device)
    phases[is_recovery] = cmd.RECOVER
    phases[is_stand] = cmd.STAND
    post_mode = post_modes[env_ids].clone()
    cmd.reset_phases(env_ids, phases, post_mode)

    if recovery_ids.numel():
        # Keep recovery starts low and non-upright; preserve every annotated
        # clip frame in the expert library without further file segmentation.
        up_b = torch.zeros_like(recovery["root_pos"])
        up_b[:, 2] = 1.0
        torso_up = quat_apply(recovery["root_quat"], up_b)[:, 2]
        lin_speed = torch.linalg.vector_norm(recovery["root_lin_vel"], dim=-1)
        ang_speed = torch.linalg.vector_norm(recovery["root_ang_vel"], dim=-1)
        candidates = ((recovery["root_pos"][:, 2] < 0.70) & (torso_up.abs() < 0.78) & (lin_speed < 1.0) & (ang_speed < 1.5)).nonzero().flatten()
        if not candidates.numel():
            candidates = (recovery["root_pos"][:, 2] < 0.70).nonzero().flatten()
        if not candidates.numel():
            candidates = torch.arange(recovery["root_pos"].shape[0], device=env.device)
        selected = candidates[torch.randint(len(candidates), (len(recovery_ids),), device=env.device)]
        manager._remember_reset_frames(env, recovery_ids, selected)
        manager._write_reset_state(env, recovery_ids, recovery, asset_cfg, selected)
    if walk_ids.numel():
        selected = torch.randint(locomotion["root_pos"].shape[0], (len(walk_ids),), device=env.device)
        manager._remember_reset_frames(env, walk_ids, selected)
        manager._write_reset_state(env, walk_ids, locomotion, asset_cfg, selected)
    if stand_ids.numel():
        manager._remember_reset_frames(env, stand_ids, torch.full((len(stand_ids),), -1, device=env.device, dtype=torch.long))
        asset: Entity = env.scene[asset_cfg.name]
        default = asset.data.default_root_state[stand_ids]
        root_pos = env.scene.env_origins[stand_ids].clone()
        root_pos[:, 2] += default[:, 2]
        root_pose = torch.cat((root_pos, default[:, 3:7]), dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=stand_ids)
        asset.write_root_link_velocity_to_sim(torch.zeros((len(stand_ids), 6), device=env.device, dtype=root_pos.dtype), env_ids=stand_ids)
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)
        asset.write_joint_state_to_sim(asset.data.default_joint_pos[stand_ids][:, joint_ids], asset.data.default_joint_vel[stand_ids][:, joint_ids], env_ids=stand_ids, joint_ids=joint_ids)
