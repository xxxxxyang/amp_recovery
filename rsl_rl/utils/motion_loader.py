from __future__ import annotations
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from tqdm import tqdm

import mjlab.utils.lab_api.math as math_utils


class AMPLoader:
    def __init__(self, motion_file: str,
                 body_names: Sequence[str],
                 anchor_name: str,
                 all_body_names: Sequence[str],
                 device: str = "cuda:0"):
        """Load AMP motion data.

        Args:
            motion_file: Path to a single .npz file or a directory of .npz files.
            body_names: Names of the target bodies to track.
            anchor_name: Name of the anchor (root) body.
            all_body_names: Ordered list of *all* body names in the model.
                The index of each name in this list must match the body
                dimension in the .npz arrays.
            device: Torch device.
        """
        assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

        # resolve name -> index
        all_names_list = list(all_body_names)
        self._body_indexes = [all_names_list.index(n) for n in body_names]
        self._anchor_indexes = all_names_list.index(anchor_name)
        self._num_bodies = len(self._body_indexes)

        # 检查是文件还是文件夹
        if os.path.isfile(motion_file):
            # 单个文件的情况（保持向后兼容）
            motion_files = [motion_file]
            motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
        elif os.path.isdir(motion_file):
            # 文件夹的情况：递归查找所有子目录下的 .npz 文件
            motion_names = []
            motion_files = []
            for root, _dirs, files in os.walk(motion_file, followlinks=True):
                for filename in sorted(files):
                    if filename.endswith('.npz'):
                        motion_names.append(os.path.splitext(filename)[0])
                        motion_files.append(os.path.join(root, filename))
            motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names))) if motion_files else ([], [])
            motion_files, motion_names = list(motion_files), list(motion_names)
            assert len(motion_files) > 0, f"No npz files found in directory: {motion_file}"
        else:
            raise ValueError(f"Path is neither a file nor a directory: {motion_file}")
        
        # 存储所有motion的数据列表
        self.motion_names = motion_names
        self.motion_sources = [name.split("_", 1)[0] for name in motion_names]
        self._body_pos_b_list = []
        self._body_quat_b_list = []
        self._body_ori_b_list = []
        self._body_lin_vel_b_list = []
        self._body_ang_vel_b_list = []
        self.motion_categories: list[str] = []
        
        # 处理每个motion文件
        for motion_idx, (motion_name, motion_path) in enumerate(zip(motion_names, motion_files)):
            print(f"Processing motion {motion_idx+1}/{len(motion_files)}: {motion_name}")
            data = np.load(motion_path)
            
            if motion_idx == 0:
                self.fps = data["fps"]
            
            _dof_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            _dof_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            _body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            _body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            _body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            _body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            
            time_step_total = _dof_pos.shape[0]
            
            # 为当前motion初始化存储
            _body_pos_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_quat_b = torch.zeros((time_step_total, self._num_bodies, 4), dtype=torch.float32, device=device)
            _body_ori_b = torch.zeros((time_step_total, self._num_bodies, 6), dtype=torch.float32, device=device)
            _body_lin_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_ang_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            
            # 处理所有帧
            for frame_idx in tqdm(range(time_step_total), desc=f"Preloading AMP data for {motion_name}"):
                # 获取当前帧的anchor和body数据
                tgt_anchor_pos_w = _body_pos_w[frame_idx, self._anchor_indexes, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_anchor_quat_w = _body_quat_w[frame_idx, self._anchor_indexes, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_body_pos_w = _body_pos_w[frame_idx, self._body_indexes, :]
                tgt_body_quat_w = _body_quat_w[frame_idx, self._body_indexes, :]
                tgt_body_lin_vel_w = _body_lin_vel_w[frame_idx, self._body_indexes, :]
                tgt_body_ang_vel_w = _body_ang_vel_w[frame_idx, self._body_indexes, :]

                # 计算body相对于anchor的位置和姿态 (局部坐标系)
                tgt_robot_body_pos_b, tgt_robot_body_quat_b = (
                    math_utils.subtract_frame_transforms(
                        tgt_anchor_pos_w,
                        tgt_anchor_quat_w,
                        tgt_body_pos_w,
                        tgt_body_quat_w,
                    )
                )

                # 将姿态四元数转换为旋转矩阵的前两列
                mat = math_utils.matrix_from_quat(tgt_robot_body_quat_b)
                tgt_robot_body_ori_b = mat[..., :, :2].reshape(self._num_bodies, 6)

                # 将速度转换到每个body自己的局部坐标系
                tgt_body_lin_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_lin_vel_w,
                )

                tgt_body_ang_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_ang_vel_w,
                )

                # 存储当前帧的局部坐标系数据
                _body_pos_b[frame_idx] = tgt_robot_body_pos_b
                _body_quat_b[frame_idx] = tgt_robot_body_quat_b
                _body_ori_b[frame_idx] = tgt_robot_body_ori_b
                _body_lin_vel_b[frame_idx] = tgt_body_lin_vel_b
                _body_ang_vel_b[frame_idx] = tgt_body_ang_vel_b
            
            # 将当前motion的数据添加到列表
            self._body_pos_b_list.append(_body_pos_b)
            self._body_quat_b_list.append(_body_quat_b)
            self._body_ori_b_list.append(_body_ori_b)
            self._body_lin_vel_b_list.append(_body_lin_vel_b)
            self._body_ang_vel_b_list.append(_body_ang_vel_b)
            # Preserve the top-level category for curated datasets laid out as
            # <root>/Recovery/*.npz and <root>/WalkandRun/*.npz.
            rel_parts = os.path.relpath(motion_path, motion_file).split(os.sep) if os.path.isdir(motion_file) else []
            self.motion_categories.append(rel_parts[0] if len(rel_parts) > 1 else "")
        
        # 为了向后兼容，使用第一个motion的数据作为默认值
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_quat_b = self._body_quat_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]

    @property
    def observation_dim(self) -> int:
        num_bodies = len(self._body_indexes)
        obs_dim = (3 + 6 + 3 + 3) * num_bodies  # pos, mat[:,:2], lin_vel, ang_vel
        return obs_dim

    def feed_forward_generator(
        self,
        num_mini_batch,
        mini_batch_size,
        category_ratio: dict[str, float] | None = None,
        source_ratio: dict[str, dict[str, float]] | None = None,
    ):
        if category_ratio or source_ratio:
            yield from self._category_feed_forward_generator(
                num_mini_batch, mini_batch_size, category_ratio, source_ratio
            )
            return
        num_motions = len(self._body_pos_b_list)
        
        for batch_idx in range(num_mini_batch):
            # 按顺序循环选择motion文件
            motion_idx = batch_idx % num_motions
            
            # 获取当前motion的数据
            current_body_pos_b = self._body_pos_b_list[motion_idx]
            current_body_ori_b = self._body_ori_b_list[motion_idx]
            current_body_lin_vel_b = self._body_lin_vel_b_list[motion_idx]
            current_body_ang_vel_b = self._body_ang_vel_b_list[motion_idx]
            current_time_step_total = current_body_pos_b.shape[0]
            
            # 从当前motion中随机采样
            idxs = torch.randint(0, current_time_step_total, (mini_batch_size,), device=current_body_pos_b.device)
            idxs = torch.clamp(idxs, max=current_time_step_total - 1)
            
            batch_body_pos_b = current_body_pos_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ori_b = current_body_ori_b[idxs]  # (mini_batch_size, num_bodies, 6)
            batch_body_lin_vel_b = current_body_lin_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ang_vel_b = current_body_ang_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            s = torch.cat(
                [
                    batch_body_pos_b.reshape(mini_batch_size, -1),
                    batch_body_ori_b.reshape(mini_batch_size, -1),
                    batch_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)

            next_idxs = (idxs + 1)
            next_idxs = torch.clamp(next_idxs, max=current_time_step_total - 1)
            batch_next_body_pos_b = current_body_pos_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ori_b = current_body_ori_b[next_idxs]  # (mini_batch_size, num_bodies, 6)
            batch_next_body_lin_vel_b = current_body_lin_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ang_vel_b = current_body_ang_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            s_next = torch.cat(
                [
                    batch_next_body_pos_b.reshape(mini_batch_size, -1),
                    batch_next_body_ori_b.reshape(mini_batch_size, -1),
                    batch_next_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_next_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)
            yield s, s_next

    def _category_feed_forward_generator(
        self, num_batches, batch_size, category_ratio, source_ratio=None
    ):
        """Sample exact category/source quotas, then sample clips and frame pairs."""
        source_ratio = source_ratio or {}
        category_ratio = category_ratio or {}
        categories = sorted(set(category_ratio) | set(source_ratio))
        category_weights = {
            name: category_ratio.get(name, 1.0) for name in categories
        }
        pools = {
            category: [i for i, value in enumerate(self.motion_categories) if value == category]
            for category in categories
        }
        missing = [name for name, indices in pools.items() if not indices]
        if missing:
            raise ValueError(f"AMP category folders have no motions: {missing}")
        weights = torch.tensor([category_weights[name] for name in categories], dtype=torch.float64)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(f"Invalid AMP category ratios: {category_weights}")
        weights /= weights.sum()
        quotas = torch.floor(weights * batch_size).long()
        for idx in torch.argsort(weights * batch_size - quotas, descending=True)[: batch_size - int(quotas.sum())]:
            quotas[idx] += 1

        source_pools: dict[str, dict[str, list[int]]] = {}
        source_quotas: dict[str, torch.Tensor] = {}
        for category in categories:
            configured = source_ratio.get(category)
            if not configured:
                continue
            category_pool = pools[category]
            sources = sorted(configured)
            source_pools[category] = {
                source: [i for i in category_pool if self.motion_sources[i] == source]
                for source in sources
            }
            missing_sources = [source for source, ids in source_pools[category].items() if not ids]
            if missing_sources:
                raise ValueError(
                    f"AMP source folders/clips missing in {category}: {missing_sources}"
                )
            source_weights = torch.tensor(
                [configured[source] for source in sources], dtype=torch.float64
            )
            if (source_weights < 0).any() or source_weights.sum() <= 0:
                raise ValueError(f"Invalid AMP source ratios for {category}: {configured}")
            source_weights /= source_weights.sum()
            category_count = int(quotas[categories.index(category)])
            weighted_source_counts = source_weights * category_count
            source_quota = torch.floor(weighted_source_counts).long()
            remainder = category_count - int(source_quota.sum())
            for idx in torch.argsort(weighted_source_counts - source_quota, descending=True)[:remainder]:
                source_quota[idx] += 1
            source_quotas[category] = source_quota

        def sample_pool(motion_pool: list[int], count: int, observations: list, next_observations: list):
            if count <= 0:
                return
            selected = torch.randint(
                len(motion_pool), (count,), device=self._body_pos_b_list[0].device
            )
            for local_clip in selected.unique().tolist():
                selected_count = int((selected == local_clip).sum())
                motion_idx = motion_pool[local_clip]
                body_pos = self._body_pos_b_list[motion_idx]
                body_ori = self._body_ori_b_list[motion_idx]
                body_lin = self._body_lin_vel_b_list[motion_idx]
                body_ang = self._body_ang_vel_b_list[motion_idx]
                last = body_pos.shape[0] - 1
                if last < 1:
                    raise ValueError(f"AMP motion {self.motion_names[motion_idx]} has fewer than 2 frames")
                frame = torch.randint(last, (selected_count,), device=body_pos.device)
                next_frame = frame + 1
                observations.append(torch.cat((body_pos[frame].flatten(1), body_ori[frame].flatten(1), body_lin[frame].flatten(1), body_ang[frame].flatten(1)), dim=-1))
                next_observations.append(torch.cat((body_pos[next_frame].flatten(1), body_ori[next_frame].flatten(1), body_lin[next_frame].flatten(1), body_ang[next_frame].flatten(1)), dim=-1))

        for _ in range(num_batches):
            observations, next_observations = [], []
            for category_idx, category in enumerate(categories):
                count = int(quotas[category_idx])
                if count == 0:
                    continue
                if category in source_pools:
                    for source_idx, source in enumerate(sorted(source_ratio[category])):
                        source_count = int(source_quotas[category][source_idx])
                        sample_pool(
                            source_pools[category][source], source_count,
                            observations, next_observations,
                        )
                else:
                    sample_pool(pools[category], count, observations, next_observations)
            s, s_next = torch.cat(observations), torch.cat(next_observations)
            order = torch.randperm(batch_size, device=s.device)
            yield s[order], s_next[order]
