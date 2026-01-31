import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import IPython

e = IPython.embed

try:
    # sampling_plan.py 在同一目录下；训练通常从 policy/ACT 目录启动，因此可直接 import
    from sampling_plan import SamplingPlan
except Exception:
    # 兼容：如果以 package 方式导入（policy.ACT.utils），则使用相对导入
    from .sampling_plan import SamplingPlan

class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len, anchors=None):
        """EpisodicDataset（episode 采样 / anchor 采样 两用版）。

        baseline（默认）：
        - DataLoader 按 episode_ids 做 shuffle
        - 每次 __getitem__ 内部随机选 start_ts（起点）

        anchor 模式（启用 sampling plan 时）：
        - anchors 里包含每个样本的 (episode_id, start_ts)
        - __getitem__ 不再随机 start_ts，而是按 anchors[index] 确定性取片段

        这样可以在不改模型的前提下，把“训练看到什么数据”固化成可审计的计划（plan），
        为后续课程学习/去冗余/优先采样提供统一入口。
        """
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        # anchors: dict-like，至少包含 episode_ids/start_ts 两个 1D 数组
        self.anchors = anchors
        self.is_sim = None
        if self.anchors is not None:
            if len(self.anchors["episode_ids"]) == 0:
                raise ValueError("anchors is empty; cannot build anchored dataset.")
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        if self.anchors is not None:
            return int(len(self.anchors["episode_ids"]))
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        # === (1) 确定 episode_id 与 start_ts（起点） ===
        # baseline：episode_id 由 DataLoader 的 shuffle 决定；start_ts 在这里随机
        # plan：episode_id/start_ts 都由 anchors[index] 决定（可复现、可加权）
        if self.anchors is not None:
            episode_id = int(self.anchors["episode_ids"][index])
            start_ts = int(self.anchors["start_ts"][index])
        else:
            episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if self.anchors is None:
                # 只有 baseline 模式才随机 start_ts
                if sample_full_episode:
                    start_ts = 0
                else:
                    start_ts = np.random.choice(episode_len)
            else:
                # anchor 模式下做一个安全裁剪：避免 start_ts 越界
                start_ts = int(np.clip(int(start_ts), 0, max(0, episode_len - 1)))
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)  # 根据max_action_len初始化
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)  # 初始化为全1（True）
        is_pad[:action_len] = 0  # 前action_len个位置设置为0（False），表示非填充部分

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Pad all tensors to the maximum size
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            # Pad with the last element
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats, max_action_len


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    *,
    sampling_plan_path: str | None = None,
    plan_num_samples: int = 0,
    plan_sampler_seed: int = 0,
    plan_no_replacement: bool = False,
    plan_verify_sha256: bool = True,
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    # === train dataloader ===
    # 默认（sampling_plan_path=None）：保持原始逻辑不变
    train_sampler = None
    anchors = None
    if sampling_plan_path:
        # 读取 plan 并过滤到 train episodes
        plan = SamplingPlan.load(sampling_plan_path, verify_sha256=bool(plan_verify_sha256))
        keep = np.isin(plan.episode_ids, train_indices)
        if not np.any(keep):
            raise ValueError(
                f"Sampling plan has zero anchors after filtering to train split. "
                f"plan={sampling_plan_path}, train_episodes={len(train_indices)}"
            )

        anchors = {
            "episode_ids": plan.episode_ids[keep].astype(np.int64, copy=False),
            "start_ts": plan.start_ts[keep].astype(np.int64, copy=False),
            "stage_ids": plan.stage_ids[keep].astype(np.int32, copy=False),
            "weights": plan.weights[keep].astype(np.float32, copy=False),
        }

        # plan_num_samples（每个 epoch 采样多少个 anchor）
        # - baseline 里每个 epoch 大约“每条 episode 采 1 个片段”，所以这里默认对齐 train_episode 数量
        num_samples = int(plan_num_samples) if int(plan_num_samples) > 0 else int(len(train_indices))
        replacement = not bool(plan_no_replacement)
        g = torch.Generator()
        g.manual_seed(int(plan_sampler_seed))

        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(anchors["weights"], dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement,
            generator=g,
        )
        print(
            f"[sampling_plan] 已启用：anchors={len(anchors['episode_ids'])}, "
            f"每个epoch采样数={num_samples}, 有放回采样={replacement}, sampler_seed={int(plan_sampler_seed)}"
        )

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_action_len, anchors=anchors)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_action_len)

    # DataLoader 的 worker 设置（默认保持原值：num_workers=1, prefetch_factor=1）。
    # 但在某些系统（例如受限容器）里，多进程可能会触发权限/共享内存问题；
    # 此时你可以在命令行前加环境变量来降级到单进程加载：
    #   ACT_NUM_WORKERS=0 bash ...
    num_workers = int(os.environ.get("ACT_NUM_WORKERS", "1"))
    prefetch_factor = int(os.environ.get("ACT_PREFETCH_FACTOR", "1"))
    common_loader_kwargs = {"pin_memory": True, "num_workers": num_workers}
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **common_loader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        **common_loader_kwargs,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
