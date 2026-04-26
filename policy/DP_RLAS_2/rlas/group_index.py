import numpy as np


def _resolve_pad_before(dataset):
    sampler = getattr(dataset, "sampler", None)
    if sampler is not None:
        pad_before = getattr(sampler, "pad_before", None)
        if pad_before is not None:
            return int(pad_before)

    pad_before = getattr(dataset, "pad_before", None)
    if pad_before is not None:
        return int(pad_before)

    for attr in ("sampler_pad_before", "dataset_pad_before", "_pad_before"):
        pad_before = getattr(dataset, attr, None)
        if pad_before is not None:
            return int(pad_before)

    raise ValueError("Unable to resolve pad_before from dataset/sampler.")


def build_window_groups(dataset, group_size: int = 8):
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    sampler = getattr(dataset, "sampler", None)
    if sampler is None:
        raise ValueError("dataset does not expose sampler")

    indices = np.asarray(sampler.indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError(f"sampler.indices must have shape (N,4), got {indices.shape}")

    replay_buffer = getattr(dataset, "replay_buffer", None)
    if replay_buffer is None:
        raise ValueError("dataset does not expose replay_buffer")
    episode_ends = np.asarray(replay_buffer.episode_ends[:], dtype=np.int64)
    if episode_ends.ndim != 1:
        raise ValueError("episode_ends must be a 1D array")

    pad_before = _resolve_pad_before(dataset)
    n_windows = int(indices.shape[0])
    if len(dataset) != n_windows:
        raise ValueError(f"len(dataset)={len(dataset)} but sampler has {n_windows} windows")

    episode_ids = np.searchsorted(episode_ends, indices[:, 0], side="right")

    prev_ends = np.zeros_like(episode_ends)
    prev_ends[1:] = episode_ends[:-1]
    episode_start_idx = prev_ends[episode_ids]

    logical_start = (indices[:, 0] - episode_start_idx) - indices[:, 2]
    decision_ts = logical_start + pad_before
    decision_ts = decision_ts.astype(np.int64)

    local_group = decision_ts // group_size

    group_key_to_gid = {}
    groups = []
    window_to_group = np.empty((n_windows,), dtype=np.int64)
    for win_idx in range(n_windows):
        key = (int(episode_ids[win_idx]), int(local_group[win_idx]))
        gid = group_key_to_gid.get(key)
        if gid is None:
            gid = len(groups)
            group_key_to_gid[key] = gid
            groups.append([])
        groups[gid].append(win_idx)
        window_to_group[win_idx] = gid

    group_to_indices = [np.asarray(g, dtype=np.int64) for g in groups]
    group_lens = np.asarray([len(g) for g in group_to_indices], dtype=np.int64)

    if window_to_group.shape != (n_windows,):
        raise ValueError("window_to_group shape mismatch")
    if np.any(window_to_group < 0) or np.any(window_to_group >= len(group_to_indices)):
        raise ValueError("window_to_group contains invalid group id")
    if np.any(group_lens <= 0):
        raise ValueError("some groups are empty")

    covered = np.concatenate(group_to_indices, axis=0) if len(group_to_indices) > 0 else np.zeros((0,), dtype=np.int64)
    if covered.shape[0] != n_windows:
        raise ValueError("group coverage count mismatch")
    if np.unique(covered).shape[0] != n_windows:
        raise ValueError("windows are not uniquely assigned to groups")
    if covered.size > 0 and (covered.min() < 0 or covered.max() >= n_windows):
        raise ValueError("group indices out of range")

    group_meta = {
        "num_windows": int(n_windows),
        "num_groups": int(len(group_to_indices)),
        "group_size": int(group_size),
        "group_len_min": int(group_lens.min()) if group_lens.size > 0 else 0,
        "group_len_max": int(group_lens.max()) if group_lens.size > 0 else 0,
        "group_len_mean": float(group_lens.mean()) if group_lens.size > 0 else 0.0,
    }

    return window_to_group, group_to_indices, group_meta