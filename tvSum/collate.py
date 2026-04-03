import torch


def pad_1d(tensor, target_len):
    if tensor.shape[0] == target_len:
        return tensor
    pad_size = target_len - tensor.shape[0]
    return torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype)], dim=0)


def pad_2d(tensor, target_len):
    if tensor.shape[0] == target_len:
        return tensor
    pad_size = target_len - tensor.shape[0]
    pad = torch.zeros((pad_size, tensor.shape[1]), dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def collate_fn(batch):
    E_list, score_list, dur_list, valid_list, names = zip(*batch)

    max_len = max(x.shape[0] for x in E_list)

    E_batch = torch.stack([pad_2d(x, max_len) for x in E_list], dim=0)
    score_batch = torch.stack([pad_1d(x, max_len) for x in score_list], dim=0)
    dur_batch = torch.stack([pad_1d(x, max_len) for x in dur_list], dim=0)
    valid_batch = torch.stack([pad_1d(x, max_len) for x in valid_list], dim=0)

    return E_batch, score_batch, dur_batch, valid_batch, names