import torch

def pad_video(E, mask, dur, n_max=40):
    n, d = E.shape

    E_pad = torch.zeros(n_max, d, dtype=E.dtype)
    mask_pad = torch.zeros(n_max, dtype=mask.dtype)
    dur_pad = torch.zeros(n_max, dtype=dur.dtype)
    valid = torch.zeros(n_max, dtype=torch.bool)

    length = min(n, n_max)

    E_pad[:length] = E[:length]
    mask_pad[:length] = mask[:length]
    dur_pad[:length] = dur[:length]
    valid[:length] = True

    return E_pad, mask_pad, dur_pad, valid