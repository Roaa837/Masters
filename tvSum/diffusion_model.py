import torch
import torch.nn as nn


class MaskDiffusionModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=512, timesteps=100):
        super().__init__()

        self.time_embed = nn.Embedding(timesteps, 64)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 1 + 1 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_t, E, dur, t):
        # E:   [B, N, D]
        # x_t: [B, N]
        # dur: [B, N]
        # t:   [B]

        B, N, D = E.shape

        t_emb = self.time_embed(t)              # [B, 64]
        t_emb = t_emb.unsqueeze(1).repeat(1, N, 1)  # [B, N, 64]

        x_t = x_t.unsqueeze(-1)   # [B, N, 1]
        dur = dur.unsqueeze(-1)   # [B, N, 1]

        inp = torch.cat([E, x_t, dur, t_emb], dim=-1)
        noise_pred = self.net(inp).squeeze(-1)  # [B, N]

        return noise_pred