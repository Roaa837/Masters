import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from tvsum_dataset import TVSumShotDataset
from collate import collate_fn
from diffusion_model import MaskDiffusionModel


# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JSON_DIR = "data/final_with_scores"
EMBEDDING_DIR = "data/text_embeddings"

BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
TIMESTEPS = 100
VAL_RATIO = 0.2


# -------------------------
# DIFFUSION HELPERS
# -------------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_constants(timesteps):
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_hat


def forward_diffusion_sample(x0, t, alpha_hat, device):
    noise = torch.randn_like(x0).to(device)

    sqrt_alpha_hat = torch.sqrt(alpha_hat[t]).unsqueeze(1)
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t]).unsqueeze(1)

    x_t = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
    return x_t, noise


# -------------------------
# DATA
# -------------------------
full_dataset = TVSumShotDataset(JSON_DIR, EMBEDDING_DIR)

train_size = int((1 - VAL_RATIO) * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print(f"Full dataset size: {len(full_dataset)} videos")
print(f"Train size: {len(train_dataset)} videos")
print(f"Val size: {len(val_dataset)} videos")


# -------------------------
# MODEL
# -------------------------
model = MaskDiffusionModel(
    embedding_dim=768,
    hidden_dim=512,
    timesteps=TIMESTEPS
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

betas, alphas, alpha_hat = get_diffusion_constants(TIMESTEPS)
alpha_hat = alpha_hat.to(DEVICE)

best_val_loss = float("inf")


# -------------------------
# TRAINING LOOP
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for E, scores, dur, valid, names in train_loader:
        E = E.to(DEVICE).float()
        scores = scores.to(DEVICE).float()
        dur = dur.to(DEVICE).float()
        valid = valid.to(DEVICE).float()

        t = torch.randint(0, TIMESTEPS, (E.shape[0],), device=DEVICE).long()

        x_t, noise = forward_diffusion_sample(scores, t, alpha_hat, DEVICE)

        noise_pred = model(x_t, E, dur, t)

        loss = ((noise_pred - noise) ** 2 * valid).sum() / valid.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for E, scores, dur, valid, names in val_loader:
            E = E.to(DEVICE).float()
            scores = scores.to(DEVICE).float()
            dur = dur.to(DEVICE).float()
            valid = valid.to(DEVICE).float()

            t = torch.randint(0, TIMESTEPS, (E.shape[0],), device=DEVICE).long()

            x_t, noise = forward_diffusion_sample(scores, t, alpha_hat, DEVICE)

            noise_pred = model(x_t, E, dur, t)

            loss = ((noise_pred - noise) ** 2 * valid).sum() / valid.sum()
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_diffusion_model.pth")
        print("Saved best model.")

print("Training complete.")