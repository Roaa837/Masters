import torch
from model import DiffusionModel   # your model class

# load model
model = DiffusionModel()
model.load_state_dict(torch.load("diffusion_model.pth"))
model.eval()

# example input
E = torch.randn(40,1280)   # replace with real embeddings
mask = torch.ones(40)

with torch.no_grad():
    scores = model(E.unsqueeze(0), mask.unsqueeze(0))

scores = scores.squeeze()

print(scores)