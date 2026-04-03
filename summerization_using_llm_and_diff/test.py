import torch 


data = torch.load("..\dataset\_Floral_ by ADIDAS - AI Spec AD.pt")

print(data.keys())
print(data.values())
print("Embeddings shape:", data["E"].shape)
print("Durations shape:", data["dur"].shape)
print("Mask shape:", data["mask"].shape)
print(data["mask"])
print(data["dur"])