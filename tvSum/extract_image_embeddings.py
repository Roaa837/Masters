import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

SEGMENTATION_DIR = "../data/segmentation"
OUTPUT_DIR = "../data/image_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading CLIP...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()

print("CLIP loaded.")


for video_folder in os.listdir(SEGMENTATION_DIR):

    video_path = os.path.join(SEGMENTATION_DIR, video_folder)
    keyframes_path = os.path.join(video_path, "keyframes")

    if not os.path.isdir(keyframes_path):
        print("No keyframes:", video_folder)
        continue

    image_files = sorted(
        os.listdir(keyframes_path),
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if len(image_files) == 0:
        print("No images:", video_folder)
        continue

    images = []

    for img_name in image_files:
        img_path = os.path.join(keyframes_path, img_name)
        image = Image.open(img_path).convert("RGB")
        images.append(image)

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features.cpu().numpy()

    save_path = os.path.join(OUTPUT_DIR, f"{video_folder}.npy")
    np.save(save_path, image_features)

    print("Saved:", save_path)

print("Done.")