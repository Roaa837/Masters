from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# Load CLIP
# --------------------------------------------------
print("Loading CLIP model...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.eval()

print("CLIP loaded.")


# --------------------------------------------------
# Paths
# --------------------------------------------------
SEGMENTATION_DIR = Path("../data/segmentation")
OUTPUT_DIR = Path("../data/embeddings")

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Process each video
# --------------------------------------------------
for video_folder in SEGMENTATION_DIR.iterdir():

    if not video_folder.is_dir():
        continue

    keyframes_dir = video_folder / "keyframes"

    if not keyframes_dir.exists():
        print("No keyframes for:", video_folder.name)
        continue

    print("\nProcessing video:", video_folder.name)

    image_paths = sorted(
        keyframes_dir.glob("*.jpg"),
        key=lambda x: int(x.stem.split("_")[-1])
    )

    if len(image_paths) == 0:
        print("No frames found.")
        continue

    images = []

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        images.append(image)

    inputs = clip_processor(images=images, return_tensors="pt", padding=True)

    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**inputs)

    image_embeddings = image_embeddings.cpu().numpy()

    embeddings_dict = {}

    for i, emb in enumerate(image_embeddings):
        embeddings_dict[str(i + 1)] = emb

    output_path = OUTPUT_DIR / f"{video_folder.name}.npz"

    np.savez(output_path, **embeddings_dict)

    print("Saved embeddings:", output_path)

print("\nAll embeddings generated.")