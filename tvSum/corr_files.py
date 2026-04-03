import os
import subprocess

VIDEO_DIR = "data/SumMe"

corrupted = []
valid = []

for file in os.listdir(VIDEO_DIR):

    if not file.endswith(".mp4"):
        continue

    path = os.path.join(VIDEO_DIR, file)

    print(f"Checking: {file}")

    # run ffmpeg check
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.stderr:
        print(f"❌ Corrupted: {file}")
        corrupted.append(file)
    else:
        print(f"✅ OK: {file}")
        valid.append(file)

print("\n--- SUMMARY ---")
print(f"Valid videos: {len(valid)}")
print(f"Corrupted videos: {len(corrupted)}")

# save list
with open("corrupted_videos.txt", "w") as f:
    for v in corrupted:
        f.write(v + "\n")