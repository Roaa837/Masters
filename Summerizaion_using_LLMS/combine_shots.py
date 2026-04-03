from moviepy import VideoFileClip, concatenate_videoclips
import os

def combine_shots(video_path, selected_shots):
    video = VideoFileClip(video_path)

    clips = []
    for shot in selected_shots:
        start = shot["data"]["start"]
        end = shot["data"]["end"]
        clips.append(video.subclipped(start, end))

    final = concatenate_videoclips(clips)

    # 🔹 build output name: originalName_summary.mp4
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base_name}_summary.mp4"

    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac"
    )

    # clean up (important on Windows)
    final.close()
    for c in clips:
        c.close()
    video.close()

    print(f"Summary video saved as: {output_path}")
