import os
import torch
from pathlib import Path
from diffussion import (
    prepering_inputs_for_diffussion,
    extract_durations,
    building_mask_from_results
)

SEGMENTATION_DIR = "..\data\segmentation"
RESULTS_DIR = "..\data\llm_results"
OUTPUT_DIR = "..\dataset"

Path(OUTPUT_DIR).mkdir(exist_ok=True)


def build_dataset():

    video_folders = os.listdir(SEGMENTATION_DIR)

    for video_name in video_folders:

        video_path = os.path.join(SEGMENTATION_DIR, video_name)

        if not os.path.isdir(video_path):
            continue

        print(f"\n Processing {video_name}")



        segmentation_json = os.path.join(
            video_path,
            f"{video_name}.json"
        )

        keyframes_path = os.path.join(
            video_path,
            "keyframes"
        )

        results_json = os.path.join(
            RESULTS_DIR,
            f"{video_name}_results.json"
        )


        if not os.path.exists(segmentation_json):
            print("⚠️ segmentation json missing")
            continue

        if not os.path.exists(results_json):
            print("⚠️ llm results missing")
            continue

        if not os.path.exists(keyframes_path):
            print("⚠️ keyframes folder missing")
            continue

        try:


            E = prepering_inputs_for_diffussion(
                segmentation_json,
                keyframes_path
            )


            dur = extract_durations(segmentation_json)


            mask = building_mask_from_results(results_json)


            if not (E.shape[0] == dur.shape[0] == mask.shape[0]):
                print("Shot alignment mismatch")
                print("E:", E.shape)
                print("dur:", dur.shape)
                print("mask:", mask.shape)
                continue



            save_path = os.path.join(
                OUTPUT_DIR,
                f"{video_name}.pt"
            )

            torch.save({
                "E": E,
                "dur": dur,
                "mask": mask
            }, save_path)

            print(" Saved:", save_path)

        except Exception as e:

            print(" Error processing video:", e)




if __name__ == "__main__":
    build_dataset()