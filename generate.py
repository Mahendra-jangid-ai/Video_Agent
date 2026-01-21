import os
import torch
import numpy as np
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.utils import export_to_video
from moviepy import VideoFileClip, concatenate_videoclips

# ---------------- SETTINGS ----------------
TOTAL_SCENES = 20        # 20 × 3 sec = 60 sec
FRAMES_PER_SCENE = 24   # 3 sec @ 8 fps
FPS = 8
OUTPUT_FINAL = "final_1_minute_video.mp4"

BASE_PROMPT = (
    "A cinematic realistic video of the same young Indian man working on a laptop, "
    "futuristic AI hologram screens, earning money using AI, night room, soft rim light, "
    "ultra smooth camera motion, film look, same face, same clothes, same environment"
)

# ---------------- LOAD MODEL ----------------
print("🔵 Loading AnimateDiff model (first time will download big files)...")

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5",
    torch_dtype=torch.float32
)

pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

# ---------------- GENERATE SCENES ----------------
scene_files = []

for i in range(TOTAL_SCENES):
    print(f"\n🎬 Generating Scene {i+1}/{TOTAL_SCENES} (3 sec)...")

    prompt = f"{BASE_PROMPT}, continuous story, scene {i+1}, subtle motion, same person"

    output = pipe(
        prompt=prompt,
        num_frames=FRAMES_PER_SCENE,
        guidance_scale=7.5,
        num_inference_steps=20
    )

    frames = []
    for frame in output.frames:
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frames.append(frame)

    scene_name = f"scene_{i+1:02d}.mp4"
    export_to_video(frames, scene_name, fps=FPS)
    scene_files.append(scene_name)

    print(f"✅ Saved: {scene_name}")

# ---------------- MERGE ALL SCENES ----------------
print("\n🧩 Merging all scenes into 1 minute cinematic video...")

clips = [VideoFileClip(f) for f in scene_files]
final_clip = concatenate_videoclips(clips, method="compose")
final_clip.write_videofile(OUTPUT_FINAL, fps=FPS)

print("\n🎉 DONE!")
print("📁 Final Video:", OUTPUT_FINAL)
print("⏱ Duration: ~60 seconds")
