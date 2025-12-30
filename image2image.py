pip install Pillow

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import os
import sys

# --- CONFIGURATION ---
model_path = "/workspace/kohya_ss/outputs/last.safetensors"
input_image_path = "image.png"  # <--- PUT YOUR REALISTIC PHOTO HERE
output_dir = "anime_conversions"
trigger_word = "sfa " 

os.makedirs(output_dir, exist_ok=True)

# --- CHECK INPUT ---
if not os.path.exists(input_image_path):
    print(f"ERROR: Please place a realistic image named '{input_image_path}' in this folder.")
    sys.exit(1)

# --- LOAD MODEL ---
print(f"Loading Model from {model_path}...")
try:
    # Notice we use 'StableDiffusionXLImg2ImgPipeline' this time
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to("cuda")
    print("Model loaded in Img2Img mode!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- LOAD INITIAL IMAGE ---
# We load the realistic image and resize it to 1024x1024 for SDXL
init_image = load_image(input_image_path).convert("RGB")
init_image = init_image.resize((1024, 1024))

# --- PROMPT ---
# We describe the image but add the trigger word to force the style
prompt = f"{trigger_word} highly detailed, vibrant colors"
negative_prompt = "photo, realistic, 3d, rendering, bad anatomy, blurry"

# --- GENERATION LOOP (TESTING STRENGTHS) ---
# Strength determines how much of the original image is preserved.
# 0.0 = Exact copy of original. 1.0 = Ignore original, generate from scratch.
strengths = [0.15,0.2,0.3,0.4,0.6,0.8] 

print(f"\nConverting image with prompt: '{prompt}'")

for strength in strengths:
    print(f"Generating with Strength {strength}...")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    save_path = os.path.join(output_dir, f"anime_strength_{strength}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")

print(f"\nDone! Check '{output_dir}'. Comparison:")
print("Strength 0.4: Keeps structure, weak anime effect.")
print("Strength 0.8: Strong anime effect, might change pose/face.")
