pip install torch torchvision diffusers transformers accelerate safetensors invisible-watermark





import torch
from diffusers import StableDiffusionXLPipeline
import os
import sys

# --- CONFIGURATION ---
model_path = "/workspace/kohya_ss/outputs/last.safetensors"
output_dir = "generated_images_sdxl"
os.makedirs(output_dir, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading SDXL Fine-tune from: {model_path}")

# Check if file exists first
if not os.path.exists(model_path):
    print(f"CRITICAL ERROR: File not found at {model_path}")
    sys.exit(1)

try:
    # Use StableDiffusionXLPipeline for SDXL models
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        # SDXL usually doesn't need load_safety_checker=False explicitly 
        # as often as 1.4, but we can leave it out to let it auto-configure.
    )
    
    # With 48GB VRAM, we put the whole model on CUDA.
    pipe.to("cuda")
    
    # We do NOT enable CPU offload because you have plenty of VRAM.
    # This ensures maximum speed.
    print("SDXL Model loaded successfully!")

except Exception as e:
    print(f"\nERROR: Could not load the model.")
    print(f"Error details: {e}")
    sys.exit(1)

# --- PROMPT LIST ---
base_prompts = [
    "sfa a girl",
    "sfa a boy in a suit",
    "sfa a cat",
    "sfa a dog",
    "sfa a car in front of a building",
    "sfa a man standing on a street",
    "sfa a woman smiling",
    "sfa a child playing",
    "sfa a bicycle on the road",
    "sfa a house with trees",
    "sfa a person sitting on a bench",
    "sfa a couple walking together",
    "sfa a bird flying in the sky",
    "sfa a flower in a garden",
    "sfa a laptop on a desk",
    "sfa a cup of coffee",
    "sfa a city street at night",
    "sfa a mountain landscape",
    "sfa a beach with waves",
    "sfa a train at a station"
]

# --- GENERATION LOOP ---
print(f"\nStarting generation of {len(base_prompts)} images...")

for prompt in base_prompts:
    print(f"Generating: {prompt}")
    
    # SDXL Specifics:
    # 1. Height/Width: Must be 1024x1024 (or similar aspect ratios) for SDXL. 
    #    512x512 will look broken.
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=1024
    ).images[0]
    
    # Create a safe filename
    safe_name = "".join([c if c.isalnum() else "_" for c in prompt])
    safe_name = safe_name[:50]
    
    save_path = os.path.join(output_dir, f"{safe_name}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")

print(f"\nDone! Check the '{output_dir}' folder.")
