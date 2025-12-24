import torch
from diffusers import StableDiffusionPipeline

# --- CONFIGURATION ---
prompt = "a man in a suit"
# Using a fixed seed ensures the composition remains similar for comparison
seed = 42 
finetuned_path = "/workspace/kohya_ss/outputs/last.safetensors"

# --- 1. GENERATE WITH BASE MODEL (WITHOUT FINETUNING) ---
print("⏳ Loading Base Model (RunwayML SD 1.5)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

print("🎨 Generating Base Image...")
generator = torch.Generator("cuda").manual_seed(seed)
image_base = pipe(prompt, generator=generator, num_inference_steps=30).images[0]
image_base.save("/workspace/comparison_base_15epoch.png")
print("✅ Saved: /workspace/comparison_base_12epoch.png")

# --- 2. GENERATE WITH FINETUNED MODEL ---
print("\n⏳ Loading Fine-Tuned Model...")
# We overwrite the 'pipe' variable to save VRAM
pipe = StableDiffusionPipeline.from_single_file(
    finetuned_path,
    torch_dtype=torch.float16
).to("cuda")

print("🎨 Generating Fine-Tuned Image...")
# Reset generator to the SAME seed so we can compare fairly
generator = torch.Generator("cuda").manual_seed(seed)
image_tuned = pipe(prompt, generator=generator, num_inference_steps=30).images[0]
image_tuned.save("/workspace/comparison_finetuned15epochs.png")
print("✅ Saved: /workspace/comparison_finetuned.png")

print("\n✨ Done! Open the file manager to compare the two images.")




