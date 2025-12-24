import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# --- CONFIGURATION ---
# Path to your fine-tuned LoRA file
lora_path = "/workspace/kohya_ss/outputs/spy.safetensors"

# Your Prompt (Add your trigger word if you used one, e.g., 'zwx')
prompt = "1girl with spectacles"
negative_prompt = "low quality, bad anatomy, worst quality, blurry, 3d, realistic, sketch"

# --- 1. LOAD BASE MODEL ---
print("⏳ Loading Base Model (RunwayML)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Use a better scheduler for Anime (DPM++ 2M Karras)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, 
    use_karras_sigmas=True
)

# --- 2. LOAD LORA ---
print(f"🔗 Loading LoRA from: {lora_path}")
try:
    pipe.load_lora_weights(lora_path)
    print("✅ LoRA loaded successfully!")
except Exception as e:
    print(f"❌ Error loading LoRA: {e}")
    print("NOTE: If this fails, you might have trained a 'Full Checkpoint' instead of a LoRA.")
    print("      In that case, use the 'from_single_file' code I gave you earlier.")
    exit()

# --- 3. GENERATE ---
print("🎨 Generating Image...")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,  # 30 is good for DPM++
    guidance_scale=7.5,      # How strictly to follow the prompt
    cross_attention_kwargs={"scale": 1.0} # 1.0 = Full LoRA strength. Try 0.7 if style is too strong.
).images[0]

# --- 4. SAVE ---
output_path = f"/workspace/{prompt}.png"
image.save(output_path)
print(f"✨ Image saved to: {output_path}")
