"""
Final Generation Script for Anime Diffusion Model
Simple and reliable image generation
"""

import torch
from torchvision.utils import save_image
from diffusers import DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, AutoencoderKL
import os
from PIL import Image
import argparse
import numpy as np

class AnimeGenerator:
    def __init__(self, checkpoint_path, config_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing generator on {self.device}")
        
        # Configuration
        self.image_size = 512
        self.latent_size = self.image_size // 8
        
        # Load models
        self.load_models(checkpoint_path)
        
        # Sampler
        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        
        print("Generator ready!")
    
    def load_models(self, checkpoint_path):
        """Load all necessary models"""
        # Load text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.eval()
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        
        # Initialize U-Net (must match training architecture)
        self.unet = UNet2DConditionModel(
            sample_size=self.latent_size,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 384, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=768,
        ).to(self.device)
        
        # Load trained weights
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.unet.load_state_dict(checkpoint)
        
        self.unet.eval()
        print("Model loaded successfully")
    
    def encode_text(self, prompt, negative_prompt=""):
        """Encode text prompt to embeddings"""
        # Tokenize
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]
        
        # Negative prompt
        if negative_prompt:
            neg_input = self.tokenizer(
                [negative_prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                neg_embeddings = self.text_encoder(neg_input.input_ids)[0]
            text_embeddings = torch.cat([neg_embeddings, text_embeddings])
        
        return text_embeddings
    
    def decode_latents(self, latents):
        """Decode latents to images"""
        with torch.no_grad():
            latents = latents / 0.18215
            images = self.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)  # [0, 1]
        return images
    
    def generate(
        self,
        prompt,
        negative_prompt="",
        num_steps=30,
        guidance_scale=7.5,
        seed=42,
        num_images=1
    ):
        """Generate images from text prompt"""
        print(f"\nGenerating: '{prompt}'")
        print(f"Steps: {num_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Encode text
        text_embeddings = self.encode_text(prompt, negative_prompt)
        
        # Prepare latents
        latents = torch.randn(
            (num_images, 4, self.latent_size, self.latent_size),
            device=self.device
        )
        
        # Configure scheduler
        self.scheduler.set_timesteps(num_steps, device=self.device)
        
        # Denoising loop
        print("Denoising...")
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t.expand(latent_model_input.shape[0]),
                    encoder_hidden_states=text_embeddings.repeat(latent_model_input.shape[0], 1, 1)
                ).sample
            
            # Apply guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Progress
            if (i + 1) % (num_steps // 5) == 0:
                print(f"  Step {i+1}/{num_steps}")
        
        # Decode to images
        images = self.decode_latents(latents)
        
        print("Generation complete!")
        return images

def save_images(images, prompt, output_dir="./generated", seed=42):
    """Save generated images"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, img in enumerate(images):
        # Convert tensor to PIL
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Create filename
        safe_prompt = "".join(c for c in prompt[:40] if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"{safe_prompt}_{seed}_{i}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save
        img_pil.save(filepath)
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_paths

def main():
    parser = argparse.ArgumentParser(description="Generate anime images")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative", type=str, default="", help="Negative prompt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num", type=int, default=1, help="Number of images")
    parser.add_argument("--output", type=str, default="./generated", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = AnimeGenerator(args.checkpoint)
    
    # Generate images
    images = generator.generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        num_images=args.num
    )
    
    # Save images
    save_images(images, args.prompt, args.output, args.seed)
    
    print(f"\nDone! Generated {args.num} image(s)")

# Example prompts for your generic captions
EXAMPLE_PROMPTS = [
    "anime character with red hair",
    "a person standing in a room",
    "anime girl with blue eyes",
    "two people talking",
    "a person holding a sword",
    "anime style portrait",
    "character with black hair",
    "person wearing school uniform"
]

if __name__ == "__main__":
    print("=" * 60)
    print("ANIME DIFFUSION GENERATOR")
    print("=" * 60)
    print("\nExample prompts you can use:")
    for i, prompt in enumerate(EXAMPLE_PROMPTS[:4]):
        print(f"  {i+1}. {prompt}")
    
    print("\nTo generate:")
    print("  python generate_final.py --prompt 'anime character' --checkpoint anime_checkpoints/anime_model_best.pth")
    
    # Run if called directly
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("\nNo arguments provided. Use --help for usage.")
