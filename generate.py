"""
Advanced Image Generation with DDIM/DDPM Sampling
Supports multiple sampling methods and quality enhancements
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path to import model
sys.path.append('.')

from train import EnhancedUNet, DiffusionModel, get_timestep_embedding

class Generator:
    def __init__(self, config_path=None, checkpoint_path=None):
        # Load configuration
        if config_path:
            from train import Config
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = type('Config', (), config_dict)()
        else:
            from train import Config
            self.config = Config()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load text encoder
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model)
        self.text_encoder = AutoModel.from_pretrained(self.config.text_model).to(self.device)
        self.text_encoder.eval()
        
        # Initialize U-Net
        self.unet = EnhancedUNet(self.config).to(self.device)
        
        # Load checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.unet.eval()
        
        # Diffusion parameters
        self.betas = get_beta_schedule(self.config.timesteps, self.config.beta_schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        print(f"Generator initialized on {self.device}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both full checkpoint and state_dict
        if 'unet_state_dict' in checkpoint:
            state_dict = checkpoint['unet_state_dict']
        elif 'ema_unet_state_dict' in checkpoint:
            state_dict = checkpoint['ema_unet_state_dict']
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle missing keys
        missing_keys, unexpected_keys = self.unet.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
        
        print(f"Loaded checkpoint from {path}")
    
    def encode_text(self, prompt, negative_prompt=""):
        """Encode text prompt with optional negative prompt"""
        # Tokenize
        text_inputs = self.tokenizer(
            [prompt],
            max_length=self.config.caption_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(**text_inputs).last_hidden_state
        
        # Handle negative prompt
        if negative_prompt:
            neg_inputs = self.tokenizer(
                [negative_prompt],
                max_length=self.config.caption_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                neg_embeddings = self.text_encoder(**neg_inputs).last_hidden_state
            
            # Concatenate for classifier-free guidance
            text_embeddings = torch.cat([neg_embeddings, text_embeddings])
        
        return text_embeddings
    
    def ddim_sample(
        self,
        prompt,
        num_steps=50,
        guidance_scale=7.5,
        seed=42,
        batch_size=1,
        height=None,
        width=None,
        eta=0.0
    ):
        """
        DDIM sampling - faster with good quality
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Text encoding
        text_embeddings = self.encode_text(prompt)
        
        # Prepare latent
        latent_height = height // 8 if height else self.config.image_size // 8
        latent_width = width // 8 if width else self.config.image_size // 8
        
        latents = torch.randn(
            (batch_size, 4, latent_height, latent_width),
            device=self.device
        )
        
        # DDIM sampling setup
        timesteps = torch.linspace(
            self.config.timesteps - 1, 0, num_steps, device=self.device
        ).long()
        
        # Reverse process
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            latent_model_input = latent_model_input / (self.alphas_cumprod[t] ** 0.5)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    torch.full((latent_model_input.shape[0],), t, device=self.device, dtype=torch.long),
                    text_embeddings.repeat(latent_model_input.shape[0], 1, 1)
                )
            
            # Apply guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDIM update
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=self.device)
            
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=self.device)
            
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Predicted x0
            pred_x0 = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            
            # Direction pointing to xt
            dir_xt = (1 - alpha_prod_t_prev - eta ** 2 * beta_prod_t_prev) ** 0.5 * noise_pred
            
            # Update latents
            latents = alpha_prod_t_prev ** 0.5 * pred_x0 + dir_xt
            
            if eta > 0:
                noise = torch.randn_like(latents)
                latents = latents + eta * beta_prod_t_prev ** 0.5 * noise
        
        return latents
    
    def dpm_solver_sample(
        self,
        prompt,
        num_steps=20,
        guidance_scale=7.5,
        seed=42,
        batch_size=1,
        order=2
    ):
        """
        DPM-Solver sampling - fast with high quality
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Text encoding
        text_embeddings = self.encode_text(prompt)
        
        # Prepare latent
        latents = torch.randn(
            (batch_size, 4, self.config.image_size // 8, self.config.image_size // 8),
            device=self.device
        )
        
        # Time steps
        timesteps = torch.linspace(
            self.config.timesteps - 1, 0, num_steps + 1, device=self.device
        ).long()
        
        # DPM-Solver steps
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Expand for guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    torch.full((latent_model_input.shape[0],), t, device=self.device, dtype=torch.long),
                    text_embeddings.repeat(latent_model_input.shape[0], 1, 1)
                )
            
            # Apply guidance
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DPM-Solver update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next]
            
            sigma_t = (1 - alpha_t) ** 0.5
            sigma_t_next = (1 - alpha_t_next) ** 0.5
            
            # First order (Euler)
            if order == 1:
                x0_t = (latents - sigma_t * noise_pred) / (alpha_t ** 0.5)
                latents = alpha_t_next ** 0.5 * x0_t + sigma_t_next * noise_pred
            
            # Second order
            elif order == 2 and i < num_steps - 1:
                t_mid = (t + t_next) // 2
                alpha_t_mid = self.alphas_cumprod[t_mid]
                
                # First step to midpoint
                x0_t = (latents - sigma_t * noise_pred) / (alpha_t ** 0.5)
                latents_mid = alpha_t_mid ** 0.5 * x0_t + (1 - alpha_t_mid) ** 0.5 * noise_pred
                
                # Predict noise at midpoint
                with torch.no_grad():
                    noise_pred_mid = self.unet(
                        latents_mid.unsqueeze(0),
                        torch.tensor([t_mid], device=self.device),
                        text_embeddings
                    ).squeeze(0)
                
                # Apply guidance at midpoint
                if guidance_scale > 1:
                    noise_pred_mid = noise_pred_mid + guidance_scale * (noise_pred_mid - noise_pred_mid)
                
                # Final step
                latents = alpha_t_next ** 0.5 * x0_t + sigma_t_next * noise_pred_mid
        
        return latents
    
    def decode_latents(self, latents):
        """Decode latents to images (simplified - add VAE for better quality)"""
        # Scale latents
        latents = latents / 0.18215
        
        # Simple upsampling and color mapping
        images = []
        for latent in latents:
            # Upsample to image size
            img = F.interpolate(
                latent.unsqueeze(0),
                size=(self.config.image_size, self.config.image_size),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            
            # Convert to RGB (simplified - assumes 4 channels)
            if img.shape[0] == 4:
                # Simple mapping from latent to RGB
                rgb = torch.stack([
                    img[0] - img[1] + img[2],  # R
                    img[1] + img[2] - img[0],  # G  
                    img[2] - img[0] + img[1]   # B
                ])
            else:
                rgb = img[:3]
            
            # Normalize to [0, 1]
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            images.append(rgb)
        
        return torch.stack(images)
    
    def generate(
        self,
        prompt,
        negative_prompt="",
        num_samples=1,
        num_steps=50,
        guidance_scale=7.5,
        seed=42,
        sampler="ddim",
        height=None,
        width=None,
        output_dir="./outputs"
    ):
        """
        Main generation function
        """
        print(f"Generating {num_samples} image(s) for prompt: '{prompt}'")
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_images = []
        
        for sample_idx in range(0, num_samples, self.config.batch_size):
            current_batch = min(self.config.batch_size, num_samples - sample_idx)
            
            # Generate latents
            if sampler.lower() == "ddim":
                latents = self.ddim_sample(
                    prompt=prompt,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=seed + sample_idx,
                    batch_size=current_batch,
                    height=height,
                    width=width
                )
            elif sampler.lower() == "dpm":
                latents = self.dpm_solver_sample(
                    prompt=prompt,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=seed + sample_idx,
                    batch_size=current_batch
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
            
            # Decode to images
            images = self.decode_latents(latents)
            all_images.append(images)
            
            # Save images
            for i, image in enumerate(images):
                img_idx = sample_idx + i
                
                # Convert to PIL
                img_np = image.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                # Save
                safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '_')).rstrip()
                filename = f"{safe_prompt}_{seed}_{img_idx}.png"
                save_path = os.path.join(output_dir, filename)
                img_pil.save(save_path)
                
                print(f"Saved: {save_path}")
        
        return torch.cat(all_images, dim=0) if all_images else None

# ==================== UTILITY FUNCTIONS ====================
def get_beta_schedule(num_timesteps, schedule="cosine"):
    """Get beta schedule for diffusion"""
    if schedule == "linear":
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

def save_image(tensor, path):
    """Save tensor as image"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor, path)

# ==================== COMMAND LINE INTERFACE ====================
def main():
    parser = argparse.ArgumentParser(description="Generate images with diffusion model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "dpm"], help="Sampling method")
    parser.add_argument("--height", type=int, default=None, help="Output height")
    parser.add_argument("--width", type=int, default=None, help="Output width")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./generated", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = Generator(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    # Generate images
    images = generator.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        sampler=args.sampler,
        height=args.height,
        width=args.width,
        output_dir=args.output_dir
    )
    
    print(f"\nGenerated {args.num_samples} image(s) in '{args.output_dir}'")

if __name__ == "__main__":
    main()

# ==================== EXAMPLE USAGE ====================
"""
# Quick test
if __name__ == "__test__":
    # Initialize
    gen = Generator(checkpoint_path="./checkpoints/model_best.pth")
    
    # Generate single image
    images = gen.generate(
        prompt="a beautiful sunset over mountains, digital art",
        num_samples=1,
        num_steps=50,
        guidance_scale=7.5,
        seed=42
    )
    
    # Batch generation
    images = gen.generate(
        prompt="a portrait of a cyberpunk samurai",
        num_samples=4,
        num_steps=30,
        sampler="dpm",
        guidance_scale=8.0,
        seed=123
    )
"""
