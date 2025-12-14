"""
Final Training Script for Anime Diffusion Model
Optimized for 32GB GPU with 150K images and basic captions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.models as models
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
import math

# ==================== CONFIGURATION ====================
class Config:
    # Paths
    training_dataset = "./training_dataset"  # Main folder
    images_folder = "images"                  # Images inside this
    captions_folder = "."                     # Captions in training_dataset folder
    
    # Model
    pretrained_model = "runwayml/stable-diffusion-v1-5"  # For text encoder
    image_size = 512                          # Optimal for anime
    latent_size = image_size // 8             # For U-Net
    in_channels = 4                           # Latent space
    out_channels = 4
    layers_per_block = 2
    
    # U-Net Architecture (optimized for anime)
    block_out_channels = (128, 256, 384, 512)  # Balanced for 32GB
    down_block_types = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    cross_attention_dim = 768
    
    # Training
    batch_size = 6                            # Fits in 32GB
    micro_batch = 2                           # Gradient accumulation
    gradient_accumulation = batch_size // micro_batch
    learning_rate = 5e-5                      # Lower for anime
    warmup_steps = 1000
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    max_grad_norm = 1.0
    
    # Schedule
    epochs = 30                               # Good for anime
    timesteps = 1000
    beta_schedule = "squaredcos_cap_v2"       # Best for anime
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = "fp16"
    seed = 42
    
    # Checkpoints
    output_dir = "./anime_checkpoints"
    save_every = 2500
    sample_every = 1000
    log_every = 100

config = Config()

# Set seed
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

# ==================== DATASET ====================
class SimpleAnimeDataset(Dataset):
    """Simplified dataset for generic captions"""
    
    def __init__(self, base_path, image_size=512, augment=True):
        self.base_path = base_path
        self.image_size = image_size
        
        # Find all images
        images_dir = os.path.join(base_path, config.images_folder)
        self.image_files = []
        self.caption_files = []
        
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext) for ext in valid_exts):
                img_path = os.path.join(images_dir, file)
                caption_path = os.path.join(base_path, 
                                           os.path.splitext(file)[0] + '.txt')
                
                # Check if caption exists
                if os.path.exists(caption_path):
                    self.image_files.append(img_path)
                    self.caption_files.append(caption_path)
                else:
                    # Try with .jpg extension if image is .png
                    alt_caption = os.path.join(base_path, 
                                              os.path.splitext(file)[0] + '.jpg.txt')
                    if os.path.exists(alt_caption):
                        self.image_files.append(img_path)
                        self.caption_files.append(alt_caption)
        
        print(f"Found {len(self.image_files)} images with captions")
        
        # Load a few captions to understand format
        sample_captions = []
        for i in range(min(5, len(self.caption_files))):
            with open(self.caption_files[i], 'r', encoding='utf-8') as f:
                sample_captions.append(f.read().strip())
        print(f"Sample captions: {sample_captions}")
        
        # Anime-optimized transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(image_size + 32),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                # Anime-specific color adjustments
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # To [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.pretrained_model,
            subfolder="tokenizer"
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback to a different image if corrupted
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply transform
        pixel_values = self.transform(image)
        
        # Load and enhance caption
        with open(self.caption_files[idx], 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        # Enhance generic captions for anime
        if len(caption) < 10:  # Very short caption
            enhanced = f"anime, {caption}, anime artwork, simple"
        else:
            enhanced = f"anime style, {caption}, anime artwork, clear"
        
        # Tokenize
        text_inputs = self.tokenizer(
            enhanced,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "caption": enhanced
        }

# ==================== MODEL INITIALIZATION ====================
print("Initializing models...")

# Load text encoder (frozen)
text_encoder = CLIPTextModel.from_pretrained(
    config.pretrained_model,
    subfolder="text_encoder"
).to(config.device)
text_encoder.requires_grad_(False)

# Initialize U-Net
unet = UNet2DConditionModel(
    sample_size=config.latent_size,
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    layers_per_block=config.layers_per_block,
    block_out_channels=config.block_out_channels,
    down_block_types=config.down_block_types,
    up_block_types=config.up_block_types,
    cross_attention_dim=config.cross_attention_dim,
).to(config.device)

print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")

# Noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=config.timesteps,
    beta_schedule=config.beta_schedule,
    prediction_type="epsilon"
)

# VAE for encoding/decoding (frozen)
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    config.pretrained_model,
    subfolder="vae"
).to(config.device)
vae.requires_grad_(False)

# Optimizer
optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=config.learning_rate,
    betas=(config.adam_beta1, config.adam_beta2),
    weight_decay=config.adam_weight_decay
)

# Mixed precision
scaler = GradScaler(enabled=config.mixed_precision == "fp16")

# ==================== TRAINING FUNCTIONS ====================
def encode_images(images):
    """Encode images to latents using VAE"""
    with torch.no_grad():
        # VAE expects images in [0, 1], but we have [-1, 1]
        images = (images + 1) / 2
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor
    return latents

def decode_latents(latents):
    """Decode latents to images"""
    with torch.no_grad():
        latents = latents / 0.18215
        images = vae.decode(latents).sample
        images = (images * 2) - 1  # Back to [-1, 1]
    return images

def train_epoch(dataloader, epoch, global_step):
    """Train for one epoch"""
    unet.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        images = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        
        # Encode images to latents
        with torch.no_grad():
            latents = encode_images(images)
            
            # Get text embeddings
            text_embeddings = text_encoder(input_ids)[0]
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=config.device
        ).long()
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        with autocast(enabled=config.mixed_precision == "fp16"):
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Simple MSE loss
            loss = F.mse_loss(noise_pred, noise)
            loss = loss / config.gradient_accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.gradient_accumulation == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update progress
        total_loss += loss.item() * config.gradient_accumulation
        current_loss = total_loss / (batch_idx + 1)
        
        if global_step % config.log_every == 0:
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "step": global_step,
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Save checkpoint
        if global_step % config.save_every == 0 and global_step > 0:
            save_checkpoint(global_step, epoch, current_loss)
        
        # Generate sample
        if global_step % config.sample_every == 0 and global_step > 0:
            generate_sample(global_step)
        
        global_step += 1
    
    return global_step, total_loss / len(dataloader)

def save_checkpoint(step, epoch, loss):
    """Save model checkpoint"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.__dict__
    }
    
    path = os.path.join(config.output_dir, f"anime_model_step_{step}.pth")
    torch.save(checkpoint, path)
    print(f"\nSaved checkpoint to {path}")

def generate_sample(step, prompt="anime character with colorful hair"):
    """Generate a sample image during training"""
    unet.eval()
    
    with torch.no_grad():
        # Tokenize prompt
        text_inputs = CLIPTokenizer.from_pretrained(
            config.pretrained_model,
            subfolder="tokenizer"
        )([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        input_ids = text_inputs.input_ids.to(config.device)
        
        # Get text embeddings
        text_embeddings = text_encoder(input_ids)[0]
        
        # Create noise
        latents = torch.randn((1, 4, config.latent_size, config.latent_size), 
                             device=config.device)
        
        # DDIM sampling (faster)
        from diffusers import DDIMScheduler
        ddim = DDIMScheduler.from_pretrained(config.pretrained_model, subfolder="scheduler")
        ddim.set_timesteps(30)
        
        for t in ddim.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = ddim.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t.expand(2),
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = ddim.step(noise_pred, t, latents).prev_sample
        
        # Decode to image
        image = decode_latents(latents)
        image = (image[0].cpu().clamp(-1, 1) + 1) / 2  # [0, 1]
        
        # Save
        from torchvision.utils import save_image
        sample_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        save_image(image, os.path.join(sample_dir, f"sample_step_{step}.png"))
        print(f"Generated sample at step {step}")
    
    unet.train()

# ==================== MAIN TRAINING ====================
def main():
    print(f"Starting training on {config.device}")
    print(f"Image size: {config.image_size}, Batch size: {config.batch_size}")
    print(f"Dataset: {config.training_dataset}")
    
    # Create dataset
    dataset = SimpleAnimeDataset(config.training_dataset, config.image_size)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.micro_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Training with {len(dataset)} images")
    print(f"Steps per epoch: {len(dataloader) // config.gradient_accumulation}")
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*50}")
        
        # Train
        global_step, epoch_loss = train_epoch(dataloader, epoch, global_step)
        
        print(f"Epoch {epoch+1} completed")
        print(f"Average loss: {epoch_loss:.4f}")
        print(f"Global step: {global_step}")
        
        # Save epoch checkpoint
        save_checkpoint(global_step, epoch + 1, epoch_loss)
        
        # Update best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(config.output_dir, "anime_model_best.pth")
            torch.save(unet.state_dict(), best_path)
            print(f"New best model saved with loss {best_loss:.4f}")
    
    # Save final model
    final_path = os.path.join(config.output_dir, "anime_model_final.pth")
    torch.save(unet.state_dict(), final_path)
    print(f"\nTraining complete!")
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()
