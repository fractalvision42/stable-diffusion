import os
import glob
import torch
import argparse
import math
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    DDPMScheduler, 
    AutoencoderKL,
    LMSDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup
import random
import numpy as np
from datetime import datetime

class TextImageDataset(Dataset):
    def __init__(self, image_dir, caption_dir, image_size=512):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.image_size = image_size
        
        print(f"🔍 Scanning dataset...")
        print(f"  Images: {image_dir}")
        print(f"  Captions: {caption_dir}")
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
        self.image_files = []
        
        for ext in image_extensions:
            found = glob.glob(os.path.join(image_dir, ext))
            self.image_files.extend(found)
        
        print(f"  Found {len(self.image_files)} images")
        
        # Match images with captions
        self.valid_pairs = []
        for img_path in self.image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            caption_file = os.path.join(caption_dir, f"{base_name}.txt")
            
            if os.path.exists(caption_file):
                try:
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    if caption:
                        self.valid_pairs.append((img_path, caption))
                except:
                    continue
        
        print(f"✅ Valid pairs: {len(self.valid_pairs)}")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("❌ No valid image-caption pairs found!")
        
        # Augmentations
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, caption = self.valid_pairs[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return {"pixel_values": image, "caption": caption}
        except:
            # Fallback to another random image
            return self.__getitem__(random.randint(0, len(self.valid_pairs)-1))

def main():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion (Optimized for 5 hours)")
    
    # Dataset
    parser.add_argument("--train_data_dir", type=str, default="./training_dataset/images",
                       help="Directory with images")
    parser.add_argument("--caption_dir", type=str, default="./training_dataset",
                       help="Directory with caption files")
    parser.add_argument("--output_dir", type=str, default="./sd-finetuned",
                       help="Output directory")
    
    # Model
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base model")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Training resolution")
    
    # Training (OPTIMIZED FOR 5 HOURS)
    parser.add_argument("--batch_size", type=int, default=4,  # Fits in 30GB
                       help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--warmup", type=int, default=100,
                       help="Warmup steps")
    
    # Other
    parser.add_argument("--workers", type=int, default=4,
                       help="Data loader workers")
    parser.add_argument("--save_every", type=int, default=200,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=20,
                       help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Calculate steps for 5 hours (estimated 4 sec/step on 30GB GPU)
    # 5 hours * 3600 seconds / 4 seconds per step ≈ 4500 steps
    # But with 8K images: 8000/4 = 2000 steps per epoch, 2 epochs = 4000 steps
    total_steps = 2000  # Conservative estimate for 5 hours
    
    print("\n" + "="*60)
    print("🚀 STABLE DIFFUSION FINE-TUNING")
    print("="*60)
    print(f"📊 Config:")
    print(f"  • Model: {args.pretrained_model}")
    print(f"  • Images: {args.train_data_dir}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • LR: {args.lr}")
    print(f"  • Target steps: {total_steps} (~5 hours)")
    print(f"  • Resolution: {args.resolution}")
    print(f"  • Output: {args.output_dir}")
    print("="*60)
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.grad_accum,
        log_with="wandb" if args.use_wandb else None,
        project_dir=args.output_dir
    )
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project="sd-finetune-5hr", config=vars(args))
    
    # Load models
    print("🔄 Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    
    # Freeze models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Dataset
    print("📦 Loading dataset...")
    dataset = TextImageDataset(args.train_data_dir, args.caption_dir, args.resolution)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr,
        weight_decay=1e-2
    )
    
    # Scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=total_steps
    )
    
    # Prepare with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Training
    print("🔥 Starting training...")
    global_step = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        unet.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}")
        
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Get batch
                images = batch["pixel_values"].to(accelerator.device)
                captions = batch["caption"]
                
                # Encode to latents
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode text
                text_inputs = tokenizer(
                    captions, 
                    padding="max_length", 
                    max_length=77, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                with torch.no_grad():
                    text_embeddings = text_encoder(text_inputs)[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                # Logging
                if global_step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    remaining = (total_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0
                    
                    logs = {
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "steps/sec": steps_per_sec,
                        "eta_minutes": remaining / 60
                    }
                    progress_bar.set_postfix(loss=loss.item(), lr=logs["lr"])
                    
                    if args.use_wandb and accelerator.is_main_process:
                        wandb.log(logs)
                
                # Save checkpoint
                if global_step % args.save_every == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    
                    # Save pipeline
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        vae=vae,
                        scheduler=noise_scheduler,
                        safety_checker=None,
                    )
                    pipeline.save_pretrained(os.path.join(save_path, "pipeline"))
                    print(f"\n💾 Saved checkpoint {global_step}")
                
                # Time check
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours > 4.8:  # Stop at ~4.8 hours to save final model
                    print(f"\n⏰ Time limit reached ({elapsed_hours:.1f} hours)")
                    break
            
            if global_step >= total_steps:
                break
        
        progress_bar.close()
        
        if global_step >= total_steps or elapsed_hours > 4.8:
            break
    
    # Save final model
    if accelerator.is_main_process:
        print("\n💾 Saving final model...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=noise_scheduler,
            safety_checker=None,
        )
        pipeline.save_pretrained(args.output_dir)
        
        # Save config
        training_time = time.time() - start_time
        with open(os.path.join(args.output_dir, "training_info.txt"), "w") as f:
            f.write(f"Training completed in {training_time/3600:.2f} hours\n")
            f.write(f"Total steps: {global_step}\n")
            f.write(f"Final loss: {loss.item():.4f}\n")
            f.write(f"Dataset size: {len(dataset)}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Learning rate: {args.lr}\n")
        
        print(f"✅ Training completed in {training_time/3600:.2f} hours")
        print(f"✅ Model saved to {args.output_dir}")
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
