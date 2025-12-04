import os
import glob
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup
import random

class TextImageDataset(Dataset):
    def __init__(self, image_dir, caption_dir, transform=None, image_size=512):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.image_size = image_size
        
        # Get all image files
        self.image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                          glob.glob(os.path.join(image_dir, "*.png")) + \
                          glob.glob(os.path.join(image_dir, "*.jpeg"))
        
        # Filter to ensure corresponding caption exists
        self.valid_pairs = []
        for img_path in self.image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            caption_path = os.path.join(caption_dir, f"{base_name}.txt")
            if os.path.exists(caption_path):
                self.valid_pairs.append((img_path, caption_path))
        
        print(f"Found {len(self.valid_pairs)} valid image-caption pairs")
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, caption_path = self.valid_pairs[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Load caption
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        # If caption is empty, use filename
        if not caption:
            caption = os.path.splitext(os.path.basename(img_path))[0]
        
        return {"pixel_values": image, "caption": caption}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]
    return {"pixel_values": pixel_values, "caption": captions}

def train(args):
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        project_dir=args.output_dir
    )
    
    # Initialize wandb if enabled
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project="stable-diffusion-finetune", config=args)
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet"
    )
    
    # Freeze text encoder for efficiency (optional)
    if args.freeze_text_encoder:
        text_encoder.requires_grad_(False)
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    # Enable xformers memory efficient attention if available
    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except:
            print("Xformers not available, continuing without it")
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # Create dataset and dataloader
    dataset = TextImageDataset(
        image_dir=args.train_data_dir,
        caption_dir=args.caption_dir,
        image_size=args.resolution
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )
    
    # Prepare scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move text encoder to device
    text_encoder.to(accelerator.device)
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        if not args.freeze_text_encoder:
            text_encoder.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (bsz,), device=latents.device
                ).long()
                
                # Add noise to the latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                text_inputs = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                text_embeddings = text_encoder(text_inputs)[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % args.logging_steps == 0:
                        logs = {
                            "loss": loss.detach().item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "step": global_step,
                            "epoch": epoch
                        }
                        if args.use_wandb:
                            wandb.log(logs)
                        progress_bar.set_postfix(**logs)
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    print(f"Saved checkpoint to {save_path}")
                
                # Early stopping if time is limited
                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    if accelerator.is_main_process:
        # Create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
        )
        
        # Save the model
        pipeline.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion")
    
    # Dataset arguments
    parser.add_argument("--train_data_dir", type=str, default="./training_dataset/images",
                       help="Directory containing training images")
    parser.add_argument("--caption_dir", type=str, default="./training_dataset",
                       help="Directory containing caption files")
    parser.add_argument("--output_dir", type=str, default="./sd-finetuned",
                       help="Output directory for model checkpoints")
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Path to pretrained model or model identifier")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Image resolution for training")
    parser.add_argument("--freeze_text_encoder", action="store_true",
                       help="Freeze text encoder during training")
    
    # Training arguments
    parser.add_argument("--train_batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--max_train_steps", type=int, default=2000,
                       help="Total number of training steps")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=100,
                       help="Number of warmup steps for learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # Other arguments
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of workers for dataloader")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every X steps")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use wandb for logging")
    parser.add_argument("--enable_xformers", action="store_true",
                       help="Enable xformers memory efficient attention")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate steps based on 5 hours (adjust based on your hardware)
    # Assuming ~2 seconds per step on your 32GB GPU
    if args.max_train_steps == 2000:  # default
        # 5 hours * 3600 seconds / 2 seconds per step ≈ 9000 steps max
        # But we'll use 2000 as a reasonable target
        pass
    
    train(args)

if __name__ == "__main__":
    main()
