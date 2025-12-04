import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import argparse
import os
from PIL import Image
import gradio as gr

def load_model(model_path, device="cuda"):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    
    # Try to load the fine-tuned model, fall back to base model if not found
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
    except:
        print(f"Could not load model from {model_path}, using base model")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
    
    # Use Euler Ancestral scheduler for better results
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Enable memory efficient attention if available
    if torch.cuda.is_available():
        pipe = pipe.to(device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("Xformers not available, continuing without it")
    
    # Enable CPU offload for lower VRAM usage
    try:
        pipe.enable_model_cpu_offload()
    except:
        pass
    
    return pipe

def generate_image(pipe, prompt, negative_prompt="", num_images=1, 
                   num_inference_steps=30, guidance_scale=7.5, height=512, width=512, seed=None):
    """Generate images from prompt"""
    
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    with torch.autocast("cuda") if torch.cuda.is_available() else torch.cpu.amp.autocast():
        images = pipe(
            prompt=[prompt] * num_images,
            negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images
    
    return images

def save_images(images, prompt, output_dir="./generated_images"):
    """Save generated images with prompt in filename"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, img in enumerate(images):
        # Clean prompt for filename
        clean_prompt = prompt[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = f"{clean_prompt}_{i}_{torch.randint(0, 10000, (1,)).item()}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        saved_paths.append(filepath)
        print(f"Saved image to {filepath}")
    
    return saved_paths

def gradio_interface(pipe):
    """Create Gradio interface for image generation"""
    
    def generate(prompt, negative_prompt, num_steps, guidance, seed, width, height):
        if seed == -1:
            seed = None
        
        images = generate_image(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=1,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            seed=seed
        )
        
        return images[0]
    
    with gr.Blocks(title="Stable Diffusion Image Generator") as demo:
        gr.Markdown("# 🎨 Stable Diffusion Image Generator")
        gr.Markdown("Generate images using your fine-tuned Stable Diffusion model")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What you don't want in the image...",
                    lines=2
                )
                
                with gr.Row():
                    num_steps = gr.Slider(10, 100, value=30, step=1, label="Inference Steps")
                    guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                
                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                
                generate_btn = gr.Button("Generate Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
        
        generate_btn.click(
            generate,
            inputs=[prompt, negative_prompt, num_steps, guidance, seed, width, height],
            outputs=[output_image]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["a beautiful sunset over mountains, digital art"],
                ["a cute cat wearing a hat, cartoon style"],
                ["futuristic cityscape at night, neon lights, cyberpunk"],
                ["portrait of an ancient warrior, detailed armor, realistic"],
            ],
            inputs=[prompt]
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    parser.add_argument("--model_path", type=str, default="./sd-finetuned",
                       help="Path to the trained model")
    parser.add_argument("--prompt", type=str, default="",
                       help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images to generate")
    parser.add_argument("--num_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./generated_images",
                       help="Directory to save generated images")
    parser.add_argument("--gradio", action="store_true",
                       help="Launch Gradio web interface")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    pipe = load_model(args.model_path, device)
    
    if args.gradio:
        # Launch Gradio interface
        demo = gradio_interface(pipe)
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
    else:
        # Generate from command line arguments
        if args.prompt:
            images = generate_image(
                pipe=pipe,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_images=args.num_images,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                seed=args.seed
            )
            
            saved_paths = save_images(images, args.prompt, args.output_dir)
            
            # Display first image
            if images:
                images[0].show()
        else:
            print("Please provide a prompt using --prompt or use --gradio for web interface")

if __name__ == "__main__":
    main()
