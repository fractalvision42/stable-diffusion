import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import argparse
import os
from PIL import Image
import gradio as gr
import time

class StableDiffusionGenerator:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading model from {model_path}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}, using base model")
            model_path = "runwayml/stable-diffusion-v1-5"
        
        # Load pipeline
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        except:
            print("❌ Failed to load fine-tuned model, using base model")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        
        # Optimize for speed
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_xformers_memory_efficient_attention()
        
        print("✅ Model loaded successfully!")
    
    def generate_image(self, prompt, negative_prompt="", steps=30, guidance=7.5, 
                      width=512, height=512, seed=-1, num_images=1):
        """Generate images with given parameters"""
        
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        start_time = time.time()
        with torch.autocast("cuda") if self.device == "cuda" else torch.cpu.amp.autocast():
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=num_images
            ).images
        
        gen_time = time.time() - start_time
        
        return images, seed, gen_time
    
    def save_images(self, images, prompt, output_dir="./generated"):
        """Save generated images"""
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        timestamp = int(time.time())
        
        for i, img in enumerate(images):
            # Clean filename
            clean_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{clean_prompt}_{timestamp}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            
            img.save(filepath)
            saved_paths.append(filepath)
        
        return saved_paths

def create_interface(generator):
    """Create Gradio interface"""
    
    def generate_ui(prompt, negative_prompt, steps, guidance, width, height, seed, num_images):
        images, used_seed, gen_time = generator.generate_image(
            prompt, negative_prompt, steps, guidance, width, height, seed, num_images
        )
        
        # Save images
        saved = generator.save_images(images, prompt)
        
        # Prepare output
        output_images = images if len(images) > 1 else images[0]
        
        info = f"✅ Generated {len(images)} image(s) in {gen_time:.2f}s\n"
        info += f"📝 Prompt: {prompt}\n"
        info += f"🎲 Seed: {used_seed}\n"
        info += f"💾 Saved to: {os.path.dirname(saved[0])}"
        
        return output_images, info
    
    # Interface
    with gr.Blocks(title="Stable Diffusion Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Stable Diffusion Image Generator")
        gr.Markdown("Generate images using your fine-tuned model")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your image...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What to exclude...",
                    lines=2
                )
                
                with gr.Row():
                    steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                    guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance Scale")
                
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 for random)")
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                
                generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                output_image = gr.Gallery(label="Generated Images", columns=2, height=600)
                info_box = gr.Textbox(label="Generation Info", lines=4)
        
        # Examples
        examples = [
            ["a beautiful landscape with mountains and lake, digital art, 4k"],
            ["portrait of a cyberpunk character, neon lights, detailed face"],
            ["cute anime character, pastel colors, studio ghibli style"],
            ["futuristic city at night, flying cars, cyberpunk aesthetic"],
        ]
        
        gr.Examples(examples=examples, inputs=[prompt])
        
        # Generate button
        generate_btn.click(
            fn=generate_ui,
            inputs=[prompt, negative_prompt, steps, guidance, width, height, seed, num_images],
            outputs=[output_image, info_box]
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    
    parser.add_argument("--model_path", type=str, default="./sd-finetuned",
                       help="Path to trained model")
    parser.add_argument("--prompt", type=str, default="",
                       help="Prompt for generation (command line mode)")
    parser.add_argument("--negative", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--steps", type=int, default=25,
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    parser.add_argument("--seed", type=int, default=-1,
                       help="Random seed")
    parser.add_argument("--num", type=int, default=1,
                       help="Number of images")
    parser.add_argument("--output", type=str, default="./generated",
                       help="Output directory")
    parser.add_argument("--ui", action="store_true",
                       help="Launch web UI")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = StableDiffusionGenerator(args.model_path)
    
    if args.ui:
        # Launch web interface
        print("🌐 Launching web interface...")
        print("   Open http://localhost:7860 in your browser")
        demo = create_interface(generator)
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
    else:
        # Command line mode
        if not args.prompt:
            print("❌ Please provide a prompt with --prompt or use --ui for web interface")
            return
        
        print(f"🎨 Generating: {args.prompt}")
        
        images, seed, gen_time = generator.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative,
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
            num_images=args.num
        )
        
        saved = generator.save_images(images, args.prompt, args.output)
        
        print(f"✅ Generated {len(images)} image(s) in {gen_time:.2f}s")
        print(f"🎲 Seed: {seed}")
        print(f"💾 Saved to:")
        for path in saved:
            print(f"   {path}")
        
        # Show first image
        if images:
            images[0].show()

if __name__ == "__main__":
    main()
