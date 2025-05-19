import os
import re
import torch
import argparse
from diffusers import DiffusionPipeline
from huggingface_hub import login
from dotenv import load_dotenv

# --- CLI args ---
parser = argparse.ArgumentParser(description="Batch inference with FLUX + LoRA")

parser.add_argument("--p", "--prompt-file", type=str, default="prompts.txt", help="Path to prompt file")
parser.add_argument("--o", "--output-dir", type=str, default="./outputs", help="Output directory")
parser.add_argument("--s", "--seed", type=int, default=0, help="Random seed")
parser.add_argument("--steps", type=int, default=50, help="Inference steps")
parser.add_argument("--scale", type=float, default=3.5, help="Guidance scale")
parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev", help="Base model path or HF repo")
parser.add_argument("--lora-dir", type=str, default="./output/leander_dreamframe_v1", help="Local LoRA weights dir")
parser.add_argument("--lora-weight", type=str, default="leander_dreamframe_v1.safetensors", help="LoRA .safetensors filename")

args = parser.parse_args()

# --- Read values ---
prompt_file = args.p
output_dir = args.o
seed = args.s
inference_steps = args.steps
guidance_scale = args.scale
model_path = args.model
lora_dir = args.lora_dir
lora_weight = args.lora_weight

# --- Load .env and login ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ Hugging Face token not found in .env file.")
login(token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"

def sanitize_filename(prompt_text):
    return re.sub(r'[^a-zA-Z0-9]+', '_', prompt_text.strip())[:60].strip("_")

# --- Load model and LoRA ---
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)
pipe.load_lora_weights(lora_dir, weight_name=lora_weight)

# --- Read prompts ---
with open(prompt_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

os.makedirs(output_dir, exist_ok=True)
generator = torch.Generator(device="cpu").manual_seed(seed)

# --- Generate ---
for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] {prompt}")
    filename = f"{sanitize_filename(prompt)}.png"
    output_path = os.path.join(output_dir, filename)

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        max_sequence_length=512,
        generator=generator
    ).images[0]

    image.save(output_path)
    print(f"✅ Saved: {output_path}")