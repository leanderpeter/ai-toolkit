import os
import re
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("❌ Hugging Face token not found in .env file.")

login(token=hf_token)

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "close-up portrait of a confident man with cinematic lighting, sharp details, moody background"
output_dir = "./outputs"
guidance_scale = 3.5
inference_steps = 50
height = 1024
width = 1024
seed = 0

# --- Generate filename from prompt ---
def sanitize_filename(prompt_text):
    return re.sub(r'[^a-zA-Z0-9]+', '_', prompt_text.strip())[:60].strip("_")

filename = f"{sanitize_filename(prompt)}.png"
output_path = os.path.join(output_dir, filename)

# --- Load pipeline ---
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to(device)

pipe.load_lora_weights(
    "./output/leander_dreamframe_v1",  # local path
    weight_name="leander_dreamframe_v1.safetensors"
)

# --- Generate image ---
generator = torch.Generator(device="cpu").manual_seed(seed)
image = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=inference_steps,
    max_sequence_length=512,
    generator=generator
).images[0]

# --- Save result ---
os.makedirs(output_dir, exist_ok=True)
image.save(output_path)
print(f"✅ Image saved at: {os.path.abspath(output_path)}")