import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import random

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("mps")
url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

seed = random.randint(1, 2147483647)
generator = torch.manual_seed(seed)
print(seed)

init_image = load_image('test2.jpg').convert("RGB")
prompt = "a hyper realistic scene"
# Adjust strength and guidance scale for more creative outputs
strength = 0.4  # Lower values make the output less like the input image
guidance_scale = 8  # Higher values make the output more aligned with the text prompt

image = pipe(prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, generator=generator).images
image[0].save('ai_result.jpg')