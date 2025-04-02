import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import random
import cv2
import moviepy
import os

def generate_ai_image(seed, image, strength, guidance_scale, prompt):
    model = "stabilityai/stable-diffusion-xl-refiner-1.0"
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe = pipe.to("mps")

    generator = torch.manual_seed(seed)

    image = pipe(prompt, image=image, strength=strength, guidance_scale=guidance_scale, generator=generator).images
    return image[0]

def get_video_data(video_name):
    #get video
    all_frames = []
    video = cv2.VideoCapture(video_name)
    resolution = {'x':video.get(cv2.CAP_PROP_FRAME_WIDTH), 'y':video.get(cv2.CAP_PROP_FRAME_HEIGHT)}
    framerate = video.get(cv2.CAP_PROP_FPS)
    frames = []
    ret, frame = video.read()

    #get audio
    cur_dir = os.getcwd()
    audio = moviepy.VideoFileClip(f'{cur_dir}/{video_name}').audio
    #goes through every frame and changes its color, resizes it and appends it to the frames list
    while ret:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    return {'frames':frames, 'framerate':framerate, 'resolution':resolution, 'audio':audio}



if __name__ == '__main__':
    strength = 0.4  # Lower values make the output less like the input image 0-1
    guidance_scale = 8  # Higher values make the output more aligned with the text prompt 1-10
    #init_image = load_image('test2.jpg').convert("RGB")
    #prompt = "a hyper realistic scene"
    #seed = random.randint(1, 2147483647)

    #image = generate_ai_image(seed, init_image, strength, guidance_scale, prompt)
    #image.save('ai_result.jpg')
    video_info = get_video_data('test2.mp4')
    print(video_info['audio'])