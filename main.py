import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import random
import cv2
import moviepy
import os
from PIL import Image
import math


model = "stabilityai/stable-diffusion-xl-refiner-1.0"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("mps")


def generate_ai_image(seed, image, strength, guidance_scale, prompt):

    generator = torch.manual_seed(seed)

    image = pipe(prompt, image=image, strength=strength, guidance_scale=guidance_scale, generator=generator).images
    return image[0]

def get_video_data(video_name, video_scale):
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
    # Process each frame
    while ret:
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame,(math.floor(resolution['x']*video_scale),math.floor(resolution['y']*video_scale)))
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(resized_frame)

        # Append the PIL image to the frames list
        frames.append(pil_image)

        # Read the next frame
        ret, frame = video.read()

    return {'frames':frames, 'framerate':framerate, 'resolution':resolution, 'audio':audio}



if __name__ == '__main__':
    strength = 0.4  # Lower values make the output less like the input image 0-1
    guidance_scale = 8  # Higher values make the output more aligned with the text prompt 1-10
    video_scale = 0.7
    #init_image = load_image('test2.jpg').convert("RGB")
    prompt = "a hyper realistic scene"
    seed = random.randint(1, 2147483647)

    #image = generate_ai_image(seed, init_image, strength, guidance_scale, prompt)
    #image.save('ai_result.jpg')
    video_info = get_video_data('test2.mp4', video_scale)

    start_frame = 1
    end_frame = len(video_info['frames'])
    
    for I in range(start_frame-1, end_frame):
        frame = video_info['frames'][I]
        image = generate_ai_image(seed,frame,strength,guidance_scale,prompt)
        torch.mps.empty_cache()
        image.save('ai_result.jpg')
        del image
    #print(video_info['audio'])