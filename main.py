import random
import os
import shutil
import math
import sys
import numpy as np

from PIL import Image
import PIL
import cv2
import moviepy
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution



def initialise_ai(compute_device):
    OS = sys.platform

    model = "stabilityai/stable-diffusion-xl-refiner-1.0"
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    if compute_device == 'GPU':
        if OS == 'darwin':
            pipe = pipe.to("mps")
        else:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to('cpu')

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


def initialise_upscaler(compute_device):
    OS = sys.platform
    upscaler_processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    upscaler = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")

    if compute_device == 'GPU':
        if OS == 'darwin':
            upscaler = upscaler.to("mps")
        else:
            upscaler = upscaler.to("cuda")
    else:
        upscaler = upscaler.to('cpu')

    return {'upscaler':upscaler, 'processor':upscaler_processor}

def upscale_image(image, upscaler, upscaler_processor, compute_device):
    OS = sys.platform

    inputs = upscaler_processor(image, return_tensors="pt")
    if compute_device == 'GPU':
        if OS == 'darwin':
            inputs = {k: v.to('mps') for k, v in inputs.items()}
        else:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = upscaler(**inputs)
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8) 

    image = Image.fromarray(output)
    return image

def generate_ai_image(pipe, seed, image, prompt):
    generator = torch.manual_seed(seed)
    image = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1, seed=generator).images

    return image[0]

def get_video_data(video_name):
    #get video
    BUFFER = 0.5
    all_frames = []
    video = cv2.VideoCapture(video_name)
    resolution = {'x':video.get(cv2.CAP_PROP_FRAME_WIDTH), 'y':video.get(cv2.CAP_PROP_FRAME_HEIGHT)}
    framerate = video.get(cv2.CAP_PROP_FPS)
    frames = []
    ret, frame = video.read()
    
    if resolution['x'] > resolution['y']:
        ratio = (512 + BUFFER)/resolution['x']
    else:
        ratio = (512 + BUFFER)/resolution['y']

    #get audio
    cur_dir = os.getcwd()
    audio = moviepy.VideoFileClip(f'{cur_dir}/{video_name}').audio
    #goes through every frame and changes its color, resizes it and appends it to the frames list
    # Process each frame
    while ret:
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame,(math.floor(resolution['x']*ratio),math.floor(resolution['y']*ratio)))
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(resized_frame)
        pil_image = PIL.ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert('RGB')

        # Append the PIL image to the frames list
        frames.append(pil_image)

        # Read the next frame
        ret, frame = video.read()

    return {'frames':frames, 'framerate':framerate, 'resolution':resolution, 'audio':audio}

def generate_frames(pipe, upscaler_dict,base_frames, seed, prompt, start_frame, end_frame, upscale, compute_device):
    cur_dir = os.getcwd()
    temp_path = f'{cur_dir}/temp'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    for I in range(start_frame-1, end_frame):
        frame = base_frames[I]
        image = generate_ai_image(pipe,seed,frame,prompt)
        if upscale:
            upscaled_image = upscale_image(image, upscaler_dict['upscaler'], upscaler_dict['processor'], compute_device=compute_device)
            upscaled_image.save(f'{cur_dir}/temp/{I+1}.jpg')
            del upscaled_image
        else:
            image.save(f'{cur_dir}/temp/{I+1}.jpg')
        try:
            torch.mps.empty_cache()
        except:
            pass
        try:
            torch.cuda.empty_cache()
        except:
            pass
        del image

def generate_video(framerate, audio, video_name):
    cur_dir = os.getcwd()
    final_path = f'{cur_dir}/final_videos'
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    frames = os.listdir(f'{cur_dir}/temp')
    frame_numbers = []
    for frame in frames:
        if '.jpg' in frame:
            frame_numbers.append(int(frame.split('.')[0]))
    frame_numbers.sort()

    sorted_frames = []
    for number in frame_numbers:
        image = moviepy.ImageClip(f'{cur_dir}/temp/{str(number)}.jpg', duration=1/framerate)
        print(image.duration)
        sorted_frames.append(image)

    final_video = moviepy.concatenate_videoclips(sorted_frames, method='compose')
    final_video.audio = audio
    final_video.write_videofile(f'{video_name}.mp4', fps=framerate)
    print(frame_numbers)
    print(frames)


def clear_temp():
    cur_dir = os.getcwd()
    temp_path = f'{cur_dir}/temp'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        os.makedirs(temp_path)


if __name__ == '__main__':
    compute_device = 'GPU'  # Change to 'CPU' if you want to run on CPU
    pipe = initialise_ai(compute_device)
    upscaler_dict = initialise_upscaler(compute_device)

    strength = 0.2  # Lower values make the output less like the input image 0-1
    guidance_scale = 8  # Higher values make the output more aligned with the text prompt 1-10
    video_scale = 0.25
    prompt = "how would the image looked if it took place in ancient rome?"
    seed = random.randint(1, 2147483647)

    video_info = get_video_data('siege.mp4')
    print(len(video_info['frames']))
    start_frame = 1
    end_frame = len(video_info['frames'])
    clear_temp()
    
    generate_frames(pipe, upscaler_dict,video_info['frames'], seed, prompt, start_frame, end_frame, upscale=True, compute_device=compute_device)

    generate_video(video_info['framerate'], video_info['audio'],'siege_output2')

    #print(video_info['audio'])