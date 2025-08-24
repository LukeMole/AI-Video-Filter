from flask import Flask, render_template, jsonify, Response
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os, shutil
import math
import random
from datetime import datetime
import json
import time
import threading
import queue

import main

UPLOAD_FOLDER = f"{os.getcwd}/video"
ALLOWED_EXTENSIONS = {'mp4',"mov"}
seed = 0
pipe = ''
video_data = ''
progress_queue = queue.Queue()
generation_active = False


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video(filename):
    print(filename)
    clear_frame_cache_val = request.form.get('clearFrameCache')
    halve_fps = request.form.get('halveFps')
    if clear_frame_cache_val == 'on':
        main.clear_temp()

    half_fps = False
    if halve_fps == 'on':
        half_fps = True

    return main.get_video_data(filename, half_fps)

    



app = Flask(__name__)



@app.route('/', methods=['GET','POST'])
def index():
    global seed
    seed = random.randint(1, 2147483647)
    return render_template('index.html', seed=seed)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    global video_data
    if os.path.exists('uploads'):
        shutil.rmtree('uploads')
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    save_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(save_path)
    video_data = process_video(file.filename)
    return jsonify({'success': True, 'filename': file.filename, 
                    'frames':len(video_data['frames']), 'framerate': math.floor(video_data['framerate']),
                    'base_resolution_x':video_data['resolution']['x'], 'base_resolution_y':video_data['resolution']['y'],
                    'final_resolution_x' : video_data['final_resolution']['x'], 'final_resolution_y': video_data['final_resolution']['y']})

@app.route('/generate_seed', methods=['GET'])
def generate_seed():
    global seed
    seed = random.randint(1, 2147483647)
    return jsonify({'seed': seed})


@app.route('/progress_stream')
def progress_stream():
    def event_stream():
        while True:
            try:
                # Get progress data from queue with timeout
                data = progress_queue.get(timeout=1.0)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            except GeneratorExit:
                break
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                          'Connection': 'keep-alive',
                          'Access-Control-Allow-Origin': '*'})


def send_progress_update(current_frame, total_frames, time_remaining, frame_processing_time=None):
    """Send progress update to all connected clients"""
    frames_remaining = total_frames - current_frame
    progress_data = {
        'type': 'progress',
        'current_frame': current_frame,
        'total_frames': total_frames,
        'frames_remaining': frames_remaining,
        'time_remaining': time_remaining,
        'progress_percentage': round((current_frame / total_frames) * 100, 1)
    }
    
    if frame_processing_time:
        progress_data['frame_processing_time'] = frame_processing_time
    
    try:
        progress_queue.put_nowait(progress_data)
    except queue.Full:
        # Clear old messages if queue is full
        try:
            progress_queue.get_nowait()
            progress_queue.put_nowait(progress_data)
        except queue.Empty:
            pass


@app.route('/generate_frames', methods=['GET','POST'])
def generate_selected_frames():
    global seed, generation_active
    
    if generation_active:
        return jsonify({'error': 'Generation already in progress'}), 400
    
    turbo_val = request.form.get('turbo')
    prompt = request.form.get('prompt')
    seed_val = request.form.get('seed')
    start_frame = int(request.form.get('startFrame'))
    end_frame = int(request.form.get('endFrame'))

    turbo = False
    if turbo_val == 'on':
        turbo = True
    
    print(f"Turbo: {turbo}, Prompt: {prompt}, Seed: {seed_val}")
    print(f"Frames: {start_frame} to {end_frame}")
    
    # Start generation in background thread
    def generate_frames_background():
        global generation_active
        generation_active = True
        
        try:
            compute_device = 'GPU'
            pipe = main.initialise_ai(compute_device=compute_device, turbo=turbo)
            upscaler_dict = main.initialise_upscaler(compute_device=compute_device)
        except:
            compute_device = 'CPU'
            pipe = main.initialise_ai(compute_device=compute_device, turbo=turbo)
            upscaler_dict = main.initialise_upscaler(compute_device=compute_device)

        total_frames = end_frame - start_frame + 1
        
        # Send initial progress
        send_progress_update(0, total_frames, "Calculating...", None)
        
        frame_times = []  # Track processing times for better estimation
        
        for i, frame_idx in enumerate(range(start_frame-1, end_frame)):
            start_time = datetime.now()
            
            # Send progress before processing frame
            frames_processed = i
            if i == 0:
                send_progress_update(frames_processed, total_frames, "Starting first frame...", None)
            else:
                # Calculate estimated time based on average of previous frames
                avg_time = sum(frame_times) / len(frame_times)
                frames_remaining = total_frames - i
                time_left_seconds = math.ceil(avg_time * frames_remaining)
                
                hours = math.floor(time_left_seconds / 3600)
                minutes = math.floor((time_left_seconds % 3600) / 60)
                seconds = time_left_seconds % 60
                
                estimated_time = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                send_progress_update(frames_processed, total_frames, estimated_time, None)
            
            main.generate_frame(pipe, upscaler_dict, video_data['frames'], 
                              seed_val, prompt, frame_idx, upscale=True, 
                              compute_device=compute_device)

            end_time = datetime.now()
            frame_processing_time = (end_time - start_time).total_seconds()
            frame_times.append(frame_processing_time)
            
            # Keep only last 5 frame times for more accurate recent estimation
            if len(frame_times) > 5:
                frame_times = frame_times[-5:]
            
            # Calculate time remaining based on average processing time
            frames_remaining = total_frames - (i + 1)
            if frames_remaining > 0:
                avg_time = sum(frame_times) / len(frame_times)
                time_left_seconds = math.ceil(avg_time * frames_remaining)
                
                hours = math.floor(time_left_seconds / 3600)
                minutes = math.floor((time_left_seconds % 3600) / 60)
                seconds = time_left_seconds % 60
                
                time_remaining = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
            else:
                time_remaining = "00:00:00"
            
            # Send progress update after processing frame
            send_progress_update(i + 1, total_frames, time_remaining, 
                               f"{frame_processing_time:.1f}s")
            
            print(f'Frame {frame_idx + 1} processed. Time Remaining -> {time_remaining}')
        
        # Send completion message
        completion_data = {
            'type': 'complete',
            'message': 'All frames generated successfully!'
        }
        progress_queue.put_nowait(completion_data)
        
        generation_active = False
    
    # Start background thread
    thread = threading.Thread(target=generate_frames_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Frame generation started'})


@app.route('/generation_status')
def generation_status():
    global generation_active
    return jsonify({'active': generation_active})


if __name__ == "__main__":
    app.run(debug=True, port=8000, use_reloader=False)