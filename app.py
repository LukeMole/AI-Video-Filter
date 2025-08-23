from flask import Flask, render_template, jsonify
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os, shutil

import main

UPLOAD_FOLDER = f"{os.getcwd}/video"
ALLOWED_EXTENSIONS = {'mp4',"mov"}

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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
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
                    'frames':len(video_data['frames']), 'framerate': video_data['framerate'],
                    'base_resolution_x':video_data['resolution']['x'], 'base_resolution_y':video_data['resolution']['y'],
                    'final_resolution_x' : video_data['final_resolution']['x'], 'final_resolution_y': video_data['final_resolution']['y']})




@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=8000)