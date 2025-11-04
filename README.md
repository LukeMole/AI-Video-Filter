# AI Video Filter

A powerful web-based application that applies AI-generated artistic filters to videos locally using state-of-the-art machine learning models. Transform your videos with AI effects like anime style, watercolor painting, and custom artistic transformations.

## 🎬 Features

- **AI-Powered Video Transformation**: Convert videos using advanced diffusion models (InstructPix2Pix)
- **Real-time Progress Tracking**: Live progress updates with time remaining estimates
- **High-Quality Upscaling**: 4x resolution enhancement using Swin2SR
- **Audio Preservation**: Maintain original audio in processed videos
- **Web Interface**: User-friendly browser-based interface
- **GPU/CPU Support**: Automatic GPU detection with CPU fallback
- **Background Processing**: Non-blocking video processing with stop functionality

## 🎯 System Requirements

### Minimum Requirements
- **RAM**: 8GB system RAM
- **GPU VRAM**: 6GB+ VRAM (NVIDIA/AMD) or 8GB+ unified memory (Apple Silicon)
- **Storage**: 10GB free space for models and temporary files
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Recommended Requirements
- **RAM**: 16GB+ system RAM
- **GPU VRAM**: 8GB+ VRAM or 16GB+ unified memory (Apple Silicon)
- **Storage**: 20GB+ free space
- **GPU**: NVIDIA RTX 3070/4060 or better, AMD RX 6700 XT or better, Apple M1 or newer

### Performance Notes
- **AI Models Memory Usage**: ~5GB RAM/VRAM during processing

## 🚀 Installation

### Prerequisites

1. **Python 3.8+**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **FFmpeg** (Required for video processing)
   
   **Windows:**
   - Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Add to system PATH
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

3. **Git** (for cloning)
   ```bash
   git --version
   ```

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/LukeMole/AI-Video-Filter.git
cd AI-Video-Filter
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# macOS/Linux
python -m venv myenv
source myenv/bin/activate
```

#### 3. Install PyTorch
**Choose based on your system:**

**NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**AMD GPU (ROCm) - Linux:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

**Apple Silicon (M1 Or Newer):**
```bash
pip install torch torchvision torchaudio
```

**CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Install Dependencies
```bash
pip install diffusers
pip install huggingface_hub
pip install transformers
pip install accelerate
pip install sentencepiece
pip install opencv-python
pip install moviepy
pip install protobuf
pip install flask
pip install pillow
pip install numpy

# Install latest diffusers from GitHub (recommended)
pip install git+https://github.com/huggingface/diffusers.git
```

#### 5. Hugging Face Authentication

1. **Create Hugging Face Account**: [https://huggingface.co/join](https://huggingface.co/join)

2. **Generate Access Token**:
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create new token with **"Read"** access
   - Enable: "Read access to contents of all public gated repos you can access"

3. **Login via CLI**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # Enter your token when prompted
   ```

#### 6. Accept Model Licenses
Visit these pages and accept the licenses:
- [Stable Diffusion XL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 
- [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix)

#### 7. If You Are Using Cuda
Install the cuda toolkit:
- [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)

## 🎮 Usage

### Starting the Application

1. **Activate Virtual Environment**:
   ```bash
   # Windows
   myenv\Scripts\activate
   
   # macOS/Linux  
   source myenv/bin/activate
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:8000`

### Using the Web Interface

#### 1. **Upload Video**
- Click "Choose File" and select your video
- Supported formats: MP4, MOV
- Check "Clear Frame Cache" to reset previous processing
- Check "Halve FPS" to reduce processing time (recommended for long videos)

#### 2. **Configure Processing**
- **Prompt**: Describe the transformation (e.g., "make it look like an anime")
- **Seed**: Random number for reproducible results
- **Frame Range**: Select start and end frames to process
- **Render Mode**: 
  - Unchecked = GPU processing (faster)
  - Checked = CPU processing (slower, more compatible)

#### 3. **Generate Frames**
- Click "Generate Frames" to start processing
- Monitor real-time progress with time estimates
- Use "Stop Generation" to cancel if needed

#### 4. **Compile Video**
- Check "Include Source Audio" to preserve original audio
- Click "Compile Frames" to create final video
- Video will automatically download when ready

### Example Prompts

- `"make it look like a Studio Ghibli anime"`
- `"transform into a watercolor painting"`
- `"convert to oil painting style"`
- `"make it look like a pencil sketch"`
- `"add cyberpunk neon aesthetic"`
- `"turn into Van Gogh painting style"`

## 📁 Project Structure

```
AI-Video-Filter/
├── app.py                 # Flask web application
├── main.py               # Core AI processing functions
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Web interface template
├── static/
│   ├── css/
│   │   └── style.css    # Styling
│   └── temp/            # Temporary processed frames
├── uploads/             # Uploaded video files
├── final_videos/        # Compiled output videos
└── myenv/              # Python virtual environment
```

## ⚙️ Configuration Options

### Performance Tuning

**For High-End Systems (16GB+ VRAM):**
- Use GPU mode
- Process full resolution
- Enable upscaling for maximum quality

**For Mid-Range Systems (8-12GB VRAM):**
- Use GPU mode
- Enable "Halve FPS" for longer videos
- Process shorter frame ranges

**For Low-End Systems (4-6GB VRAM):**
- Use CPU mode
- Enable "Halve FPS"
- Process 10-30 frames at a time
- Close other applications

### Memory Management
The application automatically:
- Clears GPU cache between frames
- Uses garbage collection
- Provides stop functionality to free resources

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Use CPU mode or process fewer frames at once
```

**2. FFmpeg Not Found**
```
Error: ffmpeg not found
Solution: Install FFmpeg and add to PATH
```

**3. Hugging Face Authentication**
```
Error: Repository not found or access denied
Solution: Login with huggingface-cli and accept model licenses
```

**4. PyTorch Installation Issues**
```
Solution: Reinstall PyTorch with correct CUDA version
Visit: https://pytorch.org/get-started/locally/
```

**5. No Audio in Output Video**
```
Solution: Ensure "Include Source Audio" is checked and FFmpeg is properly installed
```

### Performance Optimization

**Speed up processing:**
- Use smaller frame ranges
- Enable "Halve FPS"
- Use GPU mode
- Close unnecessary applications

**Improve quality:**
- Use full resolution
- Process all frames
- Use detailed prompts
- Experiment with different seeds

## 📚 Technical Details

### AI Models Used
- **InstructPix2Pix**: Main transformation model (~2GB)
- **Swin2SR**: 4x upscaling model (~40MB)
- **CLIP**: Text encoding for prompts

### Supported Hardware
- **NVIDIA GPUs**: RTX 20 series or newer with CUDA
- **AMD GPUs**: RX 6000/7000 series with ROCm (Linux)
- **Apple Silicon**: M1 or newer with MPS acceleration
- **Intel/AMD CPUs**: Fallback support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stability AI** for Stable Diffusion models
- **Hugging Face** for model hosting and transformers
- **timbrooks** for InstructPix2Pix
- **Microsoft** for Swin2SR upscaling model

## 🐛 Support

For issues and questions:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Provide system specifications and error logs

---

**Note**: First run will download ~5GB of AI models. Ensure stable internet connection and sufficient storage space.
