FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg wget build-essential

# Install Python packages
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn "git+https://github.com/facebookresearch/sam3.git"
