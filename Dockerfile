FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y \
    ffmpeg \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    wget \
    git \
    build-essential \
    libsndfile1 \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/m-bain/whisperX.git

# Copy source code
COPY . .

# The handler will be started by RunPod
CMD ["python", "-u", "handler.py"]
