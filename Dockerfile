FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install python dependencies
# Note: whisperx might need to be installed from git if pip version is old, 
# but usually 'pip install whisperx' works.
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# The handler will be started by RunPod
CMD ["python", "-u", "handler.py"]
