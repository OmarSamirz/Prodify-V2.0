# CUDA 11.8 + Ubuntu 22.04 base
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Set workdir
WORKDIR /src

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu118

# Copy source
COPY src/ /src/

# Run your script (change main.py if needed)
CMD ["python3", "label_data.py"]
