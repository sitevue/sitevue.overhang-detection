# Overhang detection service for an x86-64 machine with an NVIDIA GPU (e.g. RTX 5070).
# Needs the NVIDIA driver + NVIDIA Container Toolkit on the host, with a driver new enough for CUDA 13.
#   docker build -t sitevue/overhang .
#   docker run --gpus all -p 8000:8000 sitevue/overhang serve rtsp://CAMERA/stream --cors-origin http://UI-HOST
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# torch for CUDA 13 (RTX 50-series / Blackwell); versions as validated on the GB10
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir ".[sam,service]"

# bake the SAM 2 weights into the image (no internet needed on site)
RUN python -c "from transformers import Sam2Model, Sam2Processor as P; m='facebook/sam2.1-hiera-small'; P.from_pretrained(m); Sam2Model.from_pretrained(m)"

ENV OVERHANG_CONFIG=/app/configs/fpl-grind.yaml
EXPOSE 8000
ENTRYPOINT ["overhang"]
CMD ["--help"]
