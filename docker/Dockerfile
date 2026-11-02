# Runtime image for the ONNX detector. Build context is the repo root:
#   docker build -f docker/Dockerfile -t aerial-vehicle-detection .
FROM python:3.11-slim

# ffmpeg writes the annotated mp4; libglib2.0-0 is what headless opencv links.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements.txt ./docker/requirements.txt
RUN pip install --no-cache-dir -r docker/requirements.txt

# Only what inference needs: the package, the entry point, the config and the
# exported graph. Training code and the checkpoint stay out of the image.
COPY avd ./avd
COPY eval/detect_onnx.py ./eval/detect_onnx.py
COPY configs/pipeline.yaml ./configs/pipeline.yaml
COPY weights/vehicle_detector.onnx ./weights/vehicle_detector.onnx

# Mount points for the clips to read and the results to write.
RUN mkdir -p videos/in videos/out
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "eval/detect_onnx.py"]
CMD ["--in", "videos/in", "--out", "videos/out"]
