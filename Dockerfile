FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_USE_LEGACY_KERAS=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN mkdir -p /tmp/ultralytics /tmp/matplotlib /root/.deepface/weights \
    && chmod -R 777 /tmp/ultralytics /tmp/matplotlib

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir .
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='morsetechlab/yolov11-license-plate-detection', filename='license-plate-finetune-v1n.pt')" \
    && python -c "from retinaface import RetinaFace; RetinaFace.build_model()"

CMD ["python", "src/main.py"]
