FROM python:3.10-slim

LABEL maintainer="sml2h3"
LABEL description="DdddOcr - 通用验证码识别 API 服务"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# API 服务配置
ENV DDDDOCR_HOST=0.0.0.0
ENV DDDDOCR_PORT=8000
ENV DDDDOCR_WORKERS=1

# OCR 引擎配置
ENV DDDDOCR_OCR=false
ENV DDDDOCR_DET=false
ENV DDDDOCR_OLD=false
ENV DDDDOCR_BETA=false
ENV DDDDOCR_USE_GPU=false
ENV DDDDOCR_DEVICE_ID=0
ENV DDDDOCR_SHOW_AD=False

# 自定义模型配置
ENV DDDDOCR_IMPORT_ONNX_PATH=
ENV DDDDOCR_CHARSETS_PATH=

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${DDDDOCR_PORT}/health || exit 1

CMD python -m ddddocr api \
    --host=${DDDDOCR_HOST} \
    --port=${DDDDOCR_PORT} \
    --workers=${DDDDOCR_WORKERS} \
    --ocr=${DDDDOCR_OCR} \
    --det=${DDDDOCR_DET} \
    --old=${DDDDOCR_OLD} \
    --beta=${DDDDOCR_BETA} \
    --use-gpu=${DDDDOCR_USE_GPU} \
    --device-id=${DDDDOCR_DEVICE_ID} \
    --show-ad=${DDDDOCR_SHOW_AD} \
    --import-onnx-path=${DDDDOCR_IMPORT_ONNX_PATH} \
    --charsets-path=${DDDDOCR_CHARSETS_PATH}
