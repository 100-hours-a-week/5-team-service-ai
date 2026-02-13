# syntax=docker/dockerfile:1.6

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ARG INSTALL_TORCH=false
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
WORKDIR /app

COPY requirements.txt requirements-torch.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv "${VENV_PATH}" && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "${INSTALL_TORCH}" = "true" ]; then \
      pip install --no-cache-dir -r requirements-torch.txt; \
    fi

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV PORT=8000
WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /opt/venv /opt/venv
COPY app /app/app
COPY requirements.txt requirements-torch.txt ./

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health').read()" || exit 1

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]

############################################
# Serverless Target (LLM)
############################################
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS serverless

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 서버리스 전용 requirements
COPY requirements-serverless.txt .
RUN pip3 install --no-cache-dir -r requirements-serverless.txt

# handler 복사
COPY handler.py .

# Network Volume 캐시 경로
ENV HF_HOME=/runpod-volume/hf
ENV TRANSFORMERS_CACHE=/runpod-volume/hf/transformers
ENV HF_HUB_CACHE=/runpod-volume/hf/hub

CMD ["python3", "-u", "handler.py"]
