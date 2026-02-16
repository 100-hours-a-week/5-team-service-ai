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

CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
