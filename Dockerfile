# syntax=docker/dockerfile:1.6

ARG PYTHON_BASE_IMAGE=python:3.10-slim@sha256:6a5861123aa815f92e5d20ce8372a8ba6668540c1081e5c4c44933cc1ba4fd3a

FROM ${PYTHON_BASE_IMAGE} AS builder
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
WORKDIR /app

COPY requirements.txt requirements-torch.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv "${VENV_PATH}" && \
    pip install --upgrade pip && \
    pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

FROM ${PYTHON_BASE_IMAGE} AS runtime
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
