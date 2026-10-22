ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
WORKDIR /app
COPY pyproject.toml README.en.md LICENSE /app/
COPY src/ /app/src/
COPY deployment/shadow/deploy/requirements.shadow.txt /opt/requirements.shadow.txt
# Install our source first; do not substitute a same-named package from an index.
RUN python -m pip install --no-cache-dir --no-deps /app && python -m pip install --no-cache-dir --only-binary=:all: -r /opt/requirements.shadow.txt
COPY deployment/shadow/shadow/ /opt/shadow/
COPY deployment/cpu/container_probe.py /opt/container_probe.py
USER 10001:10001
CMD ["python", "/opt/shadow/serve_shadow.py", "--store", "/models", "--socket", "/run/shadow/score.sock"]
