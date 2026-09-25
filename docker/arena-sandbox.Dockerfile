# Sandbox image for model-written paint programs (PAINT_SANDBOX=docker).
# Build: paint arena-sandbox --build   (or: docker build -t paint-arena-sandbox -f docker/arena-sandbox.Dockerfile docker)
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy scipy pillow
USER 65534:65534
WORKDIR /tmp
