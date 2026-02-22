FROM python:3.10-slim

# libglib2.0-0 and libgomp1 are required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install CPU-only PyTorch first to avoid pulling the large CUDA wheel from PyPI,
# then install the rest of the dependencies. pip sees torch as already satisfied
# when processing requirements.txt and skips it.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace
