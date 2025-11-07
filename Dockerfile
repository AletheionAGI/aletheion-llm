# Multi-stage Dockerfile for Aletheion LLM

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Development image
FROM base as development

# Install development dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install development tools
RUN pip install \
    pytest \
    pytest-cov \
    black \
    ruff \
    isort \
    mypy \
    pre-commit \
    ipython \
    jupyter

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e ".[dev,docs]"

# Set default command
CMD ["bash"]

# Stage 3: Production image
FROM base as production

# Copy only necessary files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install production dependencies
RUN pip install --upgrade pip && \
    pip install .

# Create non-root user
RUN useradd -m -u 1000 aletheion && \
    chown -R aletheion:aletheion /app

USER aletheion

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import aletheion_llm" || exit 1

# Default command
CMD ["python", "-m", "aletheion_llm"]

# Stage 4: Training image with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as training

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source and install
COPY . .
RUN pip install -e ".[dev]"

# Set default command
CMD ["python", "examples/train.py"]
