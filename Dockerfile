# Dockerfile for Obesity ML Project
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/interim data/processed models reports/figures reports/metrics mlruns

# Set Python path
ENV PYTHONPATH=/app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Make scripts executable
RUN chmod +x scripts/*.py 2>/dev/null || true

# Default command (can be overridden)
CMD ["python", "scripts/run_eda.py"]
