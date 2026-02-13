FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix the circular import issue
ENV PYTHONUNBUFFERED=1

# Run with single worker (no multiprocessing)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]