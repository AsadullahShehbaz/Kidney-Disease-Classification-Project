FROM python:3.12-slim-bookworm

WORKDIR /app

# Install git only (minimum requirements)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use gunicorn to avoid multiprocessing issues
CMD ["gunicorn", "app:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]