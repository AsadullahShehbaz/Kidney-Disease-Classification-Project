FROM python:3.12-slim

# Install git (required for pip to clone git repositories)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","main.py"] 
