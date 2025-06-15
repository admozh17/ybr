FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p cache instance
EXPOSE 8080
CMD gunicorn web_app:app --bind 0.0.0.0:$PORT --timeout 300
