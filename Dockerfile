FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application modules
COPY config.py .
COPY main.py .
COPY utils/ ./utils/
COPY agents/ ./agents/
COPY telegram_bot/ ./telegram_bot/

# Data folders (mounted as volumes at runtime)
RUN mkdir -p /app/data /app/chroma_db /app/knowledge /app/reports
RUN chmod -R 777 /app

CMD ["python", "main.py"]
