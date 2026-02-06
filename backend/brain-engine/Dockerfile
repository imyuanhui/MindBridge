FROM python:3.11-slim

# Install system dependencies for C-based math libraries
# RUN apt-get update && apt-get install -y --no-install-config-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your CSV dataset and the engine script
COPY brain_features_cleaned.csv .
COPY main.py .

# The engine is a worker script, not a web server
CMD ["python", "main.py"]