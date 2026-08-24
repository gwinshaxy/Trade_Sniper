FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies for psycopg2 and TA compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Grant execution permissions to entrypoint
RUN chmod +x entrypoint.sh

# Run as non-root user for security best practices
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["./entrypoint.sh"]