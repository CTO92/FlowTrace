# Use the official Playwright image which includes Python and browser dependencies
FROM mcr.microsoft.com/playwright/python:v1.41.0-jammy

WORKDIR /app

# Optimization: Install Playwright and browsers first to cache the heavy binary download layer.
# This prevents re-downloading browsers every time requirements.txt changes.
RUN pip install --no-cache-dir playwright==1.41.0
RUN python -m playwright install chromium

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyvis pandas_datareader

# Copy the rest of the application code
COPY . .

# Default command (overridden by docker-compose)
CMD ["python", "ingestion_listener.py"]