# Use the official Playwright image which includes Python and browser dependencies
FROM mcr.microsoft.com/playwright/python:v1.41.0-jammy

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium is sufficient for our agents)
RUN python -m playwright install chromium

# Copy the rest of the application code
COPY . .

# Default command (overridden by docker-compose)
CMD ["python", "ingestion_listener.py"]