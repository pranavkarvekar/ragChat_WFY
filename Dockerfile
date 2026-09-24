FROM python:3.12-slim

# Create a non-root user (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set environment variables
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies if any are needed (e.g., for OpenCV or other ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user . .

# Switch to the non-root user
USER user

# Collect static files for Django
RUN python manage.py collectstatic --noinput

# Expose the port Hugging Face uses
EXPOSE 7860

# Start Gunicorn on port 7860
CMD ["gunicorn", "ragWFY.wsgi:application", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120"]
