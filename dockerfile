
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies for the script (requests, Pillow, transformers, torch, einops, timm)
RUN pip install --no-cache-dir \
    requests \
    Pillow \
    torch \
    transformers \
    einops \
    timm \
    flask

# Copy the Python script into the container at /app
COPY app.py /app/

# Set environment variables to not generate bytecode and to enable debugging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Expose the port Flask runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
