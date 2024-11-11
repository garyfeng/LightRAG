# Use an official Python 3.11 runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the LightRAG repository
RUN git clone https://github.com/HKUDS/LightRAG.git

# Set the working directory to the LightRAG directory
WORKDIR /app/LightRAG

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install LightRAG
RUN pip install --no-cache-dir -e .

# Additional libraries
RUN pip install python-dotenv

# Set the working directory back to /app
WORKDIR /app

# Copy your application code to the container
COPY . .

# Expose any ports the app is expected to run on
EXPOSE 8000

# Run the application
CMD ["python", "test.py"]
#CMD ["/bin/bash", "ls"]
