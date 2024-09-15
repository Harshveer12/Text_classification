# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3.10 -m pip install --upgrade pip

# Install TensorFlow-Text and TensorFlow Models Official
RUN pip install \
    tensorflow-text==2.13.* \
    tf-models-official==2.13.*

# Set python3.10 as the default Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set a working directory
WORKDIR /app

# Copy your project files (if you have any to include in the container)
# COPY . /app

# Set the entry point (if needed, for example to run a script at container startup)
# ENTRYPOINT ["python3"]

# Expose ports (if needed)
# EXPOSE 8080

