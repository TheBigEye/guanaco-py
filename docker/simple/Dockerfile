# Define the image argument and provide a default value
ARG IMAGE=python:3.12-slim-bookworm

# Use the image as specified
FROM ${IMAGE}

# Re-declare the ARG after FROM
ARG IMAGE

# Update and upgrade, install system deps for build
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Create app dir
RUN mkdir /app
WORKDIR /app

# Copy repo code
COPY . /app

# Upgrade pip and install Python deps (including server extras)
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pytest scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install the local package with server extras
RUN pip install .[server] --verbose --no-cache-dir

# Set environment for server
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the server script (asumiendo que existe run.sh)
CMD ["/bin/sh", "/app/docker/simple/run.sh"]
