FROM python:3-slim-bookworm

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

WORKDIR /app
COPY . .

# Install the package
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install guanaco-py from this checkout (it is not on PyPI)
RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install . --verbose

# Run the server
CMD python3 -m llama_cpp.server

