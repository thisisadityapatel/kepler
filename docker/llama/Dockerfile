FROM ubuntu:22.04

# MacOS (Automatically detect metal GPU apple silicon or CPU adoption).

# Set the llama.cpp version (optional)
ARG VERSION=master
ARG GITHUB_REPO_LLAMA=https://github.com/ggerganov/llama.cpp.git

# Install build dependencies (cached layer)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ccache \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp (only re-runs if VERSION changes)
WORKDIR /app
RUN git clone ${GITHUB_REPO_LLAMA}

# Checkout specific version
WORKDIR /app/llama.cpp
RUN git checkout ${VERSION}

# Configure ccache
ENV CCACHE_DIR=/cache/ccache
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Build llama.cpp server
RUN cmake -B build
RUN cmake --build build --config Release

# Verify llama-server was built
RUN test -f ./build/bin/llama-server || \
    (echo "ERROR: llama-server not found after build" && exit 1)

# Set working directory
WORKDIR /app/llama.cpp

# Default command
CMD ["/bin/bash"]