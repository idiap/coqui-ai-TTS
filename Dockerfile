ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}

RUN apt-get update && \
  apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    gcc g++ make python3 python3-dev python3-pip \
    python3-venv python3-wheel espeak-ng \
    libsndfile1-dev libc-dev curl && \
  rm -rf /var/lib/apt/lists/*

# Install Rust compiler (to build sudachipy for Mac)
RUN curl --proto '=https' --tlsv1.2 -sSf "https://sh.rustup.rs" | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip3 install -U pip setuptools wheel
RUN pip3 install llvmlite --ignore-installed

# Install Dependencies:
RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /root/.cache/pip

# Copy TTS repository contents:
WORKDIR /root
COPY . /root

RUN pip3 install -e .[all]

ENTRYPOINT ["tts"]
CMD ["--help"]
