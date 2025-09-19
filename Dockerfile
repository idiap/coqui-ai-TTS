ARG BASE=nvidia/cuda:11.8.0-base-ubuntu22.04
FROM ${BASE}


RUN apt-get update && \
  apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    gcc g++ make python3 python3-dev python3-pip \
    ffmpeg git \
    python3-venv python3-wheel espeak-ng \
    libsndfile1-dev libc-dev curl && \
  rm -rf /var/lib/apt/lists/*

# Install Rust compiler (to build sudachipy for Mac)
RUN curl --proto '=https' --tlsv1.2 -sSf "https://sh.rustup.rs" | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
COPY req_pytorch.txt .
RUN pip3 install -r req_pytorch.txt
RUN pip3 install -U pip setuptools wheel
RUN pip3 install llvmlite --ignore-installed

# Install Dependencies:
#RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
#RUN pip3 install torch torchaudio
RUN pip3 install git+https://github.com/openai/whisper.git openai
RUN pip3 install fastapi uvicorn dotenv python-multipart
RUN rm -rf /root/.cache/pip
RUN python3 -c "import whisper; whisper.load_model('turbo')"

# Copy TTS repository contents:
WORKDIR /root
COPY . /root

RUN pip3 install -e .[all]

RUN python3 -c "import os; os.environ['COQUI_TOS_AGREED']='1'; from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
