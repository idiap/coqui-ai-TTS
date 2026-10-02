(docker_images)=
# Docker images
We provide docker images to be able to test TTS without having to setup your own environment.

## Using premade images
You can use premade images built automatically from the latest TTS version.

### CPU version
```bash
docker pull ghcr.io/idiap/coqui-tts-cpu
```
### GPU version
```bash
docker pull ghcr.io/idiap/coqui-tts
```

## Building your own image
```bash
docker build -t tts .
```

## Basic inference
Basic usage: generating an audio file from a text passed as argument.
You can pass any tts argument after the image name.

### CPU version
```bash
docker run --rm -v ~/tts-output:/root/tts-output ghcr.io/idiap/coqui-tts-cpu --text "Hello." --out_path /root/tts-output/hello.wav
```
### GPU version
For the GPU version, you need to have the latest NVIDIA drivers installed.
With `nvidia-smi` you can check the supported CUDA version.

```bash
docker run --rm --gpus all -v ~/tts-output:/root/tts-output ghcr.io/idiap/coqui-tts --text "Hello." --out_path /root/tts-output/hello.wav --use_cuda
```

## Start a server

To launch a TTS server, start the container and get a shell inside it. You can
find more details about the server and supported parameters [here](server.md).
Note that it is not optimized for performance.

### CPU version
```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/idiap/coqui-tts-cpu
tts-server --list_models #To get the list of available models
tts-server --model_name tts_models/en/vctk/vits
```

### GPU version
```bash
docker run --rm -it -p 5002:5002 --gpus all --entrypoint /bin/bash ghcr.io/idiap/coqui-tts
tts-server --list_models #To get the list of available models
tts-server --model_name tts_models/en/vctk/vits --use_cuda
```

You can then find a web interface at: http://localhost:5002

## Docker Compose

Alternatively to `docker run`, you can use [Docker
Compose](https://docs.docker.com/compose/) with the following configuration in a
`compose.yaml` file:

```yaml
services:
  coqui:
    image: ghcr.io/idiap/coqui-tts-cpu
    container_name: coqui
    ports:
      - "5002:5002"
    entrypoint: /bin/bash
    command: -c "tts-server --model_name tts_models/multilingual/multi-dataset/xtts_v2"
```

To persistently store models onto your local hard drive, you can add the
following lines, adjusting the left-hand side as desired:

```yaml
    volumes:
      - C:\Users\<user>\AppData\Local\tts:/root/.local/share/tts
```

Then start the container with:

```bash
docker-compose up
```
