FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh


ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y wget 

# instalando python
RUN apt-get install -y python3  && apt-get install -y python3-pip

RUN apt-get update && apt-get install -y git

WORKDIR /app

RUN pip install accelerate
RUN pip install torch torchvision torchaudio transformers
RUN pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
#RUN pip install git+https://github.com/m-bain/whisperx.git --upgrade

RUN pip install pyannote.audio

# Fix numpy version to avoid np.NaN issue
RUN pip install numpy==1.24.3

ENV NVIDIA_VISIBLE_DEVICES all


CMD ["python3", "app.py"]

