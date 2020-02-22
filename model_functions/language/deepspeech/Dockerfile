FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

WORKDIR /workspace/

# install basics
RUN apt-get update -y
RUN apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop sox libsox-dev libsox-fmt-all vim

# install python deps
RUN pip install cython visdom cffi tensorboardX wget




# install apex
RUN git clone --recursive https://github.com/NVIDIA/apex.git
RUN cd apex; pip install .

# install deepspeech.pytorch
RUN git clone https://github.com/SeanNaren/deepspeech.pytorch.git
RUN cd deepspeech.pytorch; pip install -r requirements.txt

# launch jupiter
RUN pip install jupyter
RUN mkdir data; mkdir notebooks;

RUN pip install flask matplotlib
ADD librispeech_pretrained_v2.pth /workspace/models/deepspeech_final.pth
ADD server_deep.py /workspace/deepspeech.pytorch/server_deep.py