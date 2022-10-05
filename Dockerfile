ARG IMAGE_NAME=nvcr.io/nvidia/pytorch:22.09-py3
FROM ${IMAGE_NAME}

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"


RUN pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

WORKDIR /graphite
COPY . .
RUN pip install -r requirements.txt