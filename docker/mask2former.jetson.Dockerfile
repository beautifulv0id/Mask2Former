ARG BASE_IMAGE=nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_692ceb1a0a35fe8a1f37641bab978508
FROM ${BASE_IMAGE} AS base
ARG CUDA_VERSION=12.2
ARG TORCH_INSTALL=https://pypi.jetson-ai-lab.dev/jp6/cu122/+f/caa/8de487371c7f6/torch-2.4.0-cp310-cp310-linux_aarch64.whl#sha256=caa8de487371c7f66b566025700635c30728032e0a3acf1c3183cec7c2787f94

RUN wget raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh 
RUN sudo CUDA_VERSION=${CUDA_VERSION} bash ./install_cusparselt.sh

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install "numpy<2"
RUN python3 -m pip install --no-cache ${TORCH_INSTALL}

RUN git clone https://github.com/facebookresearch/detectron2.git /app/detectron2
WORKDIR /app/detectron2
RUN pip install -e .
RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip install git+https://github.com/mcordts/cityscapesScripts.git

WORKDIR /app
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts
# RUN --mount=type=ssh \
#     git clone git@github.com:beautifulv0id/Mask2Former.git /app/Mask2Former
COPY . /app/Mask2Former

WORKDIR /app/Mask2Former
RUN pip install -r requirements.txt
WORKDIR /app/Mask2Former/mask2former/modeling/pixel_decoder/ops

ENV TORCH_CUDA_ARCH_LIST="8.7"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

RUN python setup.py install
