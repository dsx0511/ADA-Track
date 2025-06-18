ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# need this for apt to not get stuck prompting the user for locale info
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV FORCE_CUDA="1"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# fix for non signed nvidia pub keys (taken from https://jdhao.github.io/2022/05/05/nvidia-apt-repo-public-key-error-fix/)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install some useful libs
RUN apt-get update && apt-get install -y wget ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 git nano sudo python3.8-dev build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

RUN pip install matplotlib \
                tqdm \
                imageio \
                numba \
                pandas \
                pyyaml \
                scipy \
                seaborn \
                tensorboard \
                numpy==1.21 \
                nuscenes-devkit \
                motmetrics==1.1.3 \
                einops \
                flash-attn==0.2.2 \
                yapf==0.40.1

RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN pip install mmdet==2.25.3 && pip install setuptools==59.5.0
RUN pip install mmsegmentation==0.30.0
RUN pip install torch_geometric==2.0.4
RUN pip install torch_scatter==2.0.9 torch_sparse==0.6.13 torch_cluster==1.6.0 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html

WORKDIR /workspace