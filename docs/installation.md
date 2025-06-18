# Installation

## Repository

First, clone this repository and the git submodules: 

```
git clone --recurse-submodules https://github.com/dsx0511/ADA-Track
```

## Environment Information

This repository was tested with:
- Python == 3.7 and 3.8
- CUDA == 11.3.1
- CUDNN == 8
- Pytorch == 1.10.0

It should work with other versions as well.

## Docker

We recommand using Docker to setup the environment.
We provide an example of Dockerfile [here](../docker/Dockerfile).
Fist, build the docker image:

```
cd docker
docker build -t ada_track:v1.0 .
```

After the docker image is built, you can run the docker container using this [script](../docker/run.sh)
```
bash run.sh $nusc_data_dir $repo_dir $exp_dir
```
while setting the arguments:
- `$nusc_data_dir`: path to the root of the nuscenes dataset,
- `$repo_dir`: path to this repository,
- `$exp_dir`: path to your workspace.

Inside the interactive session of the docker container, run thi [script](../docker/setup_in_container.sh) to install mmdetection3d and this repository as plugin:
```
cd docker && bash setup_in_container.sh
```

## Custom Installtion
If you prefer an installation using e.g. pip or conda, these dependencies are most critical:
- numpy>=1.21.0
- mmcv-full==1.4.0
- mmdet==2.25.3
- mmsegmentation==0.30.0
- torch_geometric==2.0.4, torch_scatter==2.0.9, torch_sparse==0.6.13 , torch_cluster==1.6.0, torch_spline_conv==1.2.1
- nuscenes-devkit==1.1.10
- motmetrics==1.1.3
- (optional for flash attention) einops, flash-attn==0.2.2
