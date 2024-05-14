# Data Preparation

## Download nuScenes
Download nuScenes dataset [here](https://www.nuscenes.org/download). Set a symbolic link to `./data/nuscenes.` 
Or if you are using docker installation, a shared volume is created [here](../docker/run.sh).

## Create mmdetection3d info files
We follow `MUTR3D` to create nuScenes info files for tracking task, which is based on the `mmdetection3d` data pre-processing pipeline.
Simply run
```
python ./tools/data_converter/nusc_track.py
```
This script will create info files for all train/val/test splits.

## Folder structure
The structure of the nuScenes dataset folder should look like.
```
nuscenes/
├──── v1.0-test
├──── v1.0-trainval
├──── maps/
├──── samples/
├──── sweeps/
├──── ada_track_infos_train.pkl
├──── ada_track_infos_val.pkl
└──── ada_track_infos_test.pkl
```

## Download pre-trtained detection models
We build our track based on [DETR3D](https://github.com/WangYueFt/detr3d) and [PETR](https://github.com/megvii-research/PETR). 
You can download following pretrained models from their repository:
- [DETR3D ResNet101](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view) from [DETR3D](https://github.com/WangYueFt/detr3d)
- [PETR VoVNet (900 queries)](https://drive.google.com/file/d/1SV0_n0PhIraEXHJ1jIdMu3iMg9YZsm8c/view) from [PETR](https://github.com/megvii-research/PETR).

Our PETR-based tracker uses 500 queries but the pretrained PETR model uses 900 queriey. 
Fortunately, [PF-Track](https://github.com/TRI-ML/PF-Track) provided a pretrained PETR model with 500 queries. 
You can download their pre-trained model following their [instruction](https://github.com/TRI-ML/PF-Track/blob/main/documents/pretrained.md#3-single-frame-detection-model-download-optional). 
Decompress the zip file and the file `f1_q5_fullres_e24.pth` is the checkpoint what we need.

Many thanks to aforemetioned open-source projects with their provided pre-trained models!