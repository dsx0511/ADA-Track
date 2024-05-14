nusc_data_dir=$1
repo_dir=$2
exp_dir=$3

docker run \
--gpus all --shm-size=16g \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$nusc_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
--mount source=$exp_dir,target=/workspace/work_dirs,type=bind,consistency=cached \
--name=v2.0 \
-it \
ada_track:v2.0 \

