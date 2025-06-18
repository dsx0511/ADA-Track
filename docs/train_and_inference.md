# Training and Inference

## Training

Train with single GPU with
```
python tools/train_tracker.py $CONFIG_FILE --work-dir ./work_dirs --seed 42
```
`$CONFIG_FILE` is the path to the config file.
We provide default config files for DETR3D-based tracker `./plugin/configs/ada_track_detr3d.py` as well as for PETR-based tracker `./plugin/configs/ada_track_petr.py`.

Train with multiple GPU:
```
./ada_track/tools/dist_train_tracker.sh $CONFIG_FILE $NUM_GPUS --work-dir ./work_dirs --seed 42
```
You can set the number of GPUs in `$NUM_GPUS`.
For our default setting, we used 4 V100 (`$NUM_GPUS=4`) to train the DETR3D-based tracker and 8 A100 (`$NUM_GPUS=8`) for PETR-based tracker.

## Inference

If you want to run an inference based on a trained ADA-track checkpoint at `$CKPT_PATH`, run
```
python ./tools/test.py $CONFIG_FILE $CKPT_PATH --seed 42 --eval map --out work_dirs/results
```
We only support single-GPU inference.