#!/bin/bash

image_name="cnn_attitude_estimator"
tag_name="docker"
root_path=$(pwd)

# /media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21
# is ssd path in author's environment

xhost +
docker run -it --rm \
	--gpus all \
	--privileged \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --net=host \
    -v /home/kawai/airsim_dataset:/home/ssd_dir \
	-v /home/kawai/cnn_attitude_estimator_log:/home/cnn_attitude_estimator_log \
	$image_name:$tag_name