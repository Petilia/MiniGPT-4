#!/bin/bash

cd "$(dirname "$0")"

workspace_dir=$PWD


desktop_start() {
    xhost +local:
    docker run -it -d --rm \
        --gpus all \
        --ipc host \
        --network host \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name minigptv \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $workspace_dir/:/home/docker_current/MiniGPT-4:rw \
        -v /home/petryashin_ie/Projects/minigpt4/src/vicuna_weight/:/home/docker_current/model_weights/vicuna_weight:rw \
        -v /datasets/NCLT_preprocessed:/home/docker_current/datasets/nclt:rw \
        -v /datasets/RobotCar/pnvlad_oxford_robotcar_full:/home/docker_current/datasets/pnvlad_oxford_robotcar_full:rw \
        ${ARCH}/minigptv:latest
    xhost -
}

#  -v /media/cds-k/Elements/train_whisper_cache:/home/docker_current/.cache:rw \
# -v /media/cds-k/Elements/some_models_weight/whisper:/home/docker_current/whisper_weights:rw \

main () {
    ARCH="$(uname -m)"

    if [ "$ARCH" = "x86_64" ]; then
        desktop_start;
    elif [ "$ARCH" = "aarch64" ]; then
        arm_start;
    fi

}

main;