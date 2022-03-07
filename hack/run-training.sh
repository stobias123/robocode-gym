#!/bin/bash
docker pull stobias123/robocode
docker pull stobias123/robocode-gym
mkdir -p data
docker run -it --rm --name gym --privileged --net=host --gpus all -v $PWD/data:/data -v /var/run/docker.sock:/var/run/docker.sock stobias123/robocode-gym
