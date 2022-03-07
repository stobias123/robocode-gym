#!/bin/bash
BASE_URL=https://us.download.nvidia.com/tesla
DRIVER_VERSION=450.80.02
curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
sudo sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
