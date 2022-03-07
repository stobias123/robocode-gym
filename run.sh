#!/bin/bash
mkdir -p /data
python train.py --algo ppo --env Robocode-v2 \
                           --gym-packages gym_robocode \
                           --tensorboard-log /data/stable-baselines/  \
                           --save-freq 10000 \
                           -f /data/logs \
                           --optimization-log-path /data/opt_logs\
