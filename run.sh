#!/bin/bash
mkdir -p /data
python train.py --algo ppo --env Robocode-v0 \
                           --gym-packages gym_robocode \
                           --tensorboard-log /data/stable-baselines/ 
                           --save-freq 2500 \
                           -f /data/logs \
                           --optimization-log-path /data/opt_logs\
