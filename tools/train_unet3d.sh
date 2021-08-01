#!/usr/bin/env bash
accelerate launch --multi_gpu --num_processes 2 tools/train_unet3d.py ${@:1}
