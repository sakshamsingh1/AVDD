#!/bin/bash

python evaluate.py --dataset VGG_subset \
--base_dir data/distilled_data/Distill_VGG_subset_ipc:10 \
--ipc 10 --num_eval 5 --idm_aug --num_exp 1