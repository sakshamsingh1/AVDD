#!/bin/bash

num_classes=10

for ((i=0; i<num_classes; i=i+2)); do

    # First job
    echo "Running class $i"
    > data/log/log_${i}_av.txt
    CUDA_VISIBLE_DEVICES=0 python distill_DM_AV_classwise.py --input_modality av --class_num $((i)) --dataset VGG_subset --ipc 10 --wandb_disable --num_exp 1 --idm_aug > data/log/log_${i}_av.txt 2>&1 &

    # Second job (only if i+1 is less than num_classes)
    if ((i+1 < num_classes)); then
        echo "Running class $((i+1))"
        > data/log/log_${i+1}_av.txt
        CUDA_VISIBLE_DEVICES=1 python distill_DM_AV_classwise.py --input_modality av --class_num $((i+1)) --dataset VGG_subset --ipc 10 --wandb_disable --num_exp 1 --idm_aug > data/log/log_${i+1}_av.txt 2>&1 &
    fi

    wait 
done

wait