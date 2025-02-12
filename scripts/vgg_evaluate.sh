# Evaluate VGG with ipc1
python evaluate.py --dataset VGG \
--base_dir data/syn_data/vgg_ipc1 \
--ipc 1 --num_eval 5 --idm_aug \

# Evaluate VGG with ipc20
# python evaluate.py --dataset VGG \
# --base_dir data/syn_data/vgg_ipc20 \
# --ipc 20 --num_eval 5 --idm_aug \