#!/bin/bash

# --- Shared Training and Sampling Flags ---
TRAIN_FLAGS="--lr 1e-3 --save_interval 5000 --weight_decay 0.05 --log_interval 500"
SAMPLE_FLAGS="--batch_size 64 --num_samples 64"

# --- RPLAN Dataset Configuration ---
# To run, uncomment the following lines
# echo "Running RPLAN experiment..."
# MODEL_FLAGS_RPLAN="--dataset rplan --batch_size 512 --set_name train --target_set 8"
# CUDA_VISIBLE_DEVICES='0' python image_train.py $MODEL_FLAGS_RPLAN $TRAIN_FLAGS
# echo "RPLAN training finished."


# --- FloorSet Dataset Configuration ---
# To run, uncomment the following lines
echo "Running FloorSet experiment..."
MODEL_FLAGS_FLOORSET="--dataset floorset --batch_size 256 --set_name train"
# Note: --target_set is not needed for FloorSet as it's not split in the same way.
CUDA_VISIBLE_DEVICES='0' python image_train.py $MODEL_FLAGS_FLOORSET $TRAIN_FLAGS
echo "FloorSet training finished."


# --- Example Sampling Commands ---

# To sample from a trained RPLAN model:
# echo "Sampling from RPLAN model..."
# CUDA_VISIBLE_DEVICES='0' python image_sample.py \
#     --dataset rplan --batch_size 64 --num_samples 64 --set_name eval --target_set 8 \
#     --model_path path/to/your/rplan_model.pt

# To sample from a trained FloorSet model:
# echo "Sampling from FloorSet model..."
# CUDA_VISIBLE_DEVICES='0' python image_sample.py \
#     --dataset floorset --batch_size 64 --num_samples 64 --set_name eval \
#     --model_path path/to/your/floorset_model.pt

