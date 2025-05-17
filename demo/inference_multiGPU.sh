#!/bin/bash

# Number of GPUs to use
NUM_GPUS=2

# Define paths
MODEL_NAME="edsr"
MODEL_CONFIG="configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py"
MODEL_CKPT="configs/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth"
DATASET_BASE_PATH="/home/user/mmagic/raw"
RESULTS_BASE_PATH="2024_03_29_SRresults"

# Split DATASET_BASE_PATH into subdirectories for each GPU, if not already done.
# e.g., DATASET_BASE_PATH/part0, DATASET_BASE_PATH/part1, etc.
# This step needs to be done manually or via a separate script.

# Loop to execute the script on each GPU with its corresponding data subset
for (( GPU=0; GPU<NUM_GPUS; GPU++ ))
do
    IMG_DIR="${DATASET_BASE_PATH}/part${GPU}"  # Assuming data is split into part0, part1, etc.
    RESULT_OUT_DIR="${RESULTS_BASE_PATH}/part${GPU}"  # Output directory for each part
    mkdir -p "${RESULT_OUT_DIR}"  # Ensure the output directory exists

    CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=1 --use_env \
        demo/mmagic_folder_inferencer_folder.py \
        --model-name $MODEL_NAME \
        --model-config $MODEL_CONFIG \
        --model-ckpt $MODEL_CKPT \
        --img-dir $IMG_DIR \
        --result-out-dir $RESULT_OUT_DIR &
done

# Wait for all processes to finish
wait

echo "All processes finished."
