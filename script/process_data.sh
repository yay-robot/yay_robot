#!/bin/bash

DATASETS=("aloha_bag_3_objects" "aloha_trail_mix")

# Encoding instructions
for DATASET in "${DATASETS[@]}"; do
    python script/encode_instruction.py --dataset_dir /scr/lucyshi/dataset/$DATASET --encoder distilbert
    python script/encode_instruction.py --dataset_dir /scr/lucyshi/dataset/$DATASET --encoder distilbert --from_count
done

# Copy data to cluster
# DESTINATION="iris5:/scr/lucyshi/dataset"

# copy_to_cluster() {
#     local DATASET=$1
#     rsync -av --include 'candidate_embeddings_distilbert.npy' --include 'count.txt' --exclude '*' /scr/lucyshi/dataset/$DATASET/ $DESTINATION/$DATASET/
#     rsync -av --exclude '*_video.mp4' --exclude '*.wav' --exclude '*.png' --exclude '*.txt' --ignore-existing /scr/lucyshi/dataset/$DATASET/ $DESTINATION/$DATASET/
# }

# for DATASET in "${DATASETS[@]}"; do
#     copy_to_cluster $DATASET
# done
