#!/bin/bash

DATASETS=("aloha_bag_3_objects" "aloha_trail_mix")
PATH_TO_DATASET="/scr/lucyshi/dataset"

# Encoding instructions
for DATASET in "${DATASETS[@]}"; do
    python script/encode_instruction.py --dataset_dir $PATH_TO_DATASET/$DATASET --encoder distilbert
    python script/encode_instruction.py --dataset_dir $PATH_TO_DATASET/$DATASET --encoder distilbert --from_count
done

# Copy data to cluster
# DESTINATION="iris5:$PATH_TO_DATASET"

# copy_to_cluster() {
#     local DATASET=$1
#     rsync -av --include 'candidate_embeddings_distilbert.npy' --include 'count.txt' --exclude '*' $PATH_TO_DATASET/$DATASET/ $DESTINATION/$DATASET/
#     rsync -av --exclude '*_video.mp4' --exclude '*.wav' --exclude '*.png' --exclude '*.txt' --ignore-existing $PATH_TO_DATASET/$DATASET/ $DESTINATION/$DATASET/
# }

# for DATASET in "${DATASETS[@]}"; do
#     copy_to_cluster $DATASET
# done
