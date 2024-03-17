#!/bin/bash

DESTINATION="iris5:/scr/lucyshi/dataset"
DATASETS=("aloha_bag_3_objects_d2_v1" "aloha_bag_3_objects_d1_v0" "aloha_bag_3_objects_d1_v1" "aloha_bag_3_objects_d1_v2" "aloha_bag_3_objects")

# Encoding instructions
for DATASET in "${DATASETS[@]}"; do
    python script/encode_instruction.py --dataset_dir /scr/lucyshi/dataset/$DATASET --encoder distilbert
    python script/encode_instruction.py --dataset_dir /scr/lucyshi/dataset/$DATASET --encoder distilbert --from_count
done

# Copy data to cluster function
copy_to_cluster() {
    local DATASET=$1
    rsync -av --include 'candidate_embeddings_distilbert.npy' --include 'count.txt' --exclude '*' /scr/lucyshi/dataset/$DATASET/ $DESTINATION/$DATASET/
    rsync -av --exclude '*_video.mp4' --exclude '*.wav' --exclude '*.png' --exclude '*.txt' --ignore-existing /scr/lucyshi/dataset/$DATASET/ $DESTINATION/$DATASET/
}

# Copy data to cluster
for DATASET in "${DATASETS[@]}"; do
    copy_to_cluster $DATASET
done
