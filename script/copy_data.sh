#!/bin/bash

# Array of dataset names,  "aloha_bag_3_objects_ethan_v0" "aloha_bag_3_objects_ethan_v1" "aloha_bag_3_objects"
DATASETS=("aloha_bag_3_objects_ethan_v2" "aloha_bag_3_objects_johnny_v1")

# Copy data to cluster function --ignore-existing 
copy_to_cluster() {
    local DATASET=$1
    rsync -av --exclude '*_video.mp4' --exclude '*.wav' /iris/u/lucyshi/language-dagger/data/act/$DATASET/candidate_embeddings.npy iris5:/scr/lucyshi/dataset/$DATASET/
}

# Copy data to cluster
for DATASET in "${DATASETS[@]}"; do
    copy_to_cluster $DATASET
done
