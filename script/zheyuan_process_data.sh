#!/bin/bash

DATASETS=("aloha_plate_sponge")

# Encoding instructions
for DATASET in "${DATASETS[@]}"; do
    python script/encode_instruction.py --dataset_dir /data2/data_collection/$DATASET --encoder distilbert
    python script/encode_instruction.py --dataset_dir /data2/data_collection/$DATASET --encoder clip
    python script/encode_instruction.py --dataset_dir /data2/data_collection/$DATASET --encoder clip --from_count
done
