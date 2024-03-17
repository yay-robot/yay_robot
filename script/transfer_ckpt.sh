#!/bin/bash

# Check for the required argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ckpt_dir>"
    exit 1
fi

# Set the ckpt_dir variable from the first argument
ckpt_dir=$1

# Create the ckpt_dir on the remote machine if it doesn't exist
ssh iris-robot-ws-1 "mkdir -p /scr/lucyshi/ll_ckpt/${ckpt_dir}"

# Change to the specified directory
cd /iris/u/lucyshi/yay_robot/data/ll_ckpt

# Perform the scp operations
scp -rp ${ckpt_dir}/dataset_stats.pkl iris-robot-ws-1:/scr/lucyshi/ll_ckpt/${ckpt_dir}
scp -rp ${ckpt_dir}/policy_last.ckpt iris-robot-ws-1:/scr/lucyshi/ll_ckpt/${ckpt_dir}
