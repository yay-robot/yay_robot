"""
Process each episode in the specified dataset directory, align the instructions with the option values, and save the results to corresponding JSON files.

Example usage:
$ python script/instruction_segmentation.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects --count
"""

import os
import json
import numpy as np
import h5py
import argparse
from tqdm import tqdm
import glob
from collections import Counter

import sys

sys.path.append("/home/lucyshi/code/yay_robot/src")  # to import aloha
sys.path.append("/home/huzheyuan/Desktop/yay_robot/src")
from aloha_pro.aloha_scripts.utils import break_text, modify_transcription


def load_hdf5(dataset_dir, dataset_name):
    with h5py.File(os.path.join(dataset_dir, dataset_name + ".hdf5"), "r") as root:
        option = root["/observations/option"][()].astype(int)
    return option


def process_episode(dataset_dir, episode_name):
    option_values = load_hdf5(dataset_dir, episode_name)
    new_instruction_positions = np.where(option_values == -1)[0]

    with open(os.path.join(dataset_dir, f"{episode_name}.txt"), "r") as f:
        raw_content = f.read()
        raw_content = modify_transcription(raw_content)

    raw_instructions = break_text(raw_content)

    # Check if the number of -1 values match the number of instructions; if not, print a warning and continue
    if len(new_instruction_positions) != len(raw_instructions):
        print(
            f"Warning: {episode_name}\tdifference is {abs(len(new_instruction_positions) - len(raw_instructions))}\t\tnumber of instructions ({len(raw_instructions)})\tnumber of -1 values ({len(new_instruction_positions)})"
        )

    instruction_data = []

    for idx in range(len(new_instruction_positions)):
        start = new_instruction_positions[idx]
        end = (
            new_instruction_positions[idx + 1] - 1
            if idx + 1 < len(new_instruction_positions)
            else len(option_values) - 1
        )

        next_val = option_values[start + 1] if start + 1 < len(option_values) else None
        instruction_type = (
            "action"
            if next_val == 0
            else "instruction"
            if next_val in [1, 3]
            else "correction"
        )

        if raw_instructions:
            current_instruction = raw_instructions.pop(0)
            instruction_data.append(
                {
                    "command": current_instruction,
                    "start_timestep": int(start),
                    "end_timestep": int(end),
                    "type": instruction_type,
                }
            )

    with open(os.path.join(dataset_dir, f"{episode_name}.json"), "w") as json_file:
        json.dump(instruction_data, json_file, indent=4)


def extract_commands_from_json(json_dir):
    # Load all JSON files from the directory
    json_files = glob.glob(os.path.join(json_dir, "episode_*.json"))

    # Counter to tally the frequency of each command
    command_counter = Counter()

    # Extract commands from each file and update the counter
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            for instruction in data:
                command_counter[modify_transcription(instruction["command"])] += 1

    # Sort commands by frequency
    sorted_commands = command_counter.most_common()
    return sorted_commands


def main():
    parser = argparse.ArgumentParser(
        description="Segment and align instructions from transcriptions with episodes."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing episodes and transcriptions.",
    )
    parser.add_argument(
        "--count", action="store_true", help="Count the frequency of each instruction."
    )
    args = parser.parse_args()

    episodes = sorted(
        [f.split(".")[0] for f in os.listdir(args.dataset_dir) if f.endswith(".hdf5")]
    )
    for episode_name in tqdm(episodes, desc="Processing episodes"):
        # if there's any error, skip the episode, print the episode number, and continue
        try:
            process_episode(args.dataset_dir, episode_name)
        except:
            print(f"###Error processing episode {episode_name}")
            continue

    if args.count:
        commands = extract_commands_from_json(args.dataset_dir)
        with open(os.path.join(args.dataset_dir, "count.txt"), "w") as f:
            for command, frequency in commands:
                line = f"'{command}': {frequency}"
                print(line)
                f.write(line + "\n")


if __name__ == "__main__":
    main()
