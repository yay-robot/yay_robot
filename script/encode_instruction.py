"""
This script will loop over all the "episode_{idx}.json" files in the provided dataset_dir, encode the commands using either DistilBERT or CLIP, and save the results to "episode_{idx}_encoded.json" or "episode_{idx}_encoded_clip.json" files in the same directory.

Example usage:
$ python script/encode_instruction.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects_dagger --encoder distilbert
"""

import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import sys

sys.path.append("/home/lucyshi/code/yay_robot/src")  # to import aloha
sys.path.append("/iris/u/lucyshi/yay_robot/src")  # for cluster
sys.path.append("/home/huzheyuan/Desktop/yay_robot/src")
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text


def process_file(file_path, encoder, tokenizer, model):
    with open(file_path, "r") as file:
        data = json.load(file)

    for entry in data:
        entry["embedding"] = encode_text(entry["command"], encoder, tokenizer, model)

    # Save the updated data to a new file based on the encoder type
    new_file_path = file_path.replace(".json", f"_encoded_{encoder}.json")

    with open(new_file_path, "w") as file:
        json.dump(data, file)


def generate_embeddings_file(candidate_texts, encoder, tokenizer, model, output_file):
    # List to store embeddings
    embeddings_list = []

    # Loop through each text and encode
    for text in tqdm(candidate_texts, desc="Processing candidate texts"):
        embedding = encode_text(text, encoder, tokenizer, model)
        embeddings_list.append(embedding)

    # Convert the embeddings list to a numpy array
    embeddings_array = np.array(embeddings_list)

    # Save the embeddings array to a file
    np.save(output_file, embeddings_array)
    print(f"Saved embeddings to {output_file}")


def load_candidate_texts(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        # Extract the instruction (text before the colon), strip whitespace, and then strip quotation marks
        candidate_texts = [line.split(":")[0].strip().strip("'\"") for line in lines]
    return candidate_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode commands in JSON dataset using either DistilBERT or CLIP."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the JSON files to be processed.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["distilbert", "clip"],
        default="distilbert",
        help="Encoder type to be used ('distilbert' or 'clip').",
    )
    parser.add_argument(
        "--from_count",
        action="store_true",
        help="Generate embeddings directly from instructions in 'count.txt'.",
    )
    args = parser.parse_args()

    tokenizer, model = initialize_model_and_tokenizer(args.encoder)
    dataset_dir = args.dataset_dir
    encoder = args.encoder

    if args.from_count:
        candidate_texts_path = os.path.join(dataset_dir, "count.txt")
        candidate_texts = load_candidate_texts(candidate_texts_path)
        output_file = os.path.join(dataset_dir, f"candidate_embeddings_{encoder}.npy")
        generate_embeddings_file(
            candidate_texts, encoder, tokenizer, model, output_file
        )
    else:
        # List of files to process
        files_to_process = [
            f
            for f in os.listdir(dataset_dir)
            if "episode_" in f and f.endswith(".json") and "_encoded" not in f
        ]

        # Loop over all the "episode_{idx}.json" files in the dataset_dir with a progress bar
        for file_name in tqdm(files_to_process, desc="Processing files"):
            process_file(
                os.path.join(dataset_dir, file_name), encoder, tokenizer, model
            )
