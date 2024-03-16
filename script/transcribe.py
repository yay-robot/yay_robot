"""
Transcibe all audio files in the dataset_dir.

Example usage:
$ python script/transcribe.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects
"""
import os
import subprocess
import argparse
from tqdm import tqdm
import threading
import sys
sys.path.append('/home/lucyshi/code/language-dagger/src') # to import aloha
sys.path.append('/iris/u/lucyshi/language-dagger/src') # for cluster
sys.path.append('/home/huzheyuan/Desktop/language-dagger/src') # for zheyuan
from aloha_pro.aloha_scripts.utils import memory_monitor


def generate_transcription(dataset_dir, dataset_name, model_dir='whisper_models', model='large-v2', language='English'):
    """Generate transcription using the whisper command."""
    input_path = os.path.join(dataset_dir, dataset_name + '.wav')
    output_dir = dataset_dir
    print(f"using model {model}")
    command = [
        "whisper", input_path,
        "--output_dir", output_dir,
        "--model_dir", model_dir,
        "--language", language,
        "--model", model,
        "--output_format", "txt"
    ]
    subprocess.run(command, check=True)

def main(dataset_dir):
    # Start the memory monitor thread
    threading.Thread(target=memory_monitor, daemon=True).start()

    already_transcribed = []
    just_transcribed = []

    # Get all wav files in the dataset_dir
    audio_files = [f for f in os.listdir(dataset_dir) if f.endswith('.wav')]
    
    # Filter out the already transcribed files
    to_transcribe = [f for f in audio_files if not os.path.exists(os.path.join(dataset_dir, f.rstrip('.wav') + '.txt'))]
    already_transcribed = [f for f in audio_files if f not in to_transcribe]

    # Using tqdm for a progress bar
    for audio_file in tqdm(to_transcribe, desc="Transcribing"):
        dataset_name = audio_file.rstrip('.wav')
        print(f"Currently transcribing: {dataset_name}")
        generate_transcription(dataset_dir, dataset_name)
        just_transcribed.append(dataset_name)

    # Print out the results
    if already_transcribed:
        print("\nThese audio files were already transcribed:")
        for file_name in already_transcribed:
            print(file_name)

    if just_transcribed:
        print("\nThese audio files were transcribed in this run:")
        for file_name in just_transcribed:
            print(file_name)
    else:
        print("\nNo new audio files were transcribed in this run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe .wav files into .txt using Whisper.")
    parser.add_argument('--dataset_dir', action='store', type=str, help='Directory containing .wav files.', required=True)
    args = parser.parse_args()

    main(args.dataset_dir)
