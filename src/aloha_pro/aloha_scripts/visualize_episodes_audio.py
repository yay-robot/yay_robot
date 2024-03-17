""" 
Visualize an episode, optionally with audio.

Example usage:
$ python3 visualize_episodes_audio.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects --episode_idx 0 --visualize_option --merge_audio --transcribe

Note: this script does not visualize the depth camera images.
"""

import os
import numpy as np
import cv2
import h5py
import argparse
import subprocess
import json
import psutil
import threading
import time
from tqdm import tqdm

from constants import DT
from utils import break_text, modify_transcription, generate_transcription, crop_resize

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

MEMORY_BUFFER_MB = 1000  # The amount of memory to ensure remains free

# Automatically kill the job if it’s going to exceed the memory limit. TODO: move to utils
def memory_monitor():
    while True:
        available_memory = psutil.virtual_memory().available / (1024 ** 2)  # Available memory in MB
        if available_memory < MEMORY_BUFFER_MB:
            print(f"Available memory is too low! {available_memory:.2f}MB left. Terminating...")
            os._exit(1)  # Forcefully exit the process
        time.sleep(5)  # Check every 5 seconds


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        option = root['/observations/option'][()].astype(int)
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            # skip depth images
            if '_depth' in cam_name:
                continue
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                # image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list


    return qpos, qvel, effort, action, image_dict, option


def get_duration(file_path):
    """Get the duration of a media file using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "json", 
        file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(json.loads(result.stdout)['format']['duration'])


def merge_audio(dataset_dir, dataset_name):
    video_path = os.path.join(dataset_dir, dataset_name + '_video_key.mp4')
    audio_path = os.path.join(dataset_dir, dataset_name + '.wav')
    output_path = os.path.join(dataset_dir, dataset_name + '_video_with_audio.mp4')

    # Calculate the speedup ratio to adjust the audio speed
    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)
    speedup_ratio = audio_duration / video_duration

    # Combine the video and audio
    command = [
        "ffmpeg",
        "-i", video_path,
        "-filter_complex", f"[1:a]atempo={speedup_ratio}[aout]", # speed up audio
        "-i", audio_path,
        "-map", "0:v", # map video from first input
        "-map", "[aout]", # map audio from filter complex
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]

    # Execute the command
    subprocess.run(command, check=True)


def save_videos(video, dt, options=None, video_path=None, instructions=None, crop_top=False):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 1.5 
    font_thickness = 3

    option_labels = {0: "action", 1: "instruction", 2: "correction", 3: "dagger", -1: "new instruction"}
    current_instruction = None

    # Helper function to update the current instruction
    def update_instruction(options, ts, instructions):
        nonlocal current_instruction
        if options[ts] == -1:
            if instructions and len(instructions) > 0:
                current_instruction = instructions.pop(0).strip()
        return current_instruction

    # Helper function to overlay texts
    def overlay_texts(image, option):
        # Overlay option label on the image
        option_value = option.item() if isinstance(option, np.ndarray) else option
        cv2.putText(image, option_labels[option_value], (10, 50), font, font_scale, (0, 255, 255), font_thickness, lineType=cv2.LINE_AA)
        if current_instruction:
            cv2.putText(image, current_instruction, (10, 100), font, font_scale, (255, 255, 0), font_thickness, lineType=cv2.LINE_AA)
        return image

    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                # Check for 'cam_high' and apply transformation
                if crop_top and cam_name == 'cam_high':
                    image = crop_resize(image)
                images.append(image)
            images = np.concatenate(images, axis=1)
            
            # Overlay option and instruction only after concatenating all camera views
            if options is not None:
                update_instruction(options, ts, instructions)
                images = overlay_texts(images, options[ts])

            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            cam_video = video[cam_name]
            
            # Check for 'cam_high' and apply transformation to all frames in the video for that camera
            if crop_top and cam_name == 'cam_high':
                cam_video = np.array([crop_resize(frame) for frame in cam_video])
            
            all_cam_videos.append(cam_video)
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image.copy()
            if options is not None:
                update_instruction(options, t, instructions)
                image = overlay_texts(image, options[t])
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def main(args):
    # Start the memory monitor thread
    threading.Thread(target=memory_monitor, daemon=True).start()

    dataset_dir = args['dataset_dir']
    start_episode_idx = args['start_episode_idx']
    end_episode_idx = args['end_episode_idx'] if args['end_episode_idx'] is not None else start_episode_idx
    
    for episode_idx in tqdm(range(start_episode_idx, end_episode_idx + 1)):
        dataset_name = f'episode_{episode_idx}'

        # If the video already exists, skip
        if os.path.isfile(os.path.join(dataset_dir, dataset_name + '_video.mp4')):
            print(f"Video already exists at {os.path.join(dataset_dir, dataset_name + '_video.mp4')}")
            continue

        qpos, qvel, effort, action, image_dict, option = load_hdf5(dataset_dir, dataset_name)

        if not args['visualize_option']:
            option = None

        instructions = None
        if args['transcribe']:
            file_path = os.path.join(dataset_dir, dataset_name + '.txt')
            if not os.path.isfile(file_path):
                generate_transcription(dataset_dir, dataset_name)
            else:
                print(f"Transcription already exists at {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
                content = modify_transcription(content)
                instructions = break_text(content)

        save_videos(image_dict, DT, options=option, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'), instructions=instructions, crop_top=args['crop_top'])

        if args['merge_audio']:
            merge_audio(dataset_dir, dataset_name)
            # delete the video without audio
            os.remove(os.path.join(dataset_dir, dataset_name + '_video.mp4'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--start_episode_idx', action='store', type=int, help='Start episode index.', required=True)
    parser.add_argument('--end_episode_idx', action='store', type=int, help='End episode index.', required=False)
    parser.add_argument('--visualize_option', action='store_true', default=False, help='Overlay option text on video frames.')
    parser.add_argument('--merge_audio', action='store_true', default=False, help='Merge video with sped-up audio.')
    parser.add_argument('--transcribe', action='store_true', default=False, help='Overlay instructions from text file on video frames.')
    parser.add_argument('--crop_top', action='store_true', default=False, help='Crop the top of the high camera view.')

    main(vars(parser.parse_args()))