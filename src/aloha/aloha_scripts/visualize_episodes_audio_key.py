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

import matplotlib.pyplot as plt
from constants import DT

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
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        if compressed:
            compress_len = root['/compress_len'][()]

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                image_len = int(compress_len[cam_id, frame_id])
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


def generate_transcription(dataset_dir, dataset_name, model_dir='whisper_models', model='medium', language='English'):
    """Generate transcription using the whisper command."""
    input_path = os.path.join(dataset_dir, dataset_name + '.wav')
    output_dir = dataset_dir
    command = [
        "whisper", input_path,
        "--output_dir", output_dir,
        "--model_dir", model_dir,
        "--language", language,
        "--model", model,
        "--output_format", "txt"
    ]
    subprocess.run(command, check=True)


def save_videos(video, dt, options=None, video_path=None, instructions=None):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 1.5 
    font_color = (0, 255, 255) 
    font_thickness = 3
    option_labels = {0: "action", 1: "instruction", 2: "correction"}

    def overlay_text_on_image(image, ts, options, instructions):
        """Helper function to overlay text on a given image."""
        image_copy = image.copy()
        current_instruction = None

        if options is not None:
            cv2.putText(image_copy, option_labels[options[ts]], (10, 50), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
            
            if instructions is not None:
                if options[ts] in [1, 2] and (ts == 0 or options[ts-1] not in [1, 2]):
                    current_instruction = instructions.pop(0).strip()
                if current_instruction:
                    cv2.putText(image_copy, current_instruction, (10, 80), font, font_scale, (0, 255, 255), font_thickness, lineType=cv2.LINE_AA)

        return image_copy

    fps = int(1 / dt)
    
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for ts, image_dict in enumerate(video):
            images = [overlay_text_on_image(image_dict[cam_name][:, :, [2, 1, 0]], ts, options, instructions) for cam_name in cam_names]
            out.write(np.concatenate(images, axis=1))

    elif isinstance(video, dict):
        cam_names = list(video.keys())
        n_frames, h, w, _ = next(iter(video.values())).shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * len(cam_names), h))

        for ts in range(n_frames):
            images = [overlay_text_on_image(video[cam_name][ts][:, :, [2, 1, 0]], ts, options, instructions) for cam_name in cam_names]
            out.write(np.concatenate(images, axis=1))

    out.release()
    print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_single(efforts_list, label, plot_path=None, ylim=None, label_overwrite=None):
    efforts = np.array(efforts_list) # ts, dim
    num_ts, num_dim = efforts.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()


def main(args):
    # Start the memory monitor thread
    threading.Thread(target=memory_monitor, daemon=True).start()

    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    qpos, qvel, effort, action, image_dict, option = load_hdf5(dataset_dir, dataset_name)

    if args['visualize_option']:
        output_name = '_video_key.mp4'
    else:
        output_name = '_video.mp4'
        option = None

    instructions = None
    if args['transcribe']:
        generate_transcription(dataset_dir, dataset_name)
        with open(os.path.join(dataset_dir, dataset_name + '.txt'), 'r') as f:
            instructions = f.readlines()

    save_videos(image_dict, DT, options=option, video_path=os.path.join(dataset_dir, dataset_name + output_name), instructions=instructions)

    if args['merge_audio']:
        merge_audio(dataset_dir, dataset_name)

    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    visualize_single(effort, 'effort', plot_path=os.path.join(dataset_dir, dataset_name + '_effort.png'))
    visualize_single(action - qpos, 'tracking_error', plot_path=os.path.join(dataset_dir, dataset_name + '_error.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--visualize_option', action='store_true', default=False, help='Overlay option text on video frames.')
    parser.add_argument('--merge_audio', action='store_true', default=False, help='Merge video with sped-up audio.')
    parser.add_argument('--transcribe', action='store_true', default=False, help='Overlay instructions from text file on video frames.')

    main(vars(parser.parse_args()))
