import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, effort, action, image_dict

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    save_n_videos = args['save_n_videos']

    if save_n_videos:
        os.makedirs(os.path.join(dataset_dir, 'videos/'), exist_ok=True)
        for idx in range(episode_idx):
            dataset_name = f'episode_{idx}'

            _, _, _, _, image_dict = load_hdf5(dataset_dir, dataset_name)
            save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, 'videos/', dataset_name + '_video.mp4'))
    else:
        dataset_name = f'episode_{episode_idx}'

        qpos, qvel, effort, action, image_dict = load_hdf5(dataset_dir, dataset_name)
        save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
        visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
        visualize_single(effort, 'effort', plot_path=os.path.join(dataset_dir, dataset_name + '_effort.png'))
        visualize_single(action - qpos, 'tracking_error', plot_path=os.path.join(dataset_dir, dataset_name + '_error.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back


# def save_videos(video, dt, video_path=None):
#     if isinstance(video, list):
#         cam_names = list(video[0].keys())
#         h, w, _ = video[0][cam_names[0]].shape
#         w = w * len(cam_names)
#         h = h * 2
#         fps = int(1/dt)
#         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         for ts, image_dict in enumerate(video):
#             rgb_images = []
#             depth_images = []
#             for cam_name in cam_names:
#                 rgb_image = image_dict[cam_name]
#                 depth_image = image_dict[f'{cam_name}_depth']
#                 # image = image[:, :, [2, 1, 0]] # swap B and R channel
#                 rgb_images.append(rgb_image)
#                 depth_images.append(depth_image)
#             rgb_images_cat = np.concatenate(rgb_images, axis=1)
#             depth_images_cat = np.concatenate(depth_images, axis=1)
#             depth_images_padded = np.concatenate([depth_images_cat, np.zeros((*depth_images_cat.shape[:-1], 2))], axis=-1)
#             images = np.concatenate([rgb_images_cat, depth_images_padded], axis=0)
#             out.write(images)
#         out.release()
#         print(f'Saved video to: {video_path}')
#     elif isinstance(video, dict):
#         cam_names = list(video.keys())
#         all_cam_videos = []
#         for cam_name in cam_names:
#             all_cam_videos.append(video[cam_name])
#         all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

#         n_frames, h, w, _ = all_cam_videos.shape
#         fps = int(1 / dt)
#         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         for t in range(n_frames):
#             image = all_cam_videos[t]
#             # image = image[:, :, [2, 1, 0]]  # swap B and R channel
#             out.write(image)
#         out.release()
#         print(f'Saved video to: {video_path}')


def save_videos(video, dt, video_path=None):
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
                # image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())

        rgb_videos = []
        depth_videos = []
        for cam_name in cam_names:
            if 'depth' in cam_name:
                depth_videos.append(video[cam_name])
            else:
                rgb_videos.append(video[cam_name])
        rgb_videos = np.concatenate(rgb_videos, axis=2) # width dimension

        if len(depth_videos) > 0:
            depth_videos = np.concatenate(depth_videos, axis=2) # width dimension

            converted_depth_videos = []
            for depth_image in depth_videos.squeeze():
                depth_image_converted = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                converted_depth_videos.append(depth_image_converted)
            converted_depth_videos = np.stack(converted_depth_videos, axis=0)

            all_cam_videos = np.concatenate([rgb_videos, converted_depth_videos], axis=1) # height dimension
        else:
            all_cam_videos = rgb_videos

        # all_cam_videos = []
        # for cam_name in cam_names:
        #     all_cam_videos.append(video[cam_name])
        # all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            # image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--save_n_videos', action='store_true', help='Flag to convert data in dir to videos.', required=False)
    main(vars(parser.parse_args()))