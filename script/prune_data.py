"""
Example usage:
python script/prune_data.py --data_dir /scr/lucyshi/dataset/aloha_bag_3_objects --start_idx 0 --end_idx 65
"""

import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm

import os
import shutil
import h5py
import numpy as np
import argparse
from tqdm import tqdm


def process_hdf5_data(file_path, output_dir):
    with h5py.File(file_path, 'r') as f:
        # Read data
        qpos_data = f['/observations/qpos'][:]
        qvel_data = f['/observations/qvel'][:]
        effort_data = f['/observations/effort'][:]
        options_data = f['/observations/option'][:]
        actions_data = f['/action'][:]
        images_data = {cam_name: f[f'/observations/images/{cam_name}'][:] for cam_name in f['/observations/images'].keys()}
        compress_len_data = f['/compress_len'][:] if '/compress_len' in f else None

        new_data = {
            'qpos': [],
            'qvel': [],
            'effort': [],
            'option': [],
            'action': [],
            'images': {cam_name: [] for cam_name in images_data.keys()}
        }

        idx = 0
        while idx < len(options_data):
            # If option is 0 and not following a 1 or 2
            if options_data[idx] == 0:
                new_data['qpos'].append(qpos_data[idx])
                new_data['qvel'].append(qvel_data[idx])
                new_data['effort'].append(effort_data[idx])
                new_data['option'].append(0)
                new_data['action'].append(actions_data[idx])
                for cam_name in images_data.keys():
                    new_data['images'][cam_name].append(images_data[cam_name][idx])
                idx += 1
                continue

            # If option is 1 or 2
            current_option = int(options_data[idx])
            while idx < len(options_data) and options_data[idx] == current_option:
                idx += 1

            # Add transition marker
            new_data['option'].append(-1)
            new_data['qpos'].append(qpos_data[idx - 1])  # Repeated data for transition marker
            new_data['qvel'].append(qvel_data[idx - 1])
            new_data['effort'].append(effort_data[idx - 1])
            new_data['action'].append(actions_data[idx - 1])
            for cam_name in images_data.keys():
                new_data['images'][cam_name].append(images_data[cam_name][idx - 1])

            # Change following zeros to the current option
            while idx < len(options_data) and options_data[idx] == 0:
                new_data['qpos'].append(qpos_data[idx])
                new_data['qvel'].append(qvel_data[idx])
                new_data['effort'].append(effort_data[idx])
                new_data['option'].append(current_option)
                new_data['action'].append(actions_data[idx])
                for cam_name in images_data.keys():
                    new_data['images'][cam_name].append(images_data[cam_name][idx])
                idx += 1

        # Write processed data to new hdf5 file
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        with h5py.File(output_path, 'w') as out_f:
            # Copy over attributes
            for key, value in f.attrs.items():
                out_f.attrs[key] = value

            # Save datasets
            out_f.create_dataset('/observations/qpos', data=np.array(new_data['qpos']))
            out_f.create_dataset('/observations/qvel', data=np.array(new_data['qvel']))
            out_f.create_dataset('/observations/effort', data=np.array(new_data['effort']))
            out_f.create_dataset('/observations/option', data=np.array(new_data['option']))
            out_f.create_dataset('/action', data=np.array(new_data['action']))
            
            if compress_len_data is not None:
                # Adjust the compress_len for the transition markers
                new_compress_len = []
                idx = 0
                while idx < len(options_data):
                    if options_data[idx] == 0:
                        new_compress_len.append(compress_len_data[:, idx])
                        idx += 1
                        continue

                    current_option = int(options_data[idx])
                    while idx < len(options_data) and options_data[idx] == current_option:
                        idx += 1

                    # Add entry for transition marker
                    new_compress_len.append(compress_len_data[:, idx - 1])

                    while idx < len(options_data) and options_data[idx] == 0:
                        new_compress_len.append(compress_len_data[:, idx])
                        idx += 1

                out_f.create_dataset('/compress_len', data=np.array(new_compress_len).T)

            img_grp = out_f.create_group('/observations/images')
            for cam_name, img_array in new_data['images'].items():
                img_grp.create_dataset(cam_name, data=np.array(img_array))

    return len(new_data['option'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process HDF5 files to remove frozen timesteps and relabel options.')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing the HDF5 files.')
    parser.add_argument('--start_idx', type=int, required=True, help='Starting episode index.')
    parser.add_argument('--end_idx', type=int, required=True, help='Ending episode index.')
    args = parser.parse_args()

    # Create output directory with "_new" suffix
    output_dir = args.data_dir + "_new"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate list of hdf5 files to be processed based on start_idx and end_idx
    hdf5_files = [os.path.join(args.data_dir, f"episode_{i}.hdf5") for i in range(args.start_idx, args.end_idx + 1)]
    max_episode_length = 0
    processed_lengths = []

    # Process each hdf5 file
    for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 files"):
        if os.path.exists(hdf5_file):
            episode_length = process_hdf5_data(hdf5_file, output_dir)
            processed_lengths.append(episode_length)
            if episode_length > max_episode_length:
                max_episode_length = episode_length
        else:
            print(f"File {hdf5_file} not found.")
    
    print("Done processing all hdf5 files.")
    print(f"Processed Episode Lengths: {processed_lengths}")
    print(f"Max Episode Length: {max_episode_length}")

    # Move .wav files
    for i in range(args.start_idx, args.end_idx + 1):
        # data_dir is args.data_dir but without _full in the name
        data_dir = args.data_dir.replace("_full", "")
        wav_file = os.path.join(data_dir, f"episode_{i}.wav")
        if os.path.exists(wav_file):
            shutil.move(wav_file, output_dir)


