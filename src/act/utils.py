import numpy as np
import torch
import os
import random
import h5py
import torch.utils.data
from torch.utils.data import DataLoader, ConcatDataset, Sampler
import cv2
import json
from torchvision import transforms
import albumentations as A
import sys
sys.path.append('/home/lucyshi/code/language-dagger/src') # to import aloha
sys.path.append('/iris/u/lucyshi/language-dagger/src') # for cluster

from aloha_pro.aloha_scripts.utils import crop_resize

CROP_TOP = True # hardcode
AUGMENT = False # Light augmentation from albumentations
FILTER_MISTAKES = True # Filter out mistakes from the dataset even if not use_language or not use_one_hot

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_len=None, command_list=None, use_language=False, language_encoder=None, use_one_hot=False, policy_class=None):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_len = max_len
        self.command_list = [cmd.strip('\'"') for cmd in command_list]
        self.use_language = use_language
        self.language_encoder = language_encoder
        self.use_one_hot = use_one_hot
        self.policy_class = policy_class
        self.transformations = None
        
        if self.use_one_hot:
            assert not self.use_language, "Both use_language and use_one_hot cannot be True at the same time."
            assert len(self.command_list) > 0, "command_list must be provided if use_one_hot is True"

        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        max_len = self.max_len 

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        
        if self.use_language or FILTER_MISTAKES:
            json_name = f"episode_{episode_id}_encoded_{self.language_encoder}.json"
            encoded_json_path = os.path.join(self.dataset_dir, json_name)
            
            with open(encoded_json_path, 'r') as f:
                episode_data = json.load(f)

        if len(self.command_list) > 0:
            # If command_list is provided, use the JSON file to determine the relevant timesteps
            matching_segments = []
            
            for segment in episode_data:
                if segment['command'] in self.command_list:
                    current_idx = episode_data.index(segment)
                    if current_idx + 1 < len(episode_data) and episode_data[current_idx + 1]['type'] == "correction":
                        continue
                    else:
                        matching_segments.append(segment)

            # Choose a segment randomly among the matching segments
            chosen_segment = random.choice(matching_segments)

            segment_start, segment_end = chosen_segment['start_timestep'], chosen_segment['end_timestep']
            if self.use_language:
                command_embedding = torch.tensor(chosen_segment['embedding']).squeeze()
            elif self.use_one_hot:
                command_idx = self.command_list.index(chosen_segment['command'])
                command_embedding = torch.zeros(len(self.command_list))
                command_embedding[command_idx] = 1

            if segment_start is None or segment_end is None:
                raise ValueError(f"Command segment not found for episode {episode_id}")

        elif self.use_language or FILTER_MISTAKES:
            while True:
                # Randomly sample a segment
                segment = np.random.choice(episode_data)
                current_idx = episode_data.index(segment)
                if current_idx + 1 < len(episode_data) and episode_data[current_idx + 1]['type'] == "correction":
                    continue
                segment_start, segment_end = segment['start_timestep'], segment['end_timestep']
                # if end and start are too close, skip
                if segment_end - segment_start + 1 < 20:
                    continue
                command_embedding = torch.tensor(segment['embedding']).squeeze()
                break

        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs['sim']
            self.is_sim = is_sim
            compressed = root.attrs.get('compress', False)
            original_action_shape = root['/action'].shape

            if len(self.command_list) > 0 or self.use_language:
                # Sample within the segment boundaries
                start_ts = np.random.randint(segment_start, segment_end)
                end_ts = min(segment_end, start_ts + max_len - 2)
            else:
                start_ts = np.random.choice(original_action_shape[0])
                end_ts = original_action_shape[0] - 1

            # Get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]

            # Construct the image dictionary for the desired timestep
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            
            # Decompress images if they're compressed
            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

                    # Check for 'cam_high' and apply transformation
                    if CROP_TOP and cam_name == 'cam_high':
                        image_dict[cam_name] = crop_resize(image_dict[cam_name])

            # Swap BGR to RGB
            for cam_name in image_dict.keys():
                image_dict[cam_name] = cv2.cvtColor(image_dict[cam_name], cv2.COLOR_BGR2RGB)

            # Apply augmentation if the flag is set
            if AUGMENT:
                for cam_name in image_dict.keys():
                    image_dict[cam_name] = A.RandomBrightness(p=0.5)(image=image_dict[cam_name])['image']

            # Adjusting action loading
            if is_sim:
                action = root['/action'][start_ts:end_ts+1]
                action_len = end_ts - start_ts + 1
            else:
                # hack, to make timesteps more aligned
                action = root['/action'][max(0, start_ts-1):end_ts+1]
                action_len = end_ts - max(0, start_ts - 1) + 1

            # Adjusting the padded action and padding flags
            padded_action = np.zeros((max_len,) + original_action_shape[1:], dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(max_len)
            is_pad[action_len:] = 1

            # Constructing the image data for all cameras
            all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images = np.stack(all_cam_images, axis=0)

            # debugging: save images
            # if not os.path.exists("images"):
            #     os.makedirs("images")
            # for i in range(all_cam_images.shape[0]):
            #     cv2.imwrite(f"images/{i}.png", all_cam_images[i])

            # Constructing the observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # Adjusting channel
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # Augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                ]
                if self.policy_class == 'Diffusion':
                    self.transformations.extend([transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                                                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]) #, hue=0.08

            for transform in self.transformations:
                image_data = transform(image_data)

            # Normalize image data and adjust data types
            image_data = image_data / 255.0
            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

            if self.use_language or self.use_one_hot:
                return image_data, qpos_data, action_data, is_pad, command_embedding
            else:
                return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dirs, num_episodes_list):
    all_qpos_data = []
    all_action_data = []
    
    # Iterate over each directory and the corresponding number of episodes
    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                action = root['/action'][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
    
    # Concatenate data from all directories
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # Normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # Normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    eps = 0.0001

    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": all_qpos_data[-1].numpy()}  # example from the last loaded qpos

    return stats


### Merge multiple datasets
def load_merged_data(dataset_dirs, num_episodes_list, camera_names, batch_size_train, max_len=None, command_list=None, use_language=False, language_encoder=None, use_one_hot=False, dagger_ratio=None, policy_class=None):
    assert len(dataset_dirs) == len(num_episodes_list), "Length of dataset_dirs and num_episodes_list must be the same."
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."

    all_filtered_indices = []
    last_dataset_indices = []

    for i, (dataset_dir, num_episodes) in enumerate(zip(dataset_dirs, num_episodes_list)):
        print(f'\nData from: {dataset_dir}\n')
        
        # Filter episodes based on command list if provided
        filtered_indices = []

        if len(command_list) > 0:
            cleaned_commands = [cmd.strip('\'"') for cmd in command_list]
            
            for episode_id in range(num_episodes):
                json_path = os.path.join(dataset_dir, f"episode_{episode_id}.json")  
                with open(json_path, 'r') as f:
                    instruction_data = json.load(f)

                # Check for valid command segments
                for segment in instruction_data:
                    if segment['command'] in cleaned_commands:
                        current_idx = instruction_data.index(segment)
                        if current_idx + 1 < len(instruction_data) and instruction_data[current_idx + 1]['type'] == "correction":
                            continue
                        else:
                            filtered_indices.append((dataset_dir, episode_id))
                            break
        else:
            filtered_indices = [(dataset_dir, i) for i in range(num_episodes)]
        
        if i == len(dataset_dirs) - 1:  # Last dataset
            last_dataset_indices.extend(filtered_indices)
        all_filtered_indices.extend(filtered_indices)

    print(f"Total number of episodes across datasets: {len(all_filtered_indices)}")

    # Obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dirs, num_episodes_list)

    # Construct dataset and dataloader for each dataset dir and merge them
    train_datasets = [EpisodicDataset([idx for d, idx in all_filtered_indices if d == dataset_dir], dataset_dir, camera_names, norm_stats, max_len, command_list, use_language, language_encoder, use_one_hot, policy_class) for dataset_dir in dataset_dirs]
    merged_train_dataset = ConcatDataset(train_datasets)

    if dagger_ratio is not None:
        dataset_sizes = {dataset_dir: num_episodes for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)}
        dagger_sampler = DAggerSampler(all_filtered_indices, last_dataset_indices, batch_size_train, dagger_ratio, dataset_sizes)
        train_dataloader = DataLoader(merged_train_dataset, batch_sampler=dagger_sampler, pin_memory=True, num_workers=24, prefetch_factor=4, persistent_workers=True)
    else:
        # Use default shuffling if dagger_ratio is not provided
        train_dataloader = DataLoader(merged_train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=24, prefetch_factor=4, persistent_workers=True)

    return train_dataloader, norm_stats, train_datasets[-1].is_sim

### For DAgger
class DAggerSampler(Sampler):
    def __init__(self, all_indices, last_dataset_indices, batch_size, dagger_ratio, dataset_sizes):
        self.other_indices, self.last_dataset_indices = self._flatten_indices(all_indices, last_dataset_indices, dataset_sizes)
        print(f"Len of data from the last dataset: {len(self.last_dataset_indices)}, Len of data from other datasets: {len(self.other_indices)}")
        self.batch_size = batch_size
        self.dagger_ratio = dagger_ratio
        self.num_batches = len(all_indices) // self.batch_size

    @staticmethod
    def _flatten_indices(all_indices, last_dataset_indices, dataset_sizes):
        flat_other_indices = []
        flat_last_dataset_indices = []
        cumulative_size = 0

        for dataset_dir, size in dataset_sizes.items():
            for idx in range(size):
                if (dataset_dir, idx) in last_dataset_indices:
                    flat_last_dataset_indices.append(cumulative_size + idx)
                elif (dataset_dir, idx) in all_indices:
                    flat_other_indices.append(cumulative_size + idx)
            cumulative_size += size

        return flat_other_indices, flat_last_dataset_indices

    def __iter__(self):
        num_samples_last = int(self.batch_size * self.dagger_ratio)
        num_samples_other = self.batch_size - num_samples_last

        for _ in range(self.num_batches):
            batch_indices = []

            if num_samples_last > 0 and self.last_dataset_indices:
                batch_indices.extend(np.random.choice(self.last_dataset_indices, num_samples_last, replace=True))

            if num_samples_other > 0 and self.other_indices:
                batch_indices.extend(np.random.choice(self.other_indices, num_samples_other, replace=True))

            np.random.shuffle(batch_indices)  # shuffle within each batch
            yield batch_indices

    def __len__(self):
        return self.num_batches


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def number_to_one_hot(number, size=501):
    one_hot_array = np.zeros(size)
    one_hot_array[number] = 1
    return one_hot_array
