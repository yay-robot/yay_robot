import numpy as np
import torch
import os
import h5py
import cv2
import json
import sys
sys.path.append("$PATH_TO_YAY_ROBOT/src")  # to import aloha
from torch.utils.data import DataLoader, ConcatDataset

from aloha_pro.aloha_scripts.utils import crop_resize, random_crop
from act.utils import DAggerSampler


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        history_len=5,
        prediction_offset=10,
        history_skip_frame=1,
        random_crop=False,
    ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.random_crop = random_crop

    def __len__(self):
        return len(self.episode_ids)

    def get_command_for_ts(self, episode_data, target_ts):
        for segment in episode_data:
            if segment["start_timestep"] <= target_ts <= segment["end_timestep"]:
                return torch.tensor(segment["embedding"]).squeeze(), segment["command"]
        return None, None

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        encoded_json_path = os.path.join(
            self.dataset_dir, f"episode_{episode_id}_encoded_distilbert.json"
        )

        with open(encoded_json_path, "r") as f:
            episode_data = json.load(f)

        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)

            # Sample a random curr_ts and compute the start_ts and target_ts
            total_timesteps = root["/action"].shape[0]
            prediction_offset = self.prediction_offset
            try:
                curr_ts = np.random.randint(
                    self.history_len * self.history_skip_frame,
                    total_timesteps - prediction_offset,
                )
                start_ts = curr_ts - self.history_len * self.history_skip_frame
                target_ts = curr_ts + prediction_offset
            except ValueError:
                # sample a different episode in range len(self.episode_ids)
                return self.__getitem__(np.random.randint(0, len(self.episode_ids)))

            # Retrieve the language embedding for the target_ts
            command_embedding, command_gt = self.get_command_for_ts(
                episode_data, target_ts
            )
            if command_embedding is None:
                try:
                    return self.__getitem__((index + 1) % len(self.episode_ids))
                except RecursionError:
                    print(
                        f"RecursionError: Could not find embedding for episode_id {episode_id} and target_ts {target_ts}."
                    )
                    import ipdb; ipdb.set_trace()

            # Construct the image sequences for the desired timesteps
            image_sequence = []
            for ts in range(start_ts, curr_ts + 1, self.history_skip_frame):
                image_dict = {}
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][ts]
                    if compressed:
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)
                        if cam_name == "cam_high":
                            image_dict[cam_name] = crop_resize(image_dict[cam_name])
                        if self.random_crop:
                            image_dict[cam_name] = random_crop(image_dict[cam_name])
                    image_dict[cam_name] = cv2.cvtColor(
                        image_dict[cam_name], cv2.COLOR_BGR2RGB
                    )
                all_cam_images = [
                    image_dict[cam_name] for cam_name in self.camera_names
                ]
                all_cam_images = np.stack(all_cam_images, axis=0)
                image_sequence.append(all_cam_images)

            image_sequence = np.array(image_sequence)
            image_sequence = torch.tensor(image_sequence, dtype=torch.float32)
            image_sequence = torch.einsum("t k h w c -> t k c h w", image_sequence)
            image_sequence = image_sequence / 255.0

        return image_sequence, command_embedding, command_gt


def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    batch_size_train,
    batch_size_val,
    history_len=1,
    prediction_offset=10,
    history_skip_frame=1,
    random_crop=False,
    dagger_ratio=None,
):
    assert len(dataset_dirs) == len(
        num_episodes_list
    ), "Length of dataset_dirs and num_episodes_list must be the same."
    print(f"{history_len=}, {history_skip_frame=}, {prediction_offset=}")
    if random_crop:
        print(f"Random crop enabled")
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."

    all_episode_indices = []
    last_dataset_indices = []

    for i, (dataset_dir, num_episodes) in enumerate(
        zip(dataset_dirs, num_episodes_list)
    ):
        print(f"\nData from: {dataset_dir}\n")

        # Get episode indices for current dataset
        episode_indices = [(dataset_dir, i) for i in range(num_episodes)]
        if i == len(dataset_dirs) - 1:  # Last dataset
            last_dataset_indices.extend(episode_indices)
        all_episode_indices.extend(episode_indices)

    print(f"Total number of episodes across datasets: {len(all_episode_indices)}")

    # Obtain train test split
    train_ratio = 0.95
    shuffled_indices = np.random.permutation(all_episode_indices)
    train_indices = shuffled_indices[: int(train_ratio * len(all_episode_indices))]
    val_indices = shuffled_indices[int(train_ratio * len(all_episode_indices)) :]

    # Construct dataset and dataloader for each dataset dir and merge them
    train_datasets = [
        SequenceDataset(
            [idx for d, idx in train_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            history_len,
            prediction_offset,
            history_skip_frame,
            random_crop,
        )
        for dataset_dir in dataset_dirs
    ]
    val_datasets = [
        SequenceDataset(
            [idx for d, idx in val_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            history_len,
            prediction_offset,
            history_skip_frame,
            random_crop,
        )
        for dataset_dir in dataset_dirs
    ]
    all_datasets = [
        SequenceDataset(
            [idx for d, idx in all_episode_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            history_len,
            prediction_offset,
            history_skip_frame,
            random_crop,
        )
        for dataset_dir in dataset_dirs
    ]

    merged_train_dataset = ConcatDataset(train_datasets)
    merged_val_dataset = ConcatDataset(val_datasets)
    merged_all_dataset = ConcatDataset(all_datasets)

    if dagger_ratio is not None:  # Use all data. TODO: add val_dataloader
        dataset_sizes = {
            dataset_dir: num_episodes
            for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)
        }
        dagger_sampler = DAggerSampler(
            all_episode_indices,
            last_dataset_indices,
            batch_size_train,
            dagger_ratio,
            dataset_sizes,
        )
        train_dataloader = DataLoader(
            merged_all_dataset,
            batch_sampler=dagger_sampler,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=True,
        )
        val_dataloader = None
    else:
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
        )
        val_dataloader = DataLoader(
            merged_val_dataset,
            batch_size=batch_size_val,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
        )
    test_dataloader = DataLoader(
        merged_all_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=20,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, test_dataloader


"""
Test the SequenceDataset class.

Example usage:
$ python src/instructor/dataset.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects
"""
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()

    # Parameters for the test
    camera_names = ["cam_high", "cam_low"]
    history_len = 5
    prediction_offset = 10
    num_episodes = 10  # Just to sample from the first 10 episodes for testing

    # Create a SequenceDataset instance
    dataset = SequenceDataset(
        list(range(num_episodes)),
        args.dataset_dir,
        camera_names,
        history_len,
        prediction_offset,
    )

    # Sample a random item from the dataset
    idx = np.random.randint(0, len(dataset))
    image_sequence, command_embedding, _ = dataset[idx]

    print(f"Sampled episode index: {idx}")
    print(f"Image sequence shape: {image_sequence.shape}")
    print(f"Language embedding shape: {command_embedding.shape}")

    # Save the images in the sequence
    for t in range(history_len):
        plt.figure(figsize=(10, 5))
        for cam_idx, cam_name in enumerate(camera_names):
            plt.subplot(1, len(camera_names), cam_idx + 1)
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(
                image_sequence[t, cam_idx].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB
            )
            plt.imshow(img_rgb)
            plt.title(f"{cam_name} at timestep {t}")
        plt.tight_layout()
        plt.savefig(f"plot/image_sequence_timestep_{t}.png")
        print(f"Saved image_sequence_timestep_{t}.png")
