import os

directory = "$PATH_TO_DATASET/aloha_bag"

# Filenames pattern
file_types = [".hdf5", ".json", ".txt", ".wav"]

# Function to rename files to make the episode numbers continuous
def rename_files(directory):
    # Find all episodes
    episodes = sorted(
        set(
            int(filename.split("_")[1].split(".")[0])
            for filename in os.listdir(directory)
            if filename.startswith("episode_")
        )
    )

    # Mapping of old episode numbers to new episode numbers
    episode_map = {old: new for new, old in enumerate(episodes)}

    # Rename files
    for old_episode, new_episode in episode_map.items():
        for file_type in file_types:
            old_filename = os.path.join(directory, f"episode_{old_episode}{file_type}")
            new_filename = os.path.join(directory, f"episode_{new_episode}{file_type}")
            if os.path.exists(old_filename) and old_episode != new_episode:
                # print(f"Renaming {old_episode}{file_type} to {new_episode}{file_type}")  # Dry run
                os.rename(old_filename, new_filename)


rename_files(directory)
