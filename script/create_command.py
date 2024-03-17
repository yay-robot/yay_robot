import json
import os


def create_json_files(data_dir):
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define the commands and the corresponding indices
    commands = {
        # "pick up the sharpie": list(range(0, 25)),
        "pick up the sponge": list(range(53, 75))
    }

    # Iterate over each command and its indices
    for command, indices in commands.items():
        for idx in indices:
            # Define the filename for the episode
            episode_filename = os.path.join(data_dir, f"episode_{idx}.json")

            # Define the content of the episode
            episode_content = [
                {
                    "command": command,
                    "start_timestep": 0,
                    "end_timestep": 199,
                    "type": "instruction",
                }
            ]

            # Write the episode content to the file
            with open(episode_filename, "w") as file:
                json.dump(episode_content, file)

    print(f"Files have been created in {data_dir}")


# Define the main function to execute the script
if __name__ == "__main__":
    data_directory = input("Enter the path for the data directory: ")
    create_json_files(data_directory)
