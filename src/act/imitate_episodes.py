import sys
sys.path.append("/home/lucyshi/code/yay_robot/src")  # to import aloha
sys.path.append("/iris/u/lucyshi/yay_robot/src")  # for cluster
sys.path.append("/home/huzheyuan/Desktop/yay_robot/src")  # to import aloha
import torch
import numpy as np
import os
import pickle
import argparse
import wandb
import cv2
import math
import threading
import time
import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from collections import deque
from queue import Queue
from torch.optim.lr_scheduler import LambdaLR

from constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from sim_env import BOX_POSE
from utils import load_merged_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos
from aloha_pro.aloha_scripts.utils import (
    initialize_model_and_tokenizer,
    encode_text,
    crop_resize,
    is_multi_gpu_checkpoint,
    modify_real_time,
    visualize_language_correction,
    create_dataset_path,
    memory_monitor,
    save_trajectory,
)
from instructor.train import build_instructor

CROP_TOP = True  # for aloha pro, whose top camera is high
CKPT = 0  # 0 for policy_last, otherwise put the ckpt number here
AUDIO = False
option = 0
intervention_needed = threading.Event()  # flag to signal an intervention
recorded_commands = Queue()


def signal_handler(sig, frame):
    exit()


def on_press(key):
    global option
    if hasattr(key, "char") and key.char in ["1", "2", "3", "4", "5"]:
        option = int(key.char)
    else:
        option = 0


def on_release(key):
    global option
    if hasattr(key, "char") and key.char in ["1", "2", "3", "4", "5"]:
        option = 0


def predict_instruction(instructor, history_obs, history_skip_frame, query_frequency):
    # Ensuring that instructor_input has the last few observations with length history_len + 1
    # and that the last observation in history_obs is the last one in instructor_input.
    selected_indices = [
        -1 - i * max((history_skip_frame // query_frequency), 1)
        for i in range(instructor.history_len + 1)
    ]
    selected_obs = [
        history_obs[idx] for idx in selected_indices if idx >= -len(history_obs)
    ]
    selected_obs.reverse()
    instructor_input = torch.stack(selected_obs, dim=1)
    assert instructor_input.shape[1] == min(
        instructor.history_len + 1, len(history_obs)
    )

    logits, temperature = instructor(instructor_input)
    decoded_texts = instructor.decode_logits(logits, temperature)[0]
    return decoded_texts


def transcribe_from_ros(msg):
    """Listen for commands in the background."""
    global recorded_commands
    if msg.data:
        command = msg.data
        print(f"Transcribed raw command: {command}")
        if command in ["stop", "pardon", "wait"]:
            print("Stop command detected.")
            intervention_needed.set()
        else:
            if intervention_needed.is_set():
                command = modify_real_time(command)
                # Check if the command is valid after modifications
                if command and len(command.split()) > 1:
                    print(f"put into the queue: {command}")
                    recorded_commands.put(command)
            else:
                while not recorded_commands.empty():
                    command = recorded_commands.get(block=False)
                    print(f"Intervention not needed, ignoring command: {command}.")


def get_user_command():
    global recorded_commands
    if AUDIO:
        print("Listening for command...")
        command = recorded_commands.get()

        # If a valid command is detected
        if command:
            print(f"Transcribed user command: {command}")

    else:
        command = input("Please provide a command: ")
    # Removing leading numbers from the string
    command = "".join(filter(lambda x: not x.isdigit(), command))
    command = modify_real_time(command)
    return command


def generate_command_embedding(
    command, t, language_encoder, tokenizer, model, instructor=None
):
    print(f"Command at {t=}: {command}")

    command_embedding = encode_text(command, language_encoder, tokenizer, model)
    command_embedding = torch.tensor(command_embedding).cuda()
    if instructor is not None:
        command_embedding = instructor.get_nearest_embedding(command_embedding)[0]
    return command_embedding


def main(args):
    set_seed(1)

    signal.signal(signal.SIGINT, signal_handler)
    threading.Thread(
        target=memory_monitor, daemon=True
    ).start()  # Start the memory monitor thread

    # Command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    num_epochs = args["num_epochs"]
    log_wandb = args["log_wandb"]
    # Split the command by commas to get a list of commands
    commands = args["command"].split(",") if args["command"] else []
    use_language = args["use_language"]
    language_encoder = args["language_encoder"]
    multi_gpu = args["multi_gpu"]
    instructor_path = args["instructor_path"]
    history_len = args["history_len"]
    history_skip_frame = args["history_skip_frame"]
    hl_margin = args["hl_margin"]

    # Set up wandb
    if log_wandb:
        if is_eval:
            # run_name += ".eval"
            log_wandb = False
        else:
            run_name = ckpt_dir.split("/")[-1] + f".{args['seed']}"
            wandb_run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
            # check if wandb run exists
            if os.path.exists(wandb_run_id_path):
                with open(wandb_run_id_path, "r") as f:
                    saved_run_id = f.read().strip()
                wandb.init(
                    project="yay-robot",
                    entity="$your_wandb_entity",
                    name=run_name,
                    resume=saved_run_id,
                )
            else:
                wandb.init(
                    project="yay-robot",
                    entity="$your_wandb_entity",
                    name=run_name,
                    config=args,
                    resume="allow",
                )
                # Ensure the directory exists before trying to open the file
                os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
                with open(wandb_run_id_path, "w") as f:
                    f.write(wandb.run.id)

    if args["gpu"] is not None and not multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['gpu']}"
        assert torch.cuda.is_available()

    # get task parameters
    dataset_dirs = []
    num_episodes_list = []
    max_episode_len = 0

    for task in task_name:
        is_sim = task[:4] == "sim_"
        if is_sim:
            from constants import SIM_TASK_CONFIGS

            task_config = SIM_TASK_CONFIGS[task]
        else:
            from aloha_pro.aloha_scripts.constants import TASK_CONFIGS

            task_config = TASK_CONFIGS[task]

        dataset_dirs.append(task_config["dataset_dir"])
        num_episodes_list.append(task_config["num_episodes"])
        max_episode_len = max(max_episode_len, task_config["episode_len"])
        camera_names = task_config["camera_names"]

    max_skill_len = (
        args["max_skill_len"] if args["max_skill_len"] is not None else max_episode_len
    )

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": args["image_encoder"],
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "multi_gpu": multi_gpu,
        }

    elif policy_class == "Diffusion":
        policy_config = {
            "lr": args["lr"],
            "camera_names": camera_names,
            "action_dim": 14,
            "observation_horizon": 1,
            "action_horizon": 8,  # TODO not used
            "prediction_horizon": args["chunk_size"],
            "num_queries": args["chunk_size"],
            "num_inference_timesteps": 10,
            "ema_power": 0.75,
            "vq": False,
            "backbone": args["image_encoder"],
            "multi_gpu": multi_gpu,
            "is_eval": is_eval,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": args["image_encoder"],
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": max_episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "log_wandb": log_wandb,
        "use_language": use_language,
        "language_encoder": language_encoder,
        "max_skill_len": max_skill_len,
        "instructor_path": instructor_path,
        "history_len": history_len,
        "history_skip_frame": history_skip_frame,
        "hl_margin": hl_margin,
    }

    if is_eval:
        print(f"{CKPT=}")
        ckpt_names = (
            [f"policy_last.ckpt"] if CKPT == 0 else [f"policy_epoch_{CKPT}_seed_0.ckpt"]
        )
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(
                config, ckpt_name, save_episode=True, dataset_dirs=dataset_dirs
            )
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    train_dataloader, stats, _ = load_merged_data(
        dataset_dirs,
        num_episodes_list,
        camera_names,
        batch_size_train,
        max_len=max_skill_len,
        command_list=commands,
        use_language=use_language,
        language_encoder=language_encoder,
        policy_class=policy_class,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    train_bc(train_dataloader, config)


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "Diffusion":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def make_fixed_lr_scheduler(optimizer):
    return LambdaLR(optimizer, lambda epoch: 1.0)


def make_scheduler(optimizer, num_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps // 100, num_training_steps=num_steps
    )
    # scheduler = make_fixed_lr_scheduler(optimizer)
    return scheduler


def get_image(ts, camera_names, crop_top=True, save_dir=None, t=None):
    curr_images = []
    for cam_name in camera_names:
        curr_image = ts.observation["images"][cam_name]

        # Check for 'cam_high' and apply transformation
        if crop_top and cam_name == "cam_high":
            curr_image = crop_resize(curr_image)

        # Swap BGR to RGB
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

        curr_image = rearrange(curr_image, "h w c -> c h w")
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    # Center crop and resize
    original_size = curr_image.shape[-2:]
    ratio = 0.95
    curr_image = curr_image[
        ...,
        int(original_size[0] * (1 - ratio) / 2) : int(
            original_size[0] * (1 + ratio) / 2
        ),
        int(original_size[1] * (1 - ratio) / 2) : int(
            original_size[1] * (1 + ratio) / 2
        ),
    ]
    curr_image = curr_image.squeeze(0)
    resize_transform = transforms.Resize(original_size, antialias=True)
    curr_image = resize_transform(curr_image)
    curr_image = curr_image.unsqueeze(0)

    if save_dir is not None:
        # Convert torch tensors back to numpy and concatenate for visualization
        concat_images = [
            rearrange(img.cpu().numpy(), "c h w -> h w c")
            for img in curr_image.squeeze(0)
        ]
        concat_image = np.concatenate(concat_images, axis=1)
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
        img_name = (
            "init_visualize.png" if t is None else f"gpt/{t=}.png"
        )  # save image every query_frequency for ChatGPT
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, (concat_image * 255).astype(np.uint8))

    return curr_image


def eval_bc(config, ckpt_name, save_episode=True, dataset_dirs=None):
    # intervention
    import rospy
    from pynput import keyboard
    from std_msgs.msg import String

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    global option
    option = 0
    language_correction = False

    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"
    log_wandb = config["log_wandb"]
    use_language = config["use_language"]
    language_encoder = config["language_encoder"]
    max_skill_len = config["max_skill_len"]
    instructor_path = config["instructor_path"]
    history_len = config["history_len"]
    history_skip_frame = config["history_skip_frame"]
    hl_margin = config["hl_margin"]
    print(f"{hl_margin=}")
    use_instructor = instructor_path is not None

    if use_language:
        tokenizer, model = initialize_model_and_tokenizer(language_encoder)
        assert tokenizer is not None and model is not None

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    model_state_dict = torch.load(ckpt_path)["model_state_dict"]
    if is_multi_gpu_checkpoint(model_state_dict):
        print("The checkpoint was trained on multiple GPUs.")
        model_state_dict = {
            k.replace("module.", "", 1): v for k, v in model_state_dict.items()
        }
    loading_status = policy.deserialize(model_state_dict)
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    if policy_class == "Diffusion":
        post_process = (
            lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
            + stats["action_min"]
        )
    else:
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    # load environment
    if real_robot:
        from aloha_pro.aloha_scripts.real_env import make_real_env  # requires aloha
        from aloha_pro.aloha_scripts.robot_utils import move_grippers

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 25
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []

    n_existing_rollouts = (
        len([f for f in os.listdir(ckpt_dir) if f.startswith("video")])
        if save_episode
        else 0
    )
    print(f"{n_existing_rollouts=}")

    # create dataset for language_correction
    dataset_dir = dataset_dirs[-1]
    dataset_dir_language_correction = dataset_dir + "_language_correction_v3"
    if not os.path.isdir(dataset_dir_language_correction):
        os.makedirs(dataset_dir_language_correction)
    print(
        f"\nRecording language correction dataset to {dataset_dir_language_correction}"
    )

    # set up the instructor (HL policy)
    if use_instructor:
        device = torch.device("cuda")
        instructor = build_instructor(dataset_dirs, history_len, device=device)
        instructor.load_state_dict(torch.load(instructor_path, map_location=device))
        instructor.eval()

        # Keep a queue of fixed length of the previous observations
        history_obs = deque(maxlen=history_len * history_skip_frame + 1)
        history_ts = deque(maxlen=hl_margin + 1)
        history_acs = deque(maxlen=hl_margin)
    else:
        instructor = None

    # Run the background listening
    if AUDIO:
        listener_subscriber = rospy.Subscriber(
            "/audio_transcription",
            String,
            transcribe_from_ros,
            queue_size=1,
        )

    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()
        if AUDIO:
            print("Please run the ros transcriber now!")
            time.sleep(20)
        ts.observation["option"] = option
        if use_instructor:
            history_obs.clear()
            history_ts.clear()
            history_ts.append(ts)
            history_acs.clear()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        command_list = []
        command_embedding = None
        # fixed_command_list = ['pick up the bag', 'pick up the sharpie', 'put the sharpie into the bag', 'release the sharpie', 'pick up the tape', 'put the tape into the bag', 'release the tape', 'pick up the sponge', 'put the sponge into the bag', 'release the sponge', 'release the bag'] # for ablation
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

                ### query policy
                if config["policy_class"] in ["ACT", "Diffusion"]:
                    if t % query_frequency == 0:
                        curr_image = get_image(
                            ts, camera_names, save_dir=ckpt_dir if t == 0 else None
                        )
                        # put the curr_image to the end of the deque
                        if use_instructor:
                            history_obs.append(curr_image)

                        if use_language:
                            # Check if an intervention is needed; if so, language correction
                            if use_instructor and (
                                intervention_needed.is_set() or option == 2
                            ):
                                language_correction = True
                                print(
                                    f"##### Intervention needed at {t=}. Please provide an instruction: #####"
                                )
                                last_command = command
                                intervention_needed.set()
                                command = get_user_command()
                                # Reset the intervention flag after handling the input
                                intervention_needed.clear()

                                command_embedding = generate_command_embedding(
                                    command, t, language_encoder, tokenizer, model
                                )

                                # Initialize the segment data with previous observations of length hl_margin
                                ts_segment = list(history_ts)
                                ts_segment[0].observation["option"] = -1
                                segment_data = {
                                    "ts": ts_segment,
                                    "actions": list(history_acs),
                                    "command": command,
                                }  # Initialize segment data

                                (
                                    dataset_path_language_correction,
                                    episode_idx_language_correction,
                                ) = create_dataset_path(dataset_dir_language_correction)
                                # save an image of the current timestep, with predicted_instruction and command overlaid
                                visualize_language_correction(
                                    curr_image,
                                    last_command,
                                    command,
                                    dataset_dir_language_correction,
                                    episode_idx_language_correction,
                                )

                            elif t % max_skill_len == 0 and not language_correction:
                                if t < 150:  # deterministic
                                    command = "pick up the bag"  # "pick up the plate"
                                elif use_instructor:
                                    last_command = command
                                    command = predict_instruction(
                                        instructor,
                                        history_obs,
                                        history_skip_frame,
                                        query_frequency,
                                    )
                                else:
                                    intervention_needed.set()
                                    command = get_user_command()
                                    intervention_needed.clear()
                                    # try:
                                    #     command = fixed_command_list.pop(0)
                                    # except:
                                    #     command_list.append('')
                                    #     break
                                command_embedding = generate_command_embedding(
                                    command, t, language_encoder, tokenizer, model
                                )

                            assert command_embedding is not None

                        all_actions = policy(
                            qpos, curr_image, command_embedding=command_embedding
                        )
                    elif use_instructor and t % history_skip_frame == 0:
                        curr_image = get_image(ts, camera_names)
                        history_obs.append(curr_image)

                    if use_language:
                        prefix = "user" if language_correction else "prediction"
                        command_list.append(f"{prefix}: {command}")

                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                # only update if the absolute value of the action is greater than 0.1
                if np.any(np.abs(action) > 0.1):
                    target_qpos = action

                ts = env.step(target_qpos)
                ts.observation["option"] = option if not language_correction else 2
                if use_instructor:
                    history_ts.append(ts)
                    history_acs.append(action)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

                # Check if language_correction mode is active
                if language_correction:
                    # Append data to the segment_data
                    segment_data["ts"].append(ts)
                    segment_data["actions"].append(action)

                    # Check if segment is complete
                    if len(segment_data["actions"]) >= hl_margin + max_skill_len - 1:
                        # Save the segment
                        save_trajectory_thread = threading.Thread(
                            target=save_trajectory,
                            args=(
                                dataset_path_language_correction,
                                segment_data["ts"],
                                segment_data["actions"],
                                camera_names,
                                segment_data["command"],
                                None,
                            ),
                        )
                        save_trajectory_thread.start()
                        language_correction = False

                # early termination
                if option == 5:
                    break

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
        if log_wandb:
            wandb.log(
                {
                    "test/episode_return": episode_return,
                    "test/episode_highest_reward": episode_highest_reward,
                    "test/env_max_reward": env_max_reward,
                    "test/success": episode_highest_reward == env_max_reward,
                },
                step=rollout_id,
            )

        if save_episode:
            is_language_correction = (
                "ld" in instructor_path.split("/")[-2] if use_instructor else False
            )
            postfix = f"_lc" if is_language_correction else ""
            video_name = f"video{rollout_id+n_existing_rollouts}{postfix}.mp4"
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, video_name),
                cam_names=camera_names,
                command_list=command_list,
            )
            if log_wandb:
                wandb.log(
                    {
                        "test/video": wandb.Video(
                            os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
                            fps=50,
                            format="mp4",
                        )
                    },
                    step=rollout_id,
                )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    if log_wandb:
        wandb.log({"test/success_rate": success_rate, "test/avg_return": avg_return})

    listener.stop()

    return success_rate, avg_return


def forward_pass(data, policy):
    if len(data) == 5:  # use_language
        image_data, qpos_data, action_data, is_pad, command_embedding = data
        command_embedding = command_embedding.cuda()
    else:
        image_data, qpos_data, action_data, is_pad = data
        command_embedding = None
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad, command_embedding)


def train_bc(train_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    log_wandb = config["log_wandb"]
    multi_gpu = config["policy_config"]["multi_gpu"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    optimizer = make_optimizer(policy_class, policy)
    scheduler = make_scheduler(optimizer, num_epochs)

    # if ckpt_dir is not empty, prompt the user to load the checkpoint
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 2:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input()
        if load_ckpt == "y":
            # load the latest checkpoint
            latest_idx = max(
                [
                    int(f.split("_")[2])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("policy_epoch_")
                ]
            )
            ckpt_path = os.path.join(
                ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt"
            )
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model_state_dict = checkpoint["model_state_dict"]
            # The model was trained on a single gpu, now load onto multiple gpus
            if multi_gpu and not is_multi_gpu_checkpoint(model_state_dict):
                # Add "module." prefix only to the keys associated with policy.model
                model_state_dict = {
                    k if "model" not in k else f"model.module.{k.split('.', 1)[1]}": v
                    for k, v in model_state_dict.items()
                }
            # The model was trained on multiple gpus, now load onto a single gpu
            elif not multi_gpu and is_multi_gpu_checkpoint(model_state_dict):
                # Remove "module." prefix only to the keys associated with policy.model
                model_state_dict = {
                    k.replace("module.", "", 1): v for k, v in model_state_dict.items()
                }
            loading_status = policy.deserialize(model_state_dict)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(loading_status)
        else:
            print("Not loading checkpoint")
            start_epoch = 0
    else:
        start_epoch = 0

    policy.cuda()

    train_history = []
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\nEpoch {epoch}")
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        scheduler.step()
        e = epoch - start_epoch
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (e + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        epoch_summary["lr"] = np.array(scheduler.get_last_lr()[0])
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.5f} "
        print(summary_string)
        if log_wandb:
            epoch_summary_train = {f"train/{k}": v for k, v in epoch_summary.items()}
            wandb.log(epoch_summary_train, step=epoch)

        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(
                {
                    "model_state_dict": policy.serialize(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of 1000 epochs
            prune_epoch = epoch - save_ckpt_every
            if prune_epoch % 1000 != 0:
                prune_path = os.path.join(
                    ckpt_dir, f"policy_epoch_{prune_epoch}_seed_{seed}.ckpt"
                )
                if os.path.exists(prune_path):
                    os.remove(prune_path)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(
        {
            "model_state_dict": policy.serialize(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        },
        ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', nargs='+', type=str, help='List of task names', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # language correction
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--command', action='store', type=str, help='comma-separated list of commands', default='', required=False)
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0, required=False)
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--language_encoder', action='store', type=str, choices=['distilbert', 'clip'], default='distilbert', help='Type of language encoder to use: distilbert or clip', required=False)
    parser.add_argument('--max_skill_len', action='store', type=int, help='max_skill_len', required=False)
    parser.add_argument("--image_encoder", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'resnet18film', 'resnet34film', 'resnet50film','efficientnet_b0film', 'efficientnet_b3film', 'efficientnet_b5film'], help="Which image encoder to use for the BC policy.")
    parser.add_argument('--low_res', action='store', type=int, help='lower resolution by a factor', required=False, default=1)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--instructor_path', action='store', type=str, help='instructor_path', required=False)
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=2)
    parser.add_argument('--history_skip_frame', action='store', type=int, help='history_skip_frame', default=50)
    parser.add_argument('--hl_margin', action='store', type=int, help='the number of timesteps to record before and after language correction', default=100)

    main(vars(parser.parse_args()))
