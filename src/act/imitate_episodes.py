import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import wandb
import cv2
from torch.optim.lr_scheduler import LambdaLR
import math
from torchvision import transforms
from collections import deque
import threading
from queue import Queue
import sys
sys.path.append('/home/lucyshi/code/language-dagger/src') # to import aloha
sys.path.append('/iris/u/lucyshi/language-dagger/src') # for cluster
sys.path.append('/home/huzheyuan/Desktop/language-dagger/src') # to import aloha

from constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from utils import load_merged_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text, crop_resize, center_crop, is_multi_gpu_checkpoint, modify_transcription, modify_real_time
from instructor.train import build_instructor
### dagger ###
CLUSTER = False
if not CLUSTER:
    from pynput import keyboard
    import time
    import h5py_cache
    import signal
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
    from aloha_pro.aloha_scripts.robot_utils import move_arms, torque_on, torque_off, get_arm_joint_positions, get_arm_gripper_positions, move_grippers
    from aloha_pro.aloha_scripts.real_env import get_action
    from aloha_pro.aloha_scripts.utils import visualize_language_dagger, create_dataset_path, memory_monitor
    import rospy
    from std_msgs.msg import String

from sim_env import BOX_POSE

import IPython
e = IPython.embed

CROP_TOP = True # hardcode
ONLY_RIGHT = False
CKPT = 29100 # 0 for policy_last
AUDIO = False
DAGGER = False
option = 0
intervention_needed = threading.Event() # flag to signal an intervention
recorded_commands = Queue()


def signal_handler(sig, frame):
    exit()

def on_press(key):
    global option
    if hasattr(key, 'char') and key.char in ['1', '2', '3', '4', '5']:
        option = int(key.char)
    else:
        option = 0

def on_release(key):
    global option
    if hasattr(key, 'char') and key.char in ['1', '2', '3', '4', '5']:
        option = 0

def sync_puppet_to_master(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    print("\nSyncing!")

    # activate master arms
    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # get puppet arm positions
    puppet_left_qpos = get_arm_joint_positions(puppet_bot_left)
    puppet_right_qpos = get_arm_joint_positions(puppet_bot_right)

    # get puppet gripper positions
    puppet_left_gripper = get_arm_gripper_positions(puppet_bot_left)
    puppet_right_gripper = get_arm_gripper_positions(puppet_bot_right)

    # move master arms to puppet positions
    move_arms([master_bot_left, master_bot_right], [puppet_left_qpos, puppet_right_qpos], move_time=1)

    # move master grippers to puppet positions
    move_grippers([master_bot_left, master_bot_right], [puppet_left_gripper, puppet_right_gripper], move_time=1)

def teleop(env, master_bot_left, master_bot_right, dataset_dir=None, ts=None, camera_names=None, image_list=None, command=None):
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'\nTeleop started')

    # teleop loop
    global option
    dataset_path, episode_idx = create_dataset_path(dataset_dir)
    ts.observation['option'] = -1 # indicate the start
    timesteps = [ts]
    actions = []

    while True:
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        ts.observation['option'] = option
        timesteps.append(ts)
        actions.append(action)
        image_list.append(ts.observation['images'])

        # stop if the 3rd pedal is released
        if option == 0:
            print("\nReleased Pedal 3")
            break

    return save_trajectory(dataset_path, timesteps, actions, camera_names, command, image_list)

def save_trajectory(dataset_path, timesteps, actions, camera_names, command, image_list=None):
    # save trajectory
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    - option                (1,)          'int'
    
    action                  (14,)         'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/observations/option': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        option_expanded = np.expand_dims(np.array(ts.observation['option']), axis=0)
        data_dict['/observations/option'].append(option_expanded)
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list_data = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list_data:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        # print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        # print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
    max_timesteps = len(data_dict['/action'])
    with h5py_cache.File(dataset_path + '.hdf5', 'w', chunk_cache_mem_size=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )
            else:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = obs.create_dataset('qvel', (max_timesteps, 14))
        _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = obs.create_dataset('option', (max_timesteps, 1))
        _ = root.create_dataset('action', (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len
    
    # save command in a txt file
    command_path = dataset_path + '.txt'
    with open(command_path, 'w') as f:
        f.write(command)

    # print(f'Saving: {time.time() - t0:.1f} secs')
    return ts, image_list

def predict_instruction(instructor, history_obs, history_skip_frame, query_frequency):
    # Ensuring that instructor_input has the last few observations with length history_len + 1
    # and that the last observation in history_obs is the last one in instructor_input.
    selected_indices = [
        -1 - i * max((history_skip_frame // query_frequency), 1)
        for i in range(instructor.history_len + 1)
    ]
    selected_obs = [
        history_obs[idx]
        for idx in selected_indices
        if idx >= -len(history_obs)
    ]
    selected_obs.reverse()
    instructor_input = torch.stack(selected_obs, dim=1)
    assert instructor_input.shape[1] == min(instructor.history_len + 1, len(history_obs))
    
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
    command = ''.join(filter(lambda x: not x.isdigit(), command))    
    command = modify_real_time(command)
    return command

def generate_command_embedding(command, t, language_encoder, tokenizer, model, use_one_hot=False, instructor=None):
    print(f"Command at {t=}: {command}")
    
    if use_one_hot: # TODO: map from command_list, currently limited to 2 commands
        # Assuming command can be either '0' or '1'.
        if command == '0':
            command_embedding = torch.tensor([1, 0], dtype=torch.float32)
        elif command == '1':
            command_embedding = torch.tensor([0, 1], dtype=torch.float32)
        else:
            raise ValueError("Invalid command input for one-hot encoding.")  
        command_embedding = command_embedding.cuda().unsqueeze(0)
    else:
        command_embedding = encode_text(command, language_encoder, tokenizer, model)
        command_embedding = torch.tensor(command_embedding).cuda()
        if instructor is not None:
            command_embedding = instructor.get_nearest_embedding(command_embedding)[0]
    return command_embedding

def main(args):
    set_seed(1)

    if not CLUSTER:
        signal.signal(signal.SIGINT, signal_handler)
        # Start the memory monitor thread
        threading.Thread(target=memory_monitor, daemon=True).start()

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    num_epochs = args['num_epochs']
    log_wandb = args['log_wandb']
    # Split the command by commas to get a list of commands
    commands = args['command'].split(',') if args['command'] else []
    use_language = args['use_language']
    language_encoder = args['language_encoder']
    use_one_hot = args['use_one_hot']
    multi_gpu = args['multi_gpu']
    instructor_path = args['instructor_path']
    history_len = args['history_len']
    history_skip_frame = args['history_skip_frame']
    hl_margin = args['hl_margin']

     # set up wandb
    if log_wandb:
        if is_eval:
            # run_name += ".eval"
            log_wandb = False
        else:
            run_name = ckpt_dir.split("/")[-1] + f".{args['seed']}"
            wandb_run_id_path = os.path.join(ckpt_dir, 'wandb_run_id.txt')
            # check if it exists
            if os.path.exists(wandb_run_id_path):
                with open(wandb_run_id_path, 'r') as f:
                    saved_run_id = f.read().strip()
                wandb.init(project="language-dagger", entity="lucys", name=run_name, resume=saved_run_id)
                # wandb.init(project="language-dagger", entity="dmc_hand", name=run_name, resume=saved_run_id)
            else:
                wandb.init(project="language-dagger", entity="lucys", name=run_name, config=args, resume='allow')
                # wandb.init(project="language-dagger", entity="dmc_hand", name=run_name, config=args, resume='allow')  
                # Ensure the directory exists before trying to open the file
                os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
                with open(wandb_run_id_path, 'w') as f:
                    f.write(wandb.run.id)                 

    if args['gpu'] is not None and not multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['gpu']}"
        assert torch.cuda.is_available()

    # get task parameters
    dataset_dirs = []
    num_episodes_list = []
    max_episode_len = 0

    for task in task_name:
        is_sim = task[:4] == 'sim_'
        if is_sim:
            from constants import SIM_TASK_CONFIGS
            task_config = SIM_TASK_CONFIGS[task]
        else:
            from aloha_pro.aloha_scripts.constants import TASK_CONFIGS
            task_config = TASK_CONFIGS[task]

        dataset_dirs.append(task_config['dataset_dir'])
        num_episodes_list.append(task_config['num_episodes'])
        max_episode_len = max(max_episode_len, task_config['episode_len'])
        camera_names = task_config['camera_names']

    max_skill_len = args['max_skill_len'] if args['max_skill_len'] is not None else max_episode_len

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': args['image_encoder'],
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'multi_gpu': multi_gpu,
                         }
    
    elif policy_class == 'Diffusion':
        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 14,
                         'observation_horizon': 1,
                         'action_horizon': 8, # TODO not used
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         'backbone': args['image_encoder'],
                         'multi_gpu': multi_gpu,
                         'is_eval': is_eval,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : args.image_encoder, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': max_episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'log_wandb': log_wandb,
        'use_language': use_language,
        'language_encoder': language_encoder,
        'max_skill_len': max_skill_len,
        'use_one_hot': use_one_hot,
        'instructor_path': instructor_path,
        'history_len': history_len,
        'history_skip_frame': history_skip_frame,
        'hl_margin': hl_margin,
    }

    if is_eval:
        print(f"{CKPT=}")
        ckpt_names = [f'policy_last.ckpt'] if CKPT == 0 else [f'policy_epoch_{CKPT}_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, dataset_dirs=dataset_dirs)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, stats, _ = load_merged_data(dataset_dirs, num_episodes_list, camera_names, batch_size_train, max_len=max_skill_len, 
                                                  command_list=commands, use_language=use_language, language_encoder=language_encoder, 
                                                  use_one_hot=use_one_hot, policy_class=policy_class) # , dagger_ratio=0.4
    
    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    train_bc(train_dataloader, config)


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)


def make_fixed_lr_scheduler(optimizer):
    return LambdaLR(optimizer, lambda epoch: 1.0)


def make_scheduler(optimizer, num_steps):
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_steps//100, num_training_steps=num_steps)
    # scheduler = make_fixed_lr_scheduler(optimizer)
    return scheduler


def get_image(ts, camera_names, crop_top=True, save_dir=None, t=None):
    curr_images = []
    for cam_name in camera_names:
        curr_image = ts.observation['images'][cam_name]
        
        # Check for 'cam_high' and apply transformation
        if crop_top and cam_name == 'cam_high':
            curr_image = crop_resize(curr_image)

        # Swap BGR to RGB
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

        curr_image = rearrange(curr_image, 'h w c -> c h w')
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    # Center crop and resize
    original_size = curr_image.shape[-2:]
    ratio = 0.95
    curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                    int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
    curr_image = curr_image.squeeze(0)
    resize_transform = transforms.Resize(original_size, antialias=True)
    curr_image = resize_transform(curr_image)
    curr_image = curr_image.unsqueeze(0)
    
    if save_dir is not None:
        # Convert torch tensors back to numpy and concatenate for visualization
        concat_images = [rearrange(img.cpu().numpy(), 'c h w -> h w c') for img in curr_image.squeeze(0)]
        concat_image = np.concatenate(concat_images, axis=1)
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
        img_name = 'init_visualize.png' if t is None else f'gpt/{t=}.png' # save image every query_frequency for ChatGPT
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, (concat_image * 255).astype(np.uint8))

    return curr_image

def eval_bc(config, ckpt_name, save_episode=True, dataset_dirs=None):
    ### DAgger ###
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    global option
    option = 0
    dagger = False
    language_dagger = False

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    log_wandb = config['log_wandb']
    use_language = config['use_language']
    language_encoder = config['language_encoder']
    max_skill_len = config['max_skill_len']
    use_one_hot = config['use_one_hot']
    instructor_path = config['instructor_path']
    history_len = config['history_len']
    history_skip_frame = config['history_skip_frame']
    hl_margin = config['hl_margin']
    print(f'{hl_margin=}')
    use_instructor = instructor_path is not None

    if use_language:
        tokenizer, model = initialize_model_and_tokenizer(language_encoder)
        assert tokenizer is not None and model is not None

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    model_state_dict = torch.load(ckpt_path)['model_state_dict']
    if is_multi_gpu_checkpoint(model_state_dict):
        print("The checkpoint was trained on multiple GPUs.")
        model_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict.items()}
    loading_status = policy.deserialize(model_state_dict)
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_pro.aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True, only_right=ONLY_RIGHT)
        env_max_reward = 0
        master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                        robot_name=f'master_left', init_node=False)
        master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                        robot_name=f'master_right', init_node=False)
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 25
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []

    n_existing_rollouts = len([f for f in os.listdir(ckpt_dir) if f.startswith('video')]) if save_episode else 0
    print(f'{n_existing_rollouts=}')

    # create dataset for language_dagger
    dataset_dir = dataset_dirs[-1]
    dataset_dir_language_dagger = dataset_dir + '_language_dagger_v3'
    if not os.path.isdir(dataset_dir_language_dagger):
        os.makedirs(dataset_dir_language_dagger)
    print(f"\nRecording Language DAgger dataset to {dataset_dir_language_dagger}")

    if DAGGER:
        dataset_dir_dagger = dataset_dir + '_dagger'
        if not os.path.isdir(dataset_dir_dagger):
            os.makedirs(dataset_dir_dagger)
        print(f"\nRecording DAgger dataset to {dataset_dir_dagger}")

    # set up the instructor
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

    # Run the background listening in a separate thread
    if AUDIO:
        # stop_listener_thread = threading.Thread(target=background_listening)
        # stop_listener_thread.daemon = True  # This will allow the program to exit even if the thread is running
        # stop_listener_thread.start()
        listener_subscriber = rospy.Subscriber(
            "/audio_transcription",
            String,
            transcribe_from_ros,
            queue_size=1,
        )

    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()
        if AUDIO:
            print("Please run the ros transcriber now!")
            time.sleep(20)
        ts.observation['option'] = option
        if use_instructor:
            history_obs.clear()
            history_ts.clear()
            history_ts.append(ts)
            history_acs.clear()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        command_list = []
        command_embedding = None
        continue_control = False
        t = 0
        prev_t = 0
        # fixed_command_list = ['pick up the bag', 'pick up the sharpie', 'put the sharpie into the bag', 'release the sharpie', 'pick up the tape', 'put the tape into the bag', 'release the tape', 'pick up the sponge', 'put the sponge into the bag', 'release the sponge', 'release the bag']
        with torch.inference_mode():
            while t < max_timesteps:               
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos               

                ### query policy
                if config['policy_class'] in ["ACT", "Diffusion"]:
                    if t % query_frequency == 0:
                        curr_image = get_image(ts, camera_names, save_dir=ckpt_dir if t == 0 else None)    
                        # put the curr_image to the end of the deque
                        if use_instructor:
                            history_obs.append(curr_image)

                        if use_language or use_one_hot:
                            # Check if an intervention is needed; if so, language dagger
                            if use_instructor and (intervention_needed.is_set() or option == 2):
                                language_dagger = True
                                print(f"##### Intervention needed at {t=}. Please provide an instruction: #####")
                                last_command = command
                                intervention_needed.set()
                                command = get_user_command()
                                # Reset the intervention flag after handling the input
                                intervention_needed.clear()

                                command_embedding = generate_command_embedding(command, t, language_encoder, tokenizer, model, use_one_hot=use_one_hot)

                                # Initialize the segment data with previous observations of length hl_margin
                                ts_segment = list(history_ts)
                                ts_segment[0].observation['option'] = -1
                                segment_data = {'ts': ts_segment, 'actions': list(history_acs), 'command': command}  # Initialize segment data

                                dataset_path_language_dagger, episode_idx_language_dagger = create_dataset_path(dataset_dir_language_dagger)
                                # save an image of the current timestep, with predicted_instruction and command overlaid
                                visualize_language_dagger(curr_image, last_command, command, dataset_dir_language_dagger, episode_idx_language_dagger) 
                                
                            elif t % max_skill_len == 0 and not language_dagger:
                                if t < 150: # deterministic
                                    command = "pick up the bag"
                                # if t < 150: # deterministic
                                #     command = "pick up the plate"
                                elif use_instructor:
                                    last_command = command
                                    command = predict_instruction(instructor, history_obs, history_skip_frame, query_frequency)
                                    # hack: ask the user to confirm if it should release the bag
                                    if last_command != "release the bag" and command == "release the bag":
                                        do_not_release = input("Release the bag? (y/n): ") != "y"
                                        if do_not_release:
                                            command = input("What do you want? ") 
                                            intervention_needed.set()                                                        
                                else:
                                    intervention_needed.set()
                                    command = get_user_command()
                                    intervention_needed.clear()
                                    # try:
                                    #     command = fixed_command_list.pop(0)
                                    # except:
                                    #     command_list.append('')
                                    #     break
                                command_embedding = generate_command_embedding(command, t, language_encoder, tokenizer, model, use_one_hot=use_one_hot)
                            
                            assert command_embedding is not None
   
                        all_actions = policy(qpos, curr_image, command_embedding=command_embedding)
                    elif use_instructor and t % history_skip_frame == 0:
                        curr_image = get_image(ts, camera_names)   
                        history_obs.append(curr_image)
  
                    if use_language or use_one_hot:
                        prefix = "user" if language_dagger else "prediction"
                        command_list.append(f'{prefix}: {command}')
                        # command_list.append(f'{command}')

                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
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
                ts.observation['option'] = option if not language_dagger else 4
                if use_instructor:
                    history_ts.append(ts)
                    history_acs.append(action)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

                # Check if language_dagger mode is active
                if language_dagger:
                    # Append data to the segment_data
                    segment_data['ts'].append(ts)
                    segment_data['actions'].append(action)
                    
                    # Check if segment is complete
                    if len(segment_data['actions']) >= hl_margin + max_skill_len - 1:
                        # Save the segment
                        save_trajectory_thread = threading.Thread(target=save_trajectory, args=(dataset_path_language_dagger, segment_data['ts'], segment_data['actions'], camera_names, segment_data['command'], None))
                        save_trajectory_thread.start()
                        language_dagger = False
                        continue_control = False

                # check if the 3rd pedal is pressed
                if DAGGER and option == 3:
                    dagger = True

                # early termination
                if option == 5:
                    break

                if dagger:
                    command = get_user_command()
                    # sync the master arms position from puppet arms position
                    sync_puppet_to_master(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

                    # after 2 sec, the puppet arms start to follow the master arms
                    time.sleep(2)
                    prev_len = len(image_list)
                    ts, image_list = teleop(env, master_bot_left, master_bot_right, dataset_dir_dagger, ts, camera_names, image_list, command)
                    
                    # repeat the command for the teleop_t timesteps to align with the image_list
                    new_len = len(image_list)
                    teleop_t = new_len - prev_len
                    for _ in range(teleop_t):
                        command_list.append(f"dagger: {command}")

                    # the 3rd pedal is released, torque on both master bots
                    torque_on(master_bot_left)
                    torque_on(master_bot_right)

                    # continue the policy execution
                    dagger = False

                    # increase t so that the policy will be queried in the next timestep
                    remainder = t % query_frequency
                    if remainder:
                        t += query_frequency - remainder

                    # reset the action buffer
                    if temporal_agg:
                        all_time_actions.fill_(0)
                else:
                    t += 1

            plt.close()
        if real_robot:
            if ONLY_RIGHT:
                move_grippers([env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN], move_time=0.5)  # open
            else:
                move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        if log_wandb:
            wandb.log({"test/episode_return": episode_return, "test/episode_highest_reward": episode_highest_reward, "test/env_max_reward": env_max_reward, "test/success": episode_highest_reward==env_max_reward}, step=rollout_id)

        if save_episode:
            is_language_dagger = 'ld' in instructor_path.split('/')[-2] if use_instructor else False
            postfix = f'_ld' if is_language_dagger else ''
            video_name = f'video{rollout_id+n_existing_rollouts}{postfix}.mp4'
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, video_name), cam_names=camera_names, command_list=command_list)
            if log_wandb:
                wandb.log({"test/video": wandb.Video(os.path.join(ckpt_dir, f'video{rollout_id}.mp4'), fps=50, format="mp4")}, step=rollout_id)

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    if log_wandb:
        wandb.log({"test/success_rate": success_rate, "test/avg_return": avg_return})

    listener.stop()

    return success_rate, avg_return


def forward_pass(data, policy):
    if len(data) == 5: # use_language or use_one_hot
        image_data, qpos_data, action_data, is_pad, command_embedding = data
        command_embedding = command_embedding.cuda()
    else:
        image_data, qpos_data, action_data, is_pad = data
        command_embedding = None
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()    
    return policy(qpos_data, image_data, action_data, is_pad, command_embedding)


def train_bc(train_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    log_wandb = config['log_wandb']
    multi_gpu = config['policy_config']['multi_gpu']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    optimizer = make_optimizer(policy_class, policy)
    scheduler = make_scheduler(optimizer, num_epochs)

    # if ckpt_dir is not empty, prompt the user to load the checkpoint
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 2:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input() if not CLUSTER else "y"
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
            model_state_dict = checkpoint['model_state_dict']
            # The model was trained on a single gpu, now load onto multiple gpus
            if multi_gpu and not is_multi_gpu_checkpoint(model_state_dict):
                # Add "module." prefix only to the keys associated with policy.model
                model_state_dict = {k if "model" not in k else f"model.module.{k.split('.', 1)[1]}": v for k, v in model_state_dict.items()}
            # The model was trained on multiple gpus, now load onto a single gpu
            elif not multi_gpu and is_multi_gpu_checkpoint(model_state_dict):
                # Remove "module." prefix only to the keys associated with policy.model
                model_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict.items()}
            loading_status = policy.deserialize(model_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(loading_status)
        else:
            print("Not loading checkpoint")
            start_epoch = 0
    else:
        start_epoch = 0

    policy.cuda()

    train_history = []
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        scheduler.step()
        e = epoch - start_epoch
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*e:(batch_idx+1)*(e+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        epoch_summary['lr'] = np.array(scheduler.get_last_lr()[0])
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.5f} '
        print(summary_string)
        if log_wandb:
            epoch_summary_train = {f'train/{k}': v for k, v in epoch_summary.items()}
            wandb.log(epoch_summary_train, step=epoch)

        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save({
                'model_state_dict': policy.serialize(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }, ckpt_path)

            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of 1000 epochs
            prune_epoch = epoch - save_ckpt_every
            if prune_epoch % 1000 != 0:
                prune_path = os.path.join(ckpt_dir, f'policy_epoch_{prune_epoch}_seed_{seed}.ckpt')
                if os.path.exists(prune_path):
                    os.remove(prune_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save({
        'model_state_dict': policy.serialize(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, ckpt_path)


if __name__ == '__main__':
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

    # language dagger
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--command', action='store', type=str, help='comma-separated list of commands', default='', required=False)
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0, required=False)
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--language_encoder', action='store', type=str, choices=['distilbert', 'clip'], default='distilbert', help='Type of language encoder to use: distilbert or clip', required=False)
    parser.add_argument('--max_skill_len', action='store', type=int, help='max_skill_len', required=False)
    parser.add_argument('--use_one_hot', action='store_true')
    parser.add_argument("--image_encoder", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'resnet18film', 'resnet34film', 'resnet50film','efficientnet_b0film', 'efficientnet_b3film', 'efficientnet_b5film'], help="Which image encoder to use for the BC policy.")
    parser.add_argument('--low_res', action='store', type=int, help='lower resolution by a factor', required=False, default=1)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--instructor_path', action='store', type=str, help='instructor_path', required=False)
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=2)
    parser.add_argument('--history_skip_frame', action='store', type=int, help='history_skip_frame', default=50)
    parser.add_argument('--hl_margin', action='store', type=int, help='the number of timesteps to record before and after language dagger', default=100)

    main(vars(parser.parse_args()))
