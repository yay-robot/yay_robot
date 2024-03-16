'''
Separate data collection and processing into different threads. Record audio. Listen to keyboard/pedal inputs.

Example usage: 
$ python3 record_episodes.py --task_name aloha_task --num_episodes 3
'''

import os
import time
import h5py
import cv2
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import queue
import threading
import sounddevice as sd
import wavio
from pynput import keyboard
import signal
import sys
sys.path.append('/home/lucyshi/code/language-dagger/src') # to import aloha
sys.path.append('/iris/u/lucyshi/language-dagger/src') # for cluster
sys.path.append('/home/huzheyuan/Desktop/language-dagger/src') # for zheyuan
from constants import DT, START_ARM_POSE, TASK_CONFIGS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from aloha_pro.aloha_scripts.utils import memory_monitor

import IPython
e = IPython.embed

DATA_COLLECTION_DONE = False
MAX_THREADS = 20
AUDIO = True # Temp flag, disable audio for debugging
EXIT_FLAG = False
option = 0
ONLY_RIGHT = False

def signal_handler(sig, frame):
    global EXIT_FLAG
    print("Keyboard interrupt detected. Waiting for data processing tasks to complete...")
    EXIT_FLAG = True

def on_press(key):
    global option
    if hasattr(key, 'char') and key.char in ['1', '2', '3']:
        option = int(key.char)
    else:
        option = 0

def on_release(key):
    global option
    if hasattr(key, 'char') and key.char in ['1', '2', '3']:
        option = 0

def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -1.4 #-0.3
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT/10)
    if not ONLY_RIGHT:
        torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')


def data_collection(dt, max_timesteps, master_bot_left, master_bot_right, env, episode_queue, dataset_dir, dataset_name):
    global DATA_COLLECTION_DONE
    global option
    option = 0 
    prev_option = -1  # To store the previous option
    current_option = 0

    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)
    
    ts = env.reset(fake=True)
    ts.observation['option'] = current_option
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    
    # Start recording audio
    if AUDIO:
        # print(sd.query_devices())
        # import ipdb; ipdb.set_trace()
        sd.default.device = 'USB PnP Audio Device'
        video_duration = max_timesteps * dt  # Total duration of the episode.
        audio_duration = video_duration + 140  # add time to account for latency
        audio_sampling_rate = 48000  # Standard sampling rate for this device
        audio_recording = sd.rec(int(audio_duration * audio_sampling_rate), samplerate=audio_sampling_rate, channels=1)
        audio_start = time.time()
    
    t = 0
    pbar = tqdm(total=max_timesteps)
    while t < max_timesteps:
        t0 = time.time()
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time()
        ts = env.step(action)

        # If the current option is 1 or 2, update current_option
        if option in [1, 2]:
            current_option = option
        ts.observation['option'] = current_option
        time.sleep(max(0, DT - (time.time() - t0)))

        # Check for changes in option and apply torque accordingly
        if option == 0 and (prev_option == 1 or prev_option == 2):
            torque_off(master_bot_left)
            torque_off(master_bot_right)
            ts.observation['option'] = -1 # a new segment
        # If the key 1 or 2 is pressed, pause the robot
        elif (option == 1 or option == 2) and prev_option == 0:
            torque_on(master_bot_left)
            torque_on(master_bot_right)

        prev_option = option  # Update previous option for next iteration
           
        # If the option is 1 or 2, skip the recording of the current timestep
        if option not in [1, 2]:
            t2 = time.time()
            timesteps.append(ts)
            actions.append(action)
            actual_dt_history.append([t0, t1, t2])
            episode_queue.put((ts, action))  # Push data into the queue

            t += 1
            pbar.update(1)

        if option == 3:
            break

    pbar.close()

    # stop recording
    if AUDIO:
        sd.stop()
        audio_end = time.time()

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    # Open puppet grippers
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        print(f'Warning: {freq_mean=}')

    # directly save audio recording in place, since it's fast
    if AUDIO:
        t0 = time.time()
        recorded_frames = int((audio_end - audio_start) * audio_sampling_rate)
        audio_recording = audio_recording[:recorded_frames]  # only save the portion that contains recorded audio

        audio_recording_int16 = (audio_recording * (2 ** 15 - 1)).astype(np.int16)
        wavio.write(os.path.join(dataset_dir, dataset_name + ".wav"), audio_recording_int16, audio_sampling_rate, sampwidth=2)
        print(f'Saving {dataset_name} audio: {time.time() - t0:.1f} secs')
        # stop recording # TODO: test
        # sd.stop()

    DATA_COLLECTION_DONE = True



def data_processing(max_timesteps, camera_names, dataset_path, episode_queue, collect_done_event):
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

    global DATA_COLLECTION_DONE

    dataset_name = dataset_path.split('/')[-1]
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/observations/option': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        # data_dict[f'/observations/images/{cam_name}_depth'] = []

    while not collect_done_event.is_set() or not episode_queue.empty():
        try:
            ts, action = episode_queue.get(timeout=2)  # This will block until there's data in the queue
        except queue.Empty:
            continue

        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        option_expanded = np.expand_dims(np.array(ts.observation['option']), axis=0)
        data_dict['/observations/option'].append(option_expanded)
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            # data_dict[f'/observations/images/{cam_name}_depth'].append(ts.observation['images'][f'{cam_name}_depth'][..., None])  # insert redundant channel to match RGB shape


    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'{dataset_name} compression: {time.time() - t0:.2f}s')

        # pad so it has the same length
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
        print(f'{dataset_name} padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
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
            # _ = image.create_dataset(f'{cam_name}_depth', (max_timesteps, 480, 640, 1), dtype='uint16',
            #                     chunks=(1, 480, 640, 1), )
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

    print(f'{dataset_name} saving: {time.time() - t0:.1f} secs')


def threaded_data_processing(semaphore, max_timesteps, camera_names, dataset_path, episode_queue, collect_done_event):
    # Function to handle data processing in a thread and release the semaphore when done
    data_processing(max_timesteps, camera_names, dataset_path, episode_queue, collect_done_event)
    semaphore.release()


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    global DATA_COLLECTION_DONE

    # Resetting this flag for each episode
    DATA_COLLECTION_DONE = False

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    print(f'Dataset name: {dataset_name}')
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # Start the data collection thread
    episode_queue = queue.Queue()
    data_collection_thread = threading.Thread(target=data_collection, args=(dt, max_timesteps, master_bot_left, master_bot_right, env, episode_queue, dataset_dir, dataset_name))

    data_collection_thread.start()

    # At this point, the data has been collected and is ready for processing. 
    # The data_processing task has been offloaded to the ThreadPoolExecutor in the main function.
    return episode_queue, data_collection_thread


def main(args):
    # Start the memory monitor thread
    threading.Thread(target=memory_monitor, daemon=True).start()

    signal.signal(signal.SIGINT, signal_handler)
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release, daemon=True)
    listener.start()

    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    num_episodes = args['num_episodes']

    # Get the starting episode index
    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)

    # Semaphore to ensure we don't exceed MAX_THREADS
    semaphore = threading.Semaphore(MAX_THREADS - 1)
    processing_threads = []

    for _ in range(num_episodes):
        if EXIT_FLAG:
            break

        dataset_name = f'episode_{episode_idx}'
        print(dataset_name + '\n')

        # Capture the episode's data and get its episode-specific queue
        episode_queue, data_collection_thread = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite=True)

        # Acquire semaphore before starting a new thread
        semaphore.acquire()

        # Start data processing for the current episode in a separate thread
        dataset_path = os.path.join(dataset_dir, dataset_name)
        
        # create a done collecting data signal
        collect_done_event = threading.Event()
        thread = threading.Thread(target=threaded_data_processing, args=(semaphore, max_timesteps, camera_names, dataset_path, episode_queue, collect_done_event))
        thread.start()
        processing_threads.append(thread)

        # Wait for the data collection for that episode to complete.
        data_collection_thread.join()
        collect_done_event.set()

        # Increment the episode index for the next episode
        episode_idx += 1

    # Wait for all processing tasks to complete
    for t in processing_threads:
        print(f"Processing {t.name}...")
        t.join()

    print("All data processing tasks completed.")
    listener.stop()


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Starting episode index.', default=None, required=False)
    parser.add_argument('--num_episodes', action='store', type=int, help='Number of episodes to record.', default=1, required=False)
    main(vars(parser.parse_args()))
    # debug()
