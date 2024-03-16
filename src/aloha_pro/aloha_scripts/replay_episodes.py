import os
import h5py
from robot_utils import move_grippers
import argparse
from real_env import make_real_env
from constants import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN
### dagger ###
from pynput import keyboard
from robot_utils import move_arms, torque_on, torque_off, get_arm_joint_positions, get_arm_gripper_positions
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import time
from real_env import get_action

import IPython
e = IPython.embed

STATE_NAMES = JOINT_NAMES + ["gripper", 'left_finger', 'right_finger']

option = 0

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
    move_grippers([master_bot_left, master_bot_right], [puppet_left_gripper, puppet_right_gripper], move_time=0.5)

def teleop(env, master_bot_left, master_bot_right):
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'\nTeleop started')

    # teleop loop
    global option
    while True:
        action = get_action(master_bot_left, master_bot_right)
        env.step(action)
        # stop if the 3rd pedal is released
        if option == 0:
            print("\nReleased Pedal 3")
            break

def main(args):
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]

    global option
    option = 0
    reverse_sync = False

    env = make_real_env(init_node=True)
    env.reset()
    for action in actions:
        env.step(action)

        # check if the 3rd pedal is pressed
        if option == 3:
            reverse_sync = True
            break

    if reverse_sync:
        master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                  robot_name=f'master_left', init_node=False)
        master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                   robot_name=f'master_right', init_node=False)

        # sync the master arms position from puppet arms position
        sync_puppet_to_master(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

        # after 2 sec, the puppet arms start to follow the master arms
        time.sleep(2)
        teleop(env, master_bot_left, master_bot_right)

        # the 3rd pedal is released, torque on both master bots
        torque_on(master_bot_left)
        torque_on(master_bot_right)

        # TODO: continue the policy execution (in imitate_episodes.py)
        print("\nContinuing policy execution")

    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open

    listener.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))


