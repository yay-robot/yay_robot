### Task parameters
DATA_DIR = '/scr/lucyshi/dataset'
# DATA_DIR = '/data2/data_collection'
# DATA_DIR = '/iris/u/lucyshi/language-dagger/data/act' # for cluster
TASK_CONFIGS = {
    'aloha_test':{
        'dataset_dir': DATA_DIR + '/aloha_test',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects',
        'num_episodes': 199,
        'episode_len': 2000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_pro_pencil':{
        'dataset_dir': DATA_DIR + '/aloha_pro_pencil_compressed',
        'num_episodes': 25,
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_reach_sharpie_sponge':{
        'dataset_dir': DATA_DIR + '/aloha_reach_sharpie_sponge',
        'num_episodes': 30,
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_simplified_pick_sharpie_sponge':{
        'dataset_dir': DATA_DIR + '/aloha_simplified_pick_sharpie_sponge',
        'num_episodes': 40,
        'episode_len': 300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_random_pick_sharpie_sponge':{
        'dataset_dir': DATA_DIR + '/aloha_random_pick_sharpie_sponge',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_dagger': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_dagger',
        'num_episodes': 75,
        'episode_len': 200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_ethan': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_ethan',
        'num_episodes': 100,
        'episode_len': 2500,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_ethan_v0': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_ethan_v0',
        'num_episodes': 16,
        'episode_len': 2400,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_ethan_v1': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_ethan_v1',
        'num_episodes': 48,
        'episode_len': 2200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_ethan_v2': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_ethan_v2',
        'num_episodes': 278,
        'episode_len': 2800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_ethan_v3': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_ethan_v3',
        'num_episodes': 34,
        'episode_len': 3800,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_tony_test': {
        'dataset_dir': DATA_DIR + '/aloha_tony_test',
        'num_episodes': 1,
        'episode_len': 10,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_johnny_v0': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_johnny_v0',
        'num_episodes': 50,
        'episode_len': 3000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_johnny_v1': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_johnny_v1',
        'num_episodes': 478,
        'episode_len': 8000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_johnny_v1_language_dagger': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_johnny_v1_language_dagger',
        'num_episodes': 586, 
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_johnny_v1_language_dagger_v1': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_johnny_v1_language_dagger_v1',
        'num_episodes': 55, 
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_johnny_v1_language_dagger_v2': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_johnny_v1_language_dagger_v2',
        'num_episodes': 69, 
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_johnny_v1_language_dagger_v3': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_johnny_v1_language_dagger_v3',
        'num_episodes': 352, 
        'episode_len': 350,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_bag_3_objects_johnny_v1_sponge': {
        'dataset_dir': DATA_DIR + '/aloha_bag_3_objects_johnny_v1_sponge',
        'num_episodes': 151,
        'episode_len': 2500,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_screwdriver': {
        'dataset_dir': DATA_DIR + '/aloha_screwdriver',
        'num_episodes': 30, 
        'episode_len': 400,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_tshirt': {
        'dataset_dir': DATA_DIR + '/aloha_tshirt',
        'num_episodes': 30, 
        'episode_len': 1300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_plate_sponge': {
        'dataset_dir': DATA_DIR + '/aloha_plate_sponge',
        'num_episodes': 165,
        'episode_len': 1200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_plate_sponge_v1': {
        'dataset_dir': DATA_DIR + '/aloha_plate_sponge_v1',
        'num_episodes': 100,
        'episode_len': 1200,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_plate_sponge_dagger': {
        'dataset_dir': DATA_DIR + '/aloha_plate_sponge_language_dagger',
        'num_episodes': 94,
        'episode_len': 249,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_trail_mix': {
        'dataset_dir': DATA_DIR + '/aloha_trail_mix',
        'num_episodes': 319,
        'episode_len': 6000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_trail_mix_language_dagger_v0': {
        'dataset_dir': DATA_DIR + '/aloha_trail_mix_language_dagger_v0',
        'num_episodes': 31, 
        'episode_len': 250,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_trail_mix_language_dagger_v1': {
        'dataset_dir': DATA_DIR + '/aloha_trail_mix_language_dagger_v1',
        'num_episodes': 60, 
        'episode_len': 250,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_trail_mix_language_dagger_v2': {
        'dataset_dir': DATA_DIR + '/aloha_trail_mix_language_dagger_v2',
        'num_episodes': 56, 
        'episode_len': 250,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_trail_mix_language_dagger_v3': {
        'dataset_dir': DATA_DIR + '/aloha_trail_mix_language_dagger_v3',
        'num_episodes': 73, 
        'episode_len': 250,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_trail_mix_ethan_v0': {
        'dataset_dir': DATA_DIR + '/aloha_trail_mix_ethan_v0',
        'num_episodes': 75,
        'episode_len': 4000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_trail_mix_johnny_v1': {
        'dataset_dir': DATA_DIR + '/aloha_trail_mix_johnny_v1',
        'num_episodes': 81,
        'episode_len': 5000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8 #0.3083
MASTER_GRIPPER_JOINT_CLOSE = -1.65 #-0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
