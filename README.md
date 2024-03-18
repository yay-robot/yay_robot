# Yell At Your Robot (YAY Robot)

[[Project website](https://yay-robot.github.io/)] [[Paper]()]


Yell At Your Robot (YAY Robot) leverages verbal corrections to improve robot performance on complex long-horizon tasks. It can incorporate language corrections in real-time and for continuous improvement.

![](assets/teaser.png)

If you encountered any issue, feel free to contact lucyshi (at) stanford (dot) edu

## Installation
1. Clone this repository
```bash
git clone git@github.com:yay-robot/yay_robot.git
cd yay_robot
```

2. Create a virtual environment
```bash 
conda create -n yay python=3.8.10 # requirement for aloha
conda activate yay
```

3. Install [Whisper](https://github.com/openai/whisper)
```bash
sudo apt update && sudo apt install ffmpeg
```

4. Install packages
```bash
pip install -e .
```

5. (Optional) Install package for audio recording
```bash
sudo apt install portaudio19-dev python3-pyaudio
```

## Aloha
Please follow the [Aloha](https://github.com/tonyzhaozh/aloha) installation guide to install the hardware and software (ROS). For future convenience, you may add the following aliases to your `~/.bashrc` file. 
```bash
alias ros-init='conda deactivate && source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh && cd ~/interbotix_ws/'
alias cd-lc='cd $PATH_TO_YAY_ROBOT/src'
alias cd-ps='cd $PATH_TO_YAY_ROBOT/src/aloha_pro/aloha_scripts'
alias launch='conda activate yay && cd-ps'
alias launchl='conda activate yay && cd-lc'
```

* Teleoperation
```bash
# ROS terminal
ros-init
roslaunch aloha 4arms_teleop.launch

# Left hand terminal
launch
python3 one_side_teleop.py left

# Right hand terminal
launch
python3 one_side_teleop.py right

# Sleep terminal
launch
python3 sleep.py
```

## Data Collection and Processing
1) **Language annotation:** We collect language-annotated data through live narration. By placing a microphone near the robot, the operator can narrate the skill they are performing in real-time. They first verbalize the intended skill and then teleoperate the robot to perform it. The recorded audio is then transcribed into text using the Whisper model and synchronized with the robot's trajectory. The following code automates this process. Please check out our paper (Section 4) for more details.

2) **Filter mistakes:** We differentiate language instructions from corrections, because trajectory segments preceding corrections are likely erroneous or suboptimal and should be excluded from training.
To implement this distinction, we use foot pedals during data collection. When narrating a new skill (instruction), the operator steps on the instruction pedal ('1'). If a correction is necessary, they step on the correction pedal ('2'). 

We used this [lavalier microphone](https://a.co/d/5yS4eBN) and this [foot pedal](https://a.co/d/ihOs7Mn) for data collection.

The following commands use `aloha_bag` as an example task. Feel free to replace it with your own task name and put its specs in [constants.py](src/aloha_pro/aloha_scripts/constants.py).

* Data collection
```bash
# Collect mutliple episodes 
launchl
python3 aloha_pro/aloha_scripts/record_episodes.py --task_name aloha_bag --num_episodes 3
```

* Verification
```bash
# (Optional) Visualize episode
python3 aloha_pro/aloha_scripts/visualize_episodes.py --dataset_dir $PATH_TO_DATASET/aloha_bag --episode_idx 0

# (Optional) Visualize episode with transcription
python3 aloha_pro/aloha_scripts/visualize_episodes_audio.py --dataset_dir $PATH_TO_DATASET/aloha_bag --visualize_option --transcribe --start_episode_idx 0 --end_episode_idx 2

# (Optional) Replay episode
python3 aloha_pro/aloha_scripts/replay_episodes.py --dataset_dir $PATH_TO_DATASET/aloha_bag --episode_idx 0
```

* Data Processing
```bash
# Transcibe all audio files in the dataset_dir
cd ..
python script/transcribe.py --dataset_dir $PATH_TO_DATASET/aloha_bag

# Segment instructions
# If any episode's count measure looks very off (e.g. > 5), you may want to look at the transcribed text of this episode and/or visualize it. This alignment step is very important for data quality.
python script/instruction_segmentation.py --count --dataset_dir $PATH_TO_DATASET/aloha_bag

# Encode instructions and (currently commented out) copy data to cluster
./script/process_data.sh
```

## Training and Evaluation
* Method Diagram
![](assets/method.jpeg)

* Architecture
![](assets/architecture.png)

* Train Low-Level Policy

You may put a list of dataset names after `--task_name`. These datasets will be automatically concatenated.

```bash
launchl
python act/imitate_episodes.py \
    --task_name aloha_bag ... \
    --ckpt_dir $YOUR_CKPT_PATH/ll_ckpt/aloha_bag \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --use_language --language_encoder distilbert --max_skill_len 200 --num_epochs 30000  --lr 1e-4 \
    --image_encoder efficientnet_b3film --seed 0 --log_wandb
```

* Train High-Level Policy
```bash
launchl
python instructor/train.py \
    --task_name aloha_bag ... \
    --ckpt_dir $YOUR_CKPT_PATH/hl_ckpt/aloha_bag \
    --batch_size 64 --num_epochs 15000  --lr 1e-4 \
    --history_skip_frame 50 --prediction_offset 20 --history_len 3 --seed 0 --log_wandb
```

* Deploy

Please put the ckpt number you want to evaluate in [imitate_episodes.py](src/act/imitate_episodes.py).
```bash
launchl
python act/imitate_episodes.py \
    --task_name aloha_bag ... \
    --ckpt_dir $YOUR_CKPT_PATH/ll_ckpt/aloha_bag \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --use_language --language_encoder distilbert --max_skill_len 200 --num_epochs 30000  --lr 1e-4 \
    --image_encoder efficientnet_b3film --seed 0 --log_wandb --eval \
    --history_len 3 --history_skip_frame 50 \
    --instructor_path $YOUR_CKPT_PATH/hl_ckpt/aloha_bag/epoch_14900.ckpt
```

* Language Intervention

The most reliable method is keyboard input: during policy deployment, the operator can press '2' (or step on the second pedal) to freeze the robot, and then type in the correction. We also implemented real-time audio transcription and intervention. The operator can say "stop" / "pardon" / "wait" to pause the robot, and then speak the correction. The audio will be transcribed for the robot to follow. This isn't as reliable as keyboard input when the environment is noisy, or there are heavy-duty jobs running on the same machine. So, we also implemented a walkie-talkie mode, where the machine prioritizes listening to human speech over policy execution when the second pedal (or, get a button!) is pressed.

To run the talking / walkie-talkie mode, please set `AUDIO` in [imitate_episodes.py](src/act/imitate_episodes.py) to `True` and run the following command in a separate window.


```bash
# Enable real time audio
launchl
cd ../script
python real_time_whisper.py --energy_threshold 200 --record_timeout 0.1 --phrase_timeout 0.2
```

The correction data will be automatically saved in `$LAST_DATASET_DIR_language_correction` for post-training.

* Post-Training

Please create a separate directory (eg. `aloha_bag_lc01`) to store the latest ckpt from the high-level policy. For example, copy `$YOUR_CKPT_PATH/hl_ckpt/aloha_bag/epoch_14900.ckpt` to `$YOUR_CKPT_PATH/hl_ckpt/aloha_bag_lc01/epoch_0.ckpt`. Then, run:

```bash
launchl
python instructor/train.py \
    --task_name aloha_bag ... aloha_bag_v2_language_correction \
    --ckpt_dir $YOUR_CKPT_PATH/hl_ckpt/aloha_bag_lc01 \
    --batch_size 64 --num_epochs 5000 --lr 1e-4 --dagger_ratio 0.1\
    --history_skip_frame 50 --prediction_offset 20 --history_len 3 --seed 0 --log_wandb
```

## Citation

If you find our code useful for your research, please cite:
```
TODO
```