# Language-DAgger

## Installation
1. Clone this repository
```bash
git clone git@github.com:lucys0/language-dagger.git
cd language-dagger
```

2. Create a virtual environment
```bash 
conda create -n language-dagger python=3.8.10 # requirement for aloha
conda activate language-dagger
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

## Whisper
```bash 
whisper audio/to_transcribe.wav --output_dir transcription --model_dir whisper_models --language English --model medium
```
To specify the output format, append `--output_format txt`. Run `whisper --help` for more options.

## Aloha-pro
* Put in `~/.bashrc`
```bash
alias ros-init='conda deactivate && source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh && cd ~/interbotix_ws/'
alias cd-ld='cd ~/code/language-dagger/src'
alias cd-ps='cd ~/code/language-dagger/src/aloha_pro/aloha_scripts'
alias launch='conda activate aloha && cd-ps'
alias launchl='conda activate language-dagger && cd-ld'
alias launchd='conda activate diffusion && cd-ld'
```

* Example usage
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

# Real time audio
launchl
cd ../script
python real_time_whisper.py --energy_threshold 200 --record_timeout 0.1 --phrase_timeout 0.2

# Collect mutliple episodes 
## Ethan, trail mix
launchl
python3 aloha_pro/aloha_scripts/record_episodes.py --task_name aloha_trail_mix_ethan_v0 --num_episodes 2
### write down episodes that might have issues
# episode 

## Ethan, bag
launchl
python3 aloha_pro/aloha_scripts/record_episodes.py --task_name aloha_bag_3_objects_ethan_v3 --num_episodes 3
### write down episodes that might have issues
# episode

## Johnny, trail mix
launchl
python3 aloha_pro/aloha_scripts/record_episodes.py --task_name aloha_trail_mix_johnny_v1 --num_episodes 3
### write down episodes that might have issues
# episode 

## Johnny, bag
launchl
python3 aloha_pro/aloha_scripts/record_episodes.py --task_name aloha_bag_3_objects_johnny_v1 --num_episodes 3
### write down episodes that might have issues
# episode 

# Visualize episode
python3 aloha_pro/aloha_scripts/visualize_episodes.py --dataset_dir /scr/lucyshi/dataset/aloha_trail_mix --episode_idx 0

# Visualize episode with transcription
launchl
python3 aloha_pro/aloha_scripts/visualize_episodes_audio.py --dataset_dir /scr/lucyshi/dataset/aloha_trail_mix_ethan_v0 --visualize_option --transcribe --start_episode_idx 0 --end_episode_idx 2

# Replay episode
launchl
python3 aloha_pro/aloha_scripts/replay_episodes.py --dataset_dir /scr/lucyshi/dataset/aloha_trail_mix --episode_idx 0

# Transcibe all audio files in the dataset_dir
python script/transcribe.py --dataset_dir /scr/lucyshi/dataset/aloha_trail_mix_johnny_v1

# Segment instructions
python script/instruction_segmentation.py --count --dataset_dir /scr/lucyshi/dataset/aloha_trail_mix_johnny_v1
python script/instruction_segmentation.py --count --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects_johnny_v1_language_dagger_v3

# Encode instructions and copy data to cluster
./script/process_data.sh

# display
Xvfb :10 -screen 0 1024x768x16 -ac &
export DISPLAY=:10.0
```

## ACT
For sanity check:
```bash
python act/imitate_episodes.py \
    --task_name sim_transfer_cube_human \
    --ckpt_dir data/act_ckpt/sim_transfer_cube_human \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 4000  --lr 1e-5 \
    --seed 0 --log_wandb --temporal_agg



python act/imitate_episodes.py \
    --task_name sim_transfer_cube_human \
    --ckpt_dir data/act_ckpt/sim_transfer_cube_human \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 4000  --lr 1e-5 \
    --seed 0 --log_wandb --temporal_agg
```

For training the bagging task:
```bash
launchl
python act/imitate_episodes.py \
    --task_name aloha_bag_3_objects \
    --ckpt_dir /scr/lucyshi/act_ckpt/aloha_bag_3_objects_bs_16 \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --num_epochs 10000  --lr 2e-5 \
    --use_language --language_encoder distilbert --max_skill_len 200 \
    --seed 0 --log_wandb --temporal_agg --gpu=1
```
To evaluate the policy, run the same command with `--eval`. 

To train a single skill eg. 'pick up the sharpie':
```bash
launchl
python act/imitate_episodes.py \
    --task_name aloha_bag_3_objects \
    --ckpt_dir /scr/lucyshi/act_ckpt/aloha_pick_up_sharpie \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 4000  --lr 1e-5 --max_skill_len 300 --command='pick up the sharpie' \
    --seed 0 --log_wandb --temporal_agg --gpu=0 --eval
```

Chain two skills:
```bash
launchl
python act/imitate_episodes.py \
    --task_name aloha_bag_3_objects \
    --ckpt_dir /scr/lucyshi/act_ckpt/aloha_pick_up_bag_bs_16 \
    --ckpt_dir_2 /scr/lucyshi/act_ckpt/aloha_pick_up_sharpie \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --num_epochs 10000  --lr 5e-5 --max_skill_len 300 \
    --seed 0 --log_wandb --temporal_agg --gpu=0 --eval
```
