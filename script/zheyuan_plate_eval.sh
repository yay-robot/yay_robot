python src/act/imitate_episodes.py \
    --task_name aloha_plate_sponge \
    --ckpt_dir  /nfs/kun2/users/huzheyuan/language_ckpt/aloha_plate_sponge_164_add_v1_100_20k \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 20000  --lr 1e-4 \
    --use_language --language_encoder distilbert --max_skill_len 200 \
    --seed 0 --temporal_agg --log_wandb --gpu=0 --eval \
    --instructor_path /nfs/kun2/users/huzheyuan/language_ckpt/instructor_aloha_plate_sponge_168_v0/epoch_1900.ckpt \
    --history_len 1 --history_skip_frame 50

# python src/act/imitate_episodes.py \
#     --task_name aloha_plate_sponge \
#     --ckpt_dir  /nfs/kun2/users/huzheyuan/language_ckpt/aloha_plate_sponge_164_20k \
#     --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
#     --num_epochs 20000  --lr 1e-4 \
#     --use_language --language_encoder distilbert --max_skill_len 250 \
#     --seed 0 --temporal_agg --log_wandb --gpu=0 --eval \
#     --instructor_path /nfs/kun2/users/huzheyuan/language_ckpt/instructor_aloha_plate_sponge_168_v0/epoch_1900.ckpt \
#     --history_len 1 --history_skip_frame 50 

# python src/act/imitate_episodes.py \
#     --task_name aloha_plate_sponge \
#     --ckpt_dir  /nfs/kun2/users/huzheyuan/language_ckpt/aloha_plate_sponge_164_20k \
#     --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
#     --num_epochs 20000  --lr 1e-4 \
#     --use_language --language_encoder distilbert --max_skill_len 200 \
#     --instructor_path /nfs/kun2/users/huzheyuan/language_ckpt/instructor_aloha_plate_sponge_dagger_v0/epoch_1900.ckpt \
#     --history_len 1 --history_skip_frame 50 \
#     --seed 0 --log_wandb --gpu=0 --eval
