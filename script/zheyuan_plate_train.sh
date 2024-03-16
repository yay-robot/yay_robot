python src/act/imitate_episodes.py \
    --task_name aloha_plate_sponge aloha_plate_sponge_v1\
    --ckpt_dir  /nfs/kun2/users/huzheyuan/language_ckpt/aloha_plate_sponge_164_add_v1_100_20k \
    --policy_class ACT --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
    --num_epochs 20000  --lr 1e-4 \
    --use_language --language_encoder distilbert --max_skill_len 200 \
    --seed 0 --log_wandb --temporal_agg --gpu=0
