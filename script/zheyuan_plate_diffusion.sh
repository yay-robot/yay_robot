python act/imitate_episodes.py \
    --task_name aloha_plate_sponge \
    --ckpt_dir /nfs/kun2/users/huzheyuan/language_ckpt/aloha_plate_clean_diffusion_bs16 \
    --policy_class Diffusion --kl_weight 80 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
    --use_language --language_encoder distilbert --max_skill_len 200 --num_epochs 20000  --lr 1e-4 \
    --image_encoder efficientnet_b3film --seed 0 --log_wandb --eval
