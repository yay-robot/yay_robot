python src/instructor/train.py \
    --task_name aloha_plate_sponge aloha_plate_sponge_dagger \
    --ckpt_dir /nfs/kun2/users/huzheyuan/language_ckpt/instructor_aloha_plate_sponge_dagger_v0 \
    --batch_size 256 --num_epochs 2000  --lr 1e-4 --random_crop \
    --history_skip_frame 50 --prediction_offset 20 --history_len 1 --seed 0 --log_wandb
