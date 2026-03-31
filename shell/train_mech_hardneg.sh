export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_mech_hardneg.py \
  --name mech-hardneg-main-b64 \
  --dataset_name CUHK-PEDES \
  --root_dir /root/autodl-tmp \
  --batch_size 64 \
  --test_batch_size 256 \
  --num_workers 8 \
  --num_epoch 60 \
  --lr 1e-5 \
  --MLM \
  --dehn_loss_weight 2.0 \
  --dehn_score_weight 4.0 \
  --dehn_dominance_weight 5.0 \
  --dehn_cov_weight 2.5 \
  --use_swanlab
