export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_multi_vector.py \
  --name mv-local-main-b64 \
  --dataset_name CUHK-PEDES \
  --root_dir /root/autodl-tmp \
  --batch_size 64 \
  --test_batch_size 256 \
  --num_workers 8 \
  --num_epoch 60 \
  --lr 1e-5 \
  --MLM \
  --eval_global_score_weight 0.3 \
  --eval_local_score_weight 1.0 \
  --use_swanlab
