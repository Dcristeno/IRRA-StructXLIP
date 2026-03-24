export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_part_attr.py \
  --name part-attr-align \
  --dataset_name CUHK-PEDES \
  --root_dir /root/autodl-tmp \
  --batch_size 16 \
  --test_batch_size 64 \
  --num_workers 4 \
  --val_num_workers 0 \
  --use_swanlab
