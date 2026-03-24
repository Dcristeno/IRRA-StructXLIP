## IRRA + Part-Aware Attribute Alignment

This branch adds a lightweight part-aware attribute alignment path.

Design:
- Text side extracts `upper / lower / shoes` phrases from the original caption.
- Image side reuses ViT patch tokens from the original RGB image and pools vertical regions into `upper / lower / shoes`.
- Each part pair is aligned with a lightweight SDM loss during training.

Quick screening:

```bash
python train_part_attr.py \
  --name part-attr-20ep \
  --dataset_name CUHK-PEDES \
  --root_dir /root/autodl-tmp \
  --batch_size 16 \
  --test_batch_size 64 \
  --num_workers 4 \
  --val_num_workers 0 \
  --num_epoch 20 \
  --eval_period 5 \
  --use_swanlab
```
