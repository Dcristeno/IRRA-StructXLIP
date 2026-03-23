## IRRA + Color Auxiliary Head

This branch adds a lightweight color-aware auxiliary head for text-to-image person retrieval.

Main entry:

```bash
python train_color.py \
  --name color-aux \
  --dataset_name CUHK-PEDES \
  --root_dir /root/autodl-tmp \
  --batch_size 16 \
  --test_batch_size 64 \
  --num_workers 4 \
  --val_num_workers 0 \
  --use_swanlab
```

Recommended quick-screening setup:

```bash
python train_color.py \
  --name color-aux-20ep \
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

Color labels are extracted from captions with a rule-based lexicon in `misc/color_lexicon.yaml`.
