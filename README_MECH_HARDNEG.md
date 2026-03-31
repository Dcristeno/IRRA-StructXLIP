## IRRA + Mechanism-Guided Hard Negative Learning

This branch adds a training-only module inspired by the mechanism analysis tools.

Design:
- keep the original IRRA backbone and retrieval head
- find the highest-scoring in-batch hard negative for each text query
- estimate token evidence coverage over image patch tokens
- apply an aggressive extra ranking loss when a negative has high score and is dominated by a few strong local cues

The intent is high-risk, high-reward: explicitly push down false positives that look locally similar but are globally inconsistent.

Quick screening:

```bash
python train_mech_hardneg.py \
  --name mech-hardneg-b64 \
  --dataset_name CUHK-PEDES \
  --root_dir /root/autodl-tmp \
  --batch_size 64 \
  --test_batch_size 256 \
  --num_workers 8 \
  --num_epoch 20 \
  --eval_period 5 \
  --MLM \
  --dehn_loss_weight 2.0 \
  --dehn_score_weight 4.0 \
  --dehn_dominance_weight 5.0 \
  --dehn_cov_weight 2.5 \
  --use_swanlab
```

Recommended log targets:
- `dehn_loss`
- `dehn_rank_term`
- `dehn_cov_term`
- `dehn_pos_sim` vs `dehn_neg_sim`
- final `R1`
