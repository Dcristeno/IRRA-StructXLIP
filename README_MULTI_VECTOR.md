# Multi-Vector Late Interaction

This variant intentionally moves away from the original IRRA single-vector retrieval path:

- `1` global vector for each image/text sample
- `8` learned local vectors aggregated from image patch tokens / text tokens
- default training: `id + mlm + multi_vector_loss`
- default validation score: `0.3 * global_sim + 1.0 * local_late_interaction_sim`
- `sdm` is optional instead of default, so this branch is local-dominant rather than baseline-like

Entry point:

```bash
python train_multi_vector.py --name multi-vector-li
```

Recommended first sweep:

```bash
python train_multi_vector.py --name mv-local-main
python train_multi_vector.py --name mv-hybrid --loss_names sdm+id+mlm --multi_vector_loss_weight 0.7 --eval_global_score_weight 0.5 --eval_local_score_weight 0.8
python train_multi_vector.py --name mv-local-only-score --eval_global_score_weight 0.0 --eval_local_score_weight 1.0
```
