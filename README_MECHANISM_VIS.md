# IRRA Mechanism Visualization

This add-on provides a standalone analysis script for visualizing which image patches are most aligned with specific words or phrases in a text query.

## What it does

- loads a trained IRRA checkpoint
- computes normal text-to-image retrieval on the test split
- picks one query caption
- visualizes the top-k retrieved images
- generates token-to-patch heatmaps for chosen terms such as `red`, `backpack`, or `white shoes`

## Main script

```bash
python analysis/visualize_token_patch.py \
  --config_file logs/CUHK-PEDES/your_exp/configs.yaml \
  --query_index 0 \
  --topk 5 \
  --include_gt \
  --focus_terms "red,backpack,white shoes"
```

If `--focus_terms` is omitted, the script will automatically choose a few non-stopword terms from the caption.

## Outputs

By default outputs are saved under:

```text
analysis_outputs/token_patch/<config_name>/query_xxxxx/
```

The folder includes:

- `retrieval_board.png`: the query caption with top-k retrieval results
- `top1_token_patch.png`, `top2_token_patch.png`, ...: per-image token-to-patch activation grids
- `gt-best_token_patch.png`: optional ground-truth image visualization
- `summary.txt`: query text, chosen terms, scores, and image paths

## Notes

- This script currently assumes a ViT-based CLIP backbone because it reads visual patch tokens from the transformer.
- The heatmap is based on token-patch feature similarity, so it is best viewed as a first-step probe rather than a final causal explanation.
