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

## V2 script

The second-generation script keeps the same backbone probing logic, but changes the rendering to a sparse red evidence mask and adds a positive-vs-negative comparison board.

```bash
python analysis/visualize_token_patch_v2.py \
  --config_file logs/CUHK-PEDES/your_exp/configs.yaml \
  --checkpoint logs/CUHK-PEDES/your_exp/best.pth \
  --query_index 0 \
  --topk 5 \
  --include_gt \
  --max_auto_terms 5 \
  --hard_negative_count 2 \
  --threshold_percentile 80 \
  --output_dir analysis_outputs/token_patch_v2
```

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

For v2 the folder additionally contains:

- `retrieval_board_v2.png`: retrieval board from the v2 run
- `top1_token_patch_v2.png`, `top2_token_patch_v2.png`, ...: sparse high-response evidence masks
- `comparison_board_v2.png`: best positive and hard negatives in a single aligned comparison grid
- `summary_v2.txt`: comparison rows and threshold settings

## Notes

- This script currently assumes a ViT-based CLIP backbone because it reads visual patch tokens from the transformer.
- The heatmap is based on token-patch feature similarity, so it is best viewed as a first-step probe rather than a final causal explanation.
- V2 is meant to make qualitative diagnosis easier; it still uses similarity-based token-patch attribution, not a full causal intervention.
