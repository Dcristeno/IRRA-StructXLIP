# IRRA + StructXLIP-lite

This extension keeps the original IRRA training code untouched and adds a parallel StructXLIP-lite training path.

Added files:

- `train_structxlip.py`
- `utils/options_structxlip.py`
- `datasets/structxlip.py`
- `model/build_structxlip.py`
- `processor/processor_structxlip.py`
- `misc/structxlip_utils.py`
- `misc/structxlip_lexicon.yaml`

Suggested ablations:

1. `--struct_global_weight 0 --struct_consistency_weight 0 --struct_local_weight 0`
2. `--struct_global_weight 0.5 --struct_consistency_weight 0 --struct_local_weight 0`
3. `--struct_global_weight 0.5 --struct_consistency_weight 0.1 --struct_local_weight 0`
4. `--struct_global_weight 0.5 --struct_consistency_weight 0.1 --struct_local_weight 0.05`
5. Compare `--struct_remove_colors` on vs off
