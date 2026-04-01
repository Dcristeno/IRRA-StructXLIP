# DEHN-v2

`DEHN-v2` upgrades the original hard-negative weighting idea with phrase-aware
and support-evidence-aware supervision.

Compared with `DEHN-v1`, this version:

- groups token evidence into short noun phrases
- separates dominant phrases from supporting phrases
- penalizes hard negatives that match dominant evidence but miss supporting evidence

Entry point:

```bash
python train_mech_hardneg_v2.py --help
```
