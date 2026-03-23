import re
from pathlib import Path

import torch
import yaml


def load_color_lexicon(lexicon_path):
    with open(lexicon_path, "r", encoding="utf-8") as f:
        lexicon = yaml.safe_load(f)
    colors = lexicon.get("colors", {})
    ordered_names = list(colors.keys())
    return {
        "color_names": ordered_names,
        "patterns": {
            name: [re.compile(rf"\b{re.escape(term.lower())}\b") for term in terms]
            for name, terms in colors.items()
        },
    }


def default_color_lexicon_path():
    return Path(__file__).resolve().parent / "color_lexicon.yaml"


def build_color_label(caption, lexicon):
    caption = caption.lower().strip()
    labels = torch.zeros(len(lexicon["color_names"]), dtype=torch.float32)
    if not caption:
        return labels

    for idx, color_name in enumerate(lexicon["color_names"]):
        if any(pattern.search(caption) for pattern in lexicon["patterns"][color_name]):
            labels[idx] = 1.0
    return labels
