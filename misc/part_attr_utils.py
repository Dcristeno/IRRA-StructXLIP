import re
from pathlib import Path

import yaml


def load_part_attr_lexicon(lexicon_path):
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_part_attr_lexicon_path():
    return Path(__file__).resolve().parent / "part_attr_lexicon.yaml"


def _tokenize_words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _collect_phrase(words, keywords, window):
    matched_indices = [idx for idx, word in enumerate(words) if word in keywords]
    if not matched_indices:
        return "", 0

    keep = [False] * len(words)
    for idx in matched_indices:
        start = max(0, idx - window)
        end = min(len(words), idx + window + 1)
        for pos in range(start, end):
            keep[pos] = True

    phrase_words = [word for word, keep_flag in zip(words, keep) if keep_flag]
    if not phrase_words:
        return "", 0
    return " ".join(phrase_words), 1


def build_part_attribute_captions(caption, args, lexicon):
    words = _tokenize_words(caption)
    window = args.part_window_size
    part_captions = {}
    part_masks = {}

    for part_name, keywords in lexicon["parts"].items():
        phrase, valid = _collect_phrase(words, set(keywords), window)
        part_captions[part_name] = phrase
        part_masks[part_name] = valid

    return part_captions, part_masks
