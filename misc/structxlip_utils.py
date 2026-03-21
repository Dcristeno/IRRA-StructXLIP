import random
import re
from pathlib import Path

import yaml
from PIL import ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF


def load_struct_lexicon(lexicon_path):
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collapse_spaces(text):
    return re.sub(r"\s{2,}", " ", text).strip()


def filter_english_terms(text, terms):
    filtered = text
    for term in sorted(set(terms), key=len, reverse=True):
        filtered = re.sub(rf"\b{re.escape(term.lower())}\b", " ", filtered)
    return collapse_spaces(filtered)


def build_structure_caption(caption, args, lexicon):
    caption = caption.strip()
    if not caption:
        return caption

    language_terms = lexicon.get("en", {})
    active_terms = []
    if args.struct_remove_colors:
        active_terms.extend(language_terms.get("colors", []))
    if args.struct_remove_materials:
        active_terms.extend(language_terms.get("materials", []))
    if args.struct_remove_background:
        active_terms.extend(language_terms.get("background", []))

    if not active_terms:
        return caption

    filtered = filter_english_terms(caption.lower(), active_terms)
    if len(filtered.split()) < args.struct_min_words:
        return caption
    return filtered


def make_edge_image(image, threshold):
    edge = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge = edge.point(lambda p: 255 if p >= threshold else 0)
    return edge.convert("RGB")


class PairStructTrainTransform:
    def __init__(self, size, aug, edge_threshold):
        self.size = tuple(size)
        self.aug = aug
        self.edge_threshold = edge_threshold
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __call__(self, image):
        image = image.convert("RGB")
        edge = make_edge_image(image, self.edge_threshold)

        image = TF.resize(image, self.size)
        edge = TF.resize(edge, self.size)

        if self.aug:
            if random.random() < 0.5:
                image = TF.hflip(image)
                edge = TF.hflip(edge)

            image = TF.pad(image, 10)
            edge = TF.pad(edge, 10)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.size)
            image = TF.crop(image, i, j, h, w)
            edge = TF.crop(edge, i, j, h, w)
        else:
            if random.random() < 0.5:
                image = TF.hflip(image)
                edge = TF.hflip(edge)

        image = TF.to_tensor(image)
        edge = TF.to_tensor(edge)
        image = self.normalize(image)
        edge = self.normalize(edge)

        if self.aug and random.random() < 0.5:
            image = transforms.RandomErasing(scale=(0.02, 0.4), value=self.mean)(image)

        return image, edge


def default_struct_lexicon_path():
    return Path(__file__).resolve().parent / "structxlip_lexicon.yaml"
