import math
import os
import re
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "their", "there",
    "this", "to", "up", "wearing", "with", "woman", "man", "person", "people",
}


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "item"


def unnormalize_image(image_tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return image


def pil_to_normalized_numpy(image: Image.Image, size: Sequence[int]) -> np.ndarray:
    resized = image.resize((size[1], size[0]))
    array = np.asarray(resized).astype(np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        return heatmap
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if math.isclose(max_value, min_value):
        return np.zeros_like(heatmap)
    return (heatmap - min_value) / (max_value - min_value)


def resize_heatmap_to_image(heatmap: np.ndarray, base_image_shape: Tuple[int, int, int]) -> np.ndarray:
    heatmap = normalize_heatmap(heatmap)
    if heatmap.shape[:2] == base_image_shape[:2]:
        return heatmap
    return np.asarray(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (base_image_shape[1], base_image_shape[0]),
            resample=Image.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0


def candidate_terms_from_caption(caption: str, max_terms: int = 6) -> List[str]:
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", caption.lower())
    terms = []
    for word in words:
        if word in DEFAULT_STOPWORDS or len(word) <= 2:
            continue
        if word not in terms:
            terms.append(word)
        if len(terms) >= max_terms:
            break
    return terms


def overlay_heatmap(base_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55, cmap: str = "jet") -> np.ndarray:
    heatmap = resize_heatmap_to_image(heatmap, base_image.shape)
    colored = plt.get_cmap(cmap)(heatmap)[..., :3]
    blended = (1.0 - alpha) * base_image + alpha * colored
    return np.clip(blended, 0.0, 1.0)


def overlay_red_mask(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    threshold_percentile: float = 80.0,
    alpha: float = 0.72,
    desaturate: bool = True,
) -> np.ndarray:
    resized = resize_heatmap_to_image(heatmap, base_image.shape)
    if desaturate:
        gray = np.dot(base_image[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        canvas = np.repeat(gray[..., None], 3, axis=2)
    else:
        canvas = base_image.copy()

    threshold = np.percentile(resized, threshold_percentile)
    mask_strength = np.clip((resized - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    red_mask = np.zeros_like(canvas)
    red_mask[..., 0] = 1.0
    blended = canvas * (1.0 - alpha * mask_strength[..., None]) + red_mask * (alpha * mask_strength[..., None])
    return np.clip(blended, 0.0, 1.0)


def draw_retrieval_board(
    query_caption: str,
    query_pid: int,
    gallery_items: Sequence[dict],
    output_path: str,
) -> None:
    columns = len(gallery_items)
    fig, axes = plt.subplots(1, columns, figsize=(3.6 * columns, 5.2))
    if columns == 1:
        axes = [axes]

    wrapped_caption = "\n".join(query_caption[i:i + 80] for i in range(0, len(query_caption), 80))
    fig.suptitle(f"Query PID {query_pid}\n{wrapped_caption}", fontsize=12)

    for axis, item in zip(axes, gallery_items):
        axis.imshow(item["image"])
        axis.set_xticks([])
        axis.set_yticks([])
        border_color = "lawngreen" if item["is_match"] else "red"
        for spine in axis.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        axis.set_title(
            f'{item["label"]}\nPID={item["pid"]}  score={item["score"]:.3f}',
            fontsize=10,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_token_patch_grid(
    query_caption: str,
    gallery_label: str,
    gallery_pid: int,
    gallery_score: float,
    base_image: np.ndarray,
    term_heatmaps: Sequence[dict],
    output_path: str,
) -> None:
    columns = len(term_heatmaps) + 1
    fig, axes = plt.subplots(1, columns, figsize=(3.4 * columns, 5.2))
    if columns == 1:
        axes = [axes]

    axes[0].imshow(base_image)
    axes[0].set_title(f"{gallery_label}\nPID={gallery_pid}  score={gallery_score:.3f}", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for axis, item in zip(axes[1:], term_heatmaps):
        overlay = overlay_heatmap(base_image, item["heatmap"])
        axis.imshow(overlay)
        axis.set_title(item["term"], fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])

    wrapped_caption = "\n".join(query_caption[i:i + 85] for i in range(0, len(query_caption), 85))
    fig.suptitle(f"{gallery_label} token-to-patch activation\n{wrapped_caption}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_token_patch_grid_v2(
    query_caption: str,
    gallery_label: str,
    gallery_pid: int,
    gallery_score: float,
    base_image: np.ndarray,
    term_heatmaps: Sequence[dict],
    output_path: str,
    threshold_percentile: float = 80.0,
) -> None:
    columns = len(term_heatmaps) + 1
    fig, axes = plt.subplots(1, columns, figsize=(3.1 * columns, 5.4))
    if columns == 1:
        axes = [axes]

    axes[0].imshow(base_image)
    axes[0].set_title(f"{gallery_label}\nPID={gallery_pid}  score={gallery_score:.3f}", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for axis, item in zip(axes[1:], term_heatmaps):
        overlay = overlay_red_mask(
            base_image,
            item["heatmap"],
            threshold_percentile=threshold_percentile,
        )
        axis.imshow(overlay)
        axis.set_title(f'{item["term"]}\nTop {100 - int(threshold_percentile)}% response', fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])

    wrapped_caption = "\n".join(query_caption[i:i + 82] for i in range(0, len(query_caption), 82))
    fig.suptitle(f"{gallery_label} sparse token evidence view\n{wrapped_caption}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_token_patch_comparison_v2(
    query_caption: str,
    compared_items: Sequence[dict],
    term_names: Sequence[str],
    output_path: str,
    threshold_percentile: float = 80.0,
) -> None:
    columns = len(term_names) + 1
    rows = len(compared_items)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 4.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if columns == 1:
        axes = np.expand_dims(axes, 1)

    for row_idx, item in enumerate(compared_items):
        axes[row_idx, 0].imshow(item["image"])
        axes[row_idx, 0].set_title(
            f'{item["label"]}\nPID={item["pid"]}  score={item["score"]:.3f}',
            fontsize=10,
        )
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        border_color = "lawngreen" if item["is_match"] else "red"
        for spine in axes[row_idx, 0].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        for col_idx, term_name in enumerate(term_names, start=1):
            term_heatmap = item["term_heatmaps"][term_name]
            overlay = overlay_red_mask(
                item["image"],
                term_heatmap,
                threshold_percentile=threshold_percentile,
            )
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(term_name, fontsize=10)

    wrapped_caption = "\n".join(query_caption[i:i + 95] for i in range(0, len(query_caption), 95))
    fig.suptitle(
        f"Positive vs hard-negative sparse evidence comparison\n{wrapped_caption}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_text(path: str, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
