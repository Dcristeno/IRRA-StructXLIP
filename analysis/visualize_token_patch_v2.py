import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from analysis.visualization_utils import (
    candidate_terms_from_caption,
    draw_retrieval_board,
    draw_token_patch_comparison_v2,
    draw_token_patch_grid_v2,
    mkdir,
    pil_to_normalized_numpy,
    sanitize_filename,
    save_text,
)
from datasets.build import build_transforms
from utils.simple_tokenizer import SimpleTokenizer

from analysis.visualize_token_patch import (
    compute_gallery_features,
    compute_term_heatmap,
    compute_text_features,
    load_dataset,
    load_model,
    parse_focus_terms,
    resolve_term_indices,
    select_gallery_indices,
)
from utils.iotools import load_train_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Second-generation token-to-patch visualization for IRRA.")
    parser.add_argument("--config_file", required=True, help="Path to the saved IRRA configs.yaml")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to load. Defaults to <output_dir>/best.pth")
    parser.add_argument("--query_index", type=int, default=0, help="Query caption index in the test split")
    parser.add_argument("--topk", type=int, default=5, help="How many retrieved gallery images to visualize individually")
    parser.add_argument("--focus_terms", default="", help='Comma-separated terms or phrases, e.g. "red,backpack,white shoes"')
    parser.add_argument("--max_auto_terms", type=int, default=5, help="Used when --focus_terms is empty")
    parser.add_argument("--include_gt", action="store_true", help="Also include the best-matching ground-truth image if it is outside top-k")
    parser.add_argument("--hard_negative_count", type=int, default=2, help="How many highest-ranked wrong matches to compare against the best positive")
    parser.add_argument("--threshold_percentile", type=float, default=80.0, help="Keep only the highest-response region above this percentile")
    parser.add_argument("--output_dir", default="analysis_outputs/token_patch_v2", help="Where to save figures")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    return parser.parse_args()


def choose_comparison_items(
    selected_gallery: Sequence[dict],
    query_pid: int,
    gallery_pids: torch.Tensor,
    hard_negative_count: int,
) -> List[dict]:
    positives = [item for item in selected_gallery if int(gallery_pids[item["index"]].item()) == query_pid]
    negatives = [item for item in selected_gallery if int(gallery_pids[item["index"]].item()) != query_pid]

    if not positives:
        raise RuntimeError("No positive match found in the selected gallery list. Increase --topk or add --include_gt.")

    comparison = [positives[0]]
    comparison.extend(negatives[:hard_negative_count])
    return comparison


def main():
    cli_args = parse_args()
    config = load_train_configs(cli_args.config_file)
    config.training = False
    if cli_args.checkpoint:
        config.checkpoint = cli_args.checkpoint
    device = torch.device(cli_args.device if cli_args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset = load_dataset(config)
    num_classes = len(dataset.train_id_container)
    model = load_model(config, num_classes=num_classes, device=device)
    tokenizer = SimpleTokenizer()
    transform = build_transforms(img_size=config.img_size, is_train=False)

    gallery_image_paths = dataset.test["img_paths"]
    gallery_pids = torch.tensor(dataset.test["image_pids"])
    captions = dataset.test["captions"]
    caption_pids = torch.tensor(dataset.test["caption_pids"])

    gallery_global, gallery_patches = compute_gallery_features(
        model,
        gallery_image_paths,
        transform,
        batch_size=config.test_batch_size,
        device=device,
    )
    text_global, text_tokens, caption_ids = compute_text_features(
        model,
        captions,
        text_length=config.text_length,
        tokenizer=tokenizer,
        batch_size=config.test_batch_size,
        device=device,
    )

    similarity = text_global @ gallery_global.t()

    query_index = cli_args.query_index
    if query_index < 0 or query_index >= len(captions):
        raise IndexError(f"query_index {query_index} is out of range for {len(captions)} captions")

    query_caption = captions[query_index]
    query_pid = int(caption_pids[query_index].item())
    query_caption_ids = caption_ids[query_index]
    query_token_features = text_tokens[query_index]

    focus_terms = parse_focus_terms(cli_args.focus_terms, query_caption, cli_args.max_auto_terms)
    resolved_terms = resolve_term_indices(query_caption_ids, focus_terms, tokenizer)
    if not resolved_terms:
        raise RuntimeError("No focus terms could be aligned to token positions. Try simpler terms or pass --focus_terms explicitly.")

    selected_gallery = select_gallery_indices(
        similarity_row=similarity[query_index],
        qpid=query_pid,
        gallery_pids=gallery_pids,
        topk=cli_args.topk,
        include_gt=cli_args.include_gt,
    )
    comparison_items = choose_comparison_items(
        selected_gallery=selected_gallery,
        query_pid=query_pid,
        gallery_pids=gallery_pids,
        hard_negative_count=cli_args.hard_negative_count,
    )

    query_dir = os.path.join(
        cli_args.output_dir,
        sanitize_filename(os.path.splitext(os.path.basename(cli_args.config_file))[0]),
        f"query_{query_index:05d}",
    )
    mkdir(query_dir)

    board_items = []
    for item in selected_gallery:
        gallery_index = item["index"]
        raw_image = Image.open(gallery_image_paths[gallery_index]).convert("RGB")
        board_items.append({
            "label": item["label"],
            "pid": int(gallery_pids[gallery_index].item()),
            "score": float(similarity[query_index, gallery_index].item()),
            "is_match": int(gallery_pids[gallery_index].item()) == query_pid,
            "image": pil_to_normalized_numpy(raw_image, config.img_size),
        })

    draw_retrieval_board(
        query_caption=query_caption,
        query_pid=query_pid,
        gallery_items=board_items,
        output_path=os.path.join(query_dir, "retrieval_board_v2.png"),
    )

    grid_size = (model.base_model.visual.num_y, model.base_model.visual.num_x)
    prepared_comparison = []

    for item in selected_gallery:
        gallery_index = item["index"]
        raw_image = Image.open(gallery_image_paths[gallery_index]).convert("RGB")
        base_image = pil_to_normalized_numpy(raw_image, config.img_size)
        patch_tokens = gallery_patches[gallery_index]

        term_heatmaps = []
        term_heatmap_map = {}
        for term_info in resolved_terms:
            heatmap = compute_term_heatmap(
                term_positions=term_info["positions"],
                text_tokens=query_token_features,
                patch_tokens=patch_tokens,
                grid_size=grid_size,
            )
            term_heatmaps.append({"term": term_info["term"], "heatmap": heatmap})
            term_heatmap_map[term_info["term"]] = heatmap

        draw_token_patch_grid_v2(
            query_caption=query_caption,
            gallery_label=item["label"],
            gallery_pid=int(gallery_pids[gallery_index].item()),
            gallery_score=float(similarity[query_index, gallery_index].item()),
            base_image=base_image,
            term_heatmaps=term_heatmaps,
            output_path=os.path.join(query_dir, f'{sanitize_filename(item["label"])}_token_patch_v2.png'),
            threshold_percentile=cli_args.threshold_percentile,
        )

        if item in comparison_items:
            prepared_comparison.append({
                "label": item["label"],
                "pid": int(gallery_pids[gallery_index].item()),
                "score": float(similarity[query_index, gallery_index].item()),
                "is_match": int(gallery_pids[gallery_index].item()) == query_pid,
                "image": base_image,
                "term_heatmaps": term_heatmap_map,
            })

    # Keep the same row order as comparison_items
    ordered_comparison = []
    for item in comparison_items:
        for prepared in prepared_comparison:
            if prepared["label"] == item["label"]:
                ordered_comparison.append(prepared)
                break

    draw_token_patch_comparison_v2(
        query_caption=query_caption,
        compared_items=ordered_comparison,
        term_names=[term["term"] for term in resolved_terms],
        output_path=os.path.join(query_dir, "comparison_board_v2.png"),
        threshold_percentile=cli_args.threshold_percentile,
    )

    save_text(
        os.path.join(query_dir, "summary_v2.txt"),
        [
            f"query_index: {query_index}",
            f"query_pid: {query_pid}",
            f"query_caption: {query_caption}",
            f"focus_terms: {', '.join(term['term'] for term in resolved_terms)}",
            f"threshold_percentile: {cli_args.threshold_percentile}",
            f"hard_negative_count: {cli_args.hard_negative_count}",
            "selected_gallery:",
            *[
                f'  {item["label"]}: pid={int(gallery_pids[item["index"]].item())}, score={float(similarity[query_index, item["index"]].item()):.4f}, path={gallery_image_paths[item["index"]]}'
                for item in selected_gallery
            ],
            "comparison_rows:",
            *[
                f'  {item["label"]}: pid={int(gallery_pids[item["index"]].item())}, score={float(similarity[query_index, item["index"]].item()):.4f}'
                for item in comparison_items
            ],
        ],
    )

    print(f"Saved v2 analysis to {query_dir}")


if __name__ == "__main__":
    main()
