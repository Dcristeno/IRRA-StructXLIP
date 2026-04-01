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
from tqdm import tqdm

from analysis.visualization_utils import (
    candidate_terms_from_caption,
    draw_retrieval_board,
    draw_token_patch_grid,
    mkdir,
    pil_to_normalized_numpy,
    sanitize_filename,
    save_text,
)
from datasets.build import build_transforms
from datasets.bases import tokenize
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.simple_tokenizer import SimpleTokenizer


DATASETS = {
    "CUHK-PEDES": CUHKPEDES,
    "ICFG-PEDES": ICFGPEDES,
    "RSTPReid": RSTPReid,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize token-to-patch activation maps for IRRA.")
    parser.add_argument("--config_file", required=True, help="Path to the saved IRRA configs.yaml")
    parser.add_argument("--checkpoint", default="", help="Checkpoint to load. Defaults to <output_dir>/best.pth")
    parser.add_argument("--query_index", type=int, default=0, help="Query caption index in the test split")
    parser.add_argument("--topk", type=int, default=5, help="How many retrieved gallery images to visualize")
    parser.add_argument("--focus_terms", default="", help='Comma-separated terms or phrases, e.g. "red,backpack,white shoes"')
    parser.add_argument("--max_auto_terms", type=int, default=5, help="Used when --focus_terms is empty")
    parser.add_argument("--include_gt", action="store_true", help="Also visualize the best-matching ground-truth gallery image")
    parser.add_argument("--output_dir", default="analysis_outputs/token_patch", help="Where to save figures")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    return parser.parse_args()


def load_dataset(args):
    dataset_cls = DATASETS[args.dataset_name]
    return dataset_cls(root=args.root_dir, verbose=False)


def load_model(args, num_classes: int, device: torch.device):
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "best.pth")
    checkpointer.load(f=checkpoint_path)
    model.to(device)
    model.eval()
    return model


def build_caption_tensor(caption: str, text_length: int, tokenizer: SimpleTokenizer) -> torch.LongTensor:
    return tokenize(caption, tokenizer=tokenizer, text_length=text_length, truncate=True)


def compute_gallery_features(model, image_paths: Sequence[str], transform, batch_size: int, device: torch.device):
    global_features = []
    patch_features = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding gallery images"):
        batch_paths = image_paths[start:start + batch_size]
        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            batch_images.append(transform(image))
        batch_tensor = torch.stack(batch_images, dim=0).to(device)

        with torch.no_grad():
            image_tokens = model.base_model.encode_image(batch_tensor)

        cls_features = F.normalize(image_tokens[:, 0, :].float(), p=2, dim=-1)
        patches = F.normalize(image_tokens[:, 1:, :].float(), p=2, dim=-1)

        global_features.append(cls_features.cpu())
        patch_features.append(patches.cpu())

    return (
        torch.cat(global_features, dim=0),
        torch.cat(patch_features, dim=0),
    )


def compute_text_features(model, captions: Sequence[str], text_length: int, tokenizer: SimpleTokenizer, batch_size: int, device: torch.device):
    caption_tensors = []
    global_features = []
    token_features = []

    for start in tqdm(range(0, len(captions), batch_size), desc="Encoding query captions"):
        batch_captions = captions[start:start + batch_size]
        batch_tokens = [build_caption_tensor(caption, text_length, tokenizer) for caption in batch_captions]
        batch_tensor = torch.stack(batch_tokens, dim=0).to(device)

        with torch.no_grad():
            text_tokens = model.base_model.encode_text(batch_tensor)

        eot_positions = batch_tensor.argmax(dim=-1)
        cls_features = text_tokens[torch.arange(text_tokens.shape[0]), eot_positions].float()
        cls_features = F.normalize(cls_features, p=2, dim=-1)
        token_features.append(F.normalize(text_tokens.float(), p=2, dim=-1).cpu())
        global_features.append(cls_features.cpu())
        caption_tensors.extend([tokens.cpu() for tokens in batch_tokens])

    return (
        torch.cat(global_features, dim=0),
        torch.cat(token_features, dim=0),
        caption_tensors,
    )


def parse_focus_terms(raw_terms: str, caption: str, max_auto_terms: int) -> List[str]:
    if raw_terms.strip():
        return [term.strip().lower() for term in raw_terms.split(",") if term.strip()]
    return candidate_terms_from_caption(caption, max_terms=max_auto_terms)


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> List[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    pattern = list(pattern)
    for start in range(len(sequence) - len(pattern) + 1):
        if list(sequence[start:start + len(pattern)]) == pattern:
            return list(range(start, start + len(pattern)))
    return []


def resolve_term_indices(
    caption_ids: torch.LongTensor,
    terms: Sequence[str],
    tokenizer: SimpleTokenizer,
) -> List[Dict]:
    nonzero_ids = caption_ids.tolist()
    try:
        eot_index = nonzero_ids.index(tokenizer.encoder["<|endoftext|>"])
    except ValueError:
        eot_index = len(nonzero_ids)
    content_ids = nonzero_ids[1:eot_index]

    resolved = []
    for term in terms:
        term_token_ids = tokenizer.encode(term)
        match_positions = find_subsequence(content_ids, term_token_ids)
        if not match_positions:
            continue
        resolved.append({"term": term, "positions": [position + 1 for position in match_positions]})
    return resolved


def select_gallery_indices(similarity_row: torch.Tensor, qpid: int, gallery_pids: torch.Tensor, topk: int, include_gt: bool):
    topk = min(topk, similarity_row.numel())
    top_indices = torch.topk(similarity_row, k=topk, largest=True, sorted=True).indices.tolist()

    selected = [{"label": f"top{rank + 1}", "index": idx} for rank, idx in enumerate(top_indices)]
    if include_gt:
        gt_mask = gallery_pids == qpid
        gt_indices = torch.nonzero(gt_mask, as_tuple=False).view(-1)
        if gt_indices.numel() > 0:
            gt_scores = similarity_row[gt_indices]
            gt_best = gt_indices[torch.argmax(gt_scores)].item()
            if gt_best not in [item["index"] for item in selected]:
                selected.append({"label": "gt-best", "index": gt_best})
    return selected


def compute_term_heatmap(term_positions: Sequence[int], text_tokens: torch.Tensor, patch_tokens: torch.Tensor, grid_size):
    term_feature = text_tokens[list(term_positions)].mean(dim=0)
    term_feature = F.normalize(term_feature.unsqueeze(0), p=2, dim=-1).squeeze(0)
    scores = patch_tokens @ term_feature
    heatmap = scores.view(grid_size[0], grid_size[1]).numpy()
    return heatmap


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
        output_path=os.path.join(query_dir, "retrieval_board.png"),
    )

    grid_size = (model.base_model.visual.num_y, model.base_model.visual.num_x)
    for item in selected_gallery:
        gallery_index = item["index"]
        raw_image = Image.open(gallery_image_paths[gallery_index]).convert("RGB")
        base_image = pil_to_normalized_numpy(raw_image, config.img_size)
        patch_tokens = gallery_patches[gallery_index]

        term_heatmaps = []
        for term_info in resolved_terms:
            heatmap = compute_term_heatmap(
                term_positions=term_info["positions"],
                text_tokens=query_token_features,
                patch_tokens=patch_tokens,
                grid_size=grid_size,
            )
            term_heatmaps.append({"term": term_info["term"], "heatmap": heatmap})

        draw_token_patch_grid(
            query_caption=query_caption,
            gallery_label=item["label"],
            gallery_pid=int(gallery_pids[gallery_index].item()),
            gallery_score=float(similarity[query_index, gallery_index].item()),
            base_image=base_image,
            term_heatmaps=term_heatmaps,
            output_path=os.path.join(query_dir, f'{sanitize_filename(item["label"])}_token_patch.png'),
        )

    save_text(
        os.path.join(query_dir, "summary.txt"),
        [
            f"query_index: {query_index}",
            f"query_pid: {query_pid}",
            f"query_caption: {query_caption}",
            f"focus_terms: {', '.join(term['term'] for term in resolved_terms)}",
            "selected_gallery:",
            *[
                f'  {item["label"]}: pid={int(gallery_pids[item["index"]].item())}, score={float(similarity[query_index, item["index"]].item()):.4f}, path={gallery_image_paths[item["index"]]}'
                for item in selected_gallery
            ],
        ],
    )

    print(f"Saved analysis to {query_dir}")


if __name__ == "__main__":
    main()
