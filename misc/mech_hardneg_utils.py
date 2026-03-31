import torch
import torch.nn.functional as F


def build_valid_token_mask(caption_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(caption_ids.shape[1], device=caption_ids.device).unsqueeze(0)
    eot_positions = caption_ids.argmax(dim=1, keepdim=True)
    return (positions > 0) & (positions < eot_positions)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denom


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_values = values.masked_fill(~mask, float("-inf"))
    max_values = masked_values.max(dim=1).values
    fallback = torch.zeros_like(max_values)
    return torch.where(torch.isfinite(max_values), max_values, fallback)


def select_hard_negative_indices(similarity: torch.Tensor, pids: torch.Tensor):
    negative_mask = pids.view(-1, 1) != pids.view(1, -1)
    if similarity.shape[0] <= 1:
        batch_size = similarity.shape[0]
        return torch.arange(batch_size, device=similarity.device), torch.zeros(batch_size, dtype=torch.bool, device=similarity.device)

    masked_similarity = similarity.masked_fill(~negative_mask, float("-inf"))
    hard_indices = masked_similarity.argmax(dim=1)
    has_negative = negative_mask.any(dim=1)
    fallback_indices = torch.arange(similarity.shape[0], device=similarity.device)
    hard_indices = torch.where(has_negative, hard_indices, fallback_indices)
    return hard_indices, has_negative


def compute_pairwise_token_evidence(
    text_tokens: torch.Tensor,
    image_patch_tokens: torch.Tensor,
    image_indices: torch.Tensor,
) -> torch.Tensor:
    selected_patches = image_patch_tokens[image_indices]
    token_patch_similarity = torch.einsum("btd,bpd->btp", text_tokens, selected_patches)
    return token_patch_similarity.max(dim=-1).values


def compute_dominance(evidence: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
    mean_evidence = masked_mean(evidence, valid_token_mask)
    peak_evidence = masked_max(evidence, valid_token_mask)
    return F.relu(peak_evidence - mean_evidence)
