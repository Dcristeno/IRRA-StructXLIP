import torch
import torch.nn.functional as F

from utils.simple_tokenizer import SimpleTokenizer


SUPPORT_NOUNS = {
    "pants", "shorts", "jeans", "trousers", "leggings", "skirt",
    "shoes", "shoe", "sneakers", "sneaker", "boots", "boot", "sandals", "sandal", "heels", "heel",
    "backpack", "bag", "purse", "handbag", "cap", "hat", "glasses", "goggles",
    "headphones", "headphone", "scarf", "socks", "sock",
}

ANCHOR_NOUNS = SUPPORT_NOUNS | {
    "top", "shirt", "shirts", "tee", "tshirt", "jacket", "coat", "hoodie", "sweater", "vest", "dress", "uniform",
}

STOPWORDS = {
    "a", "an", "the", "of", "and", "with", "in", "on", "at", "to", "for", "from",
    "pair", "wearing", "wears", "wear", "is", "are", "has", "have", "he", "she",
    "man", "woman", "person", "young",
}


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


def compute_pairwise_token_evidence(text_tokens: torch.Tensor, image_patch_tokens: torch.Tensor, image_indices: torch.Tensor) -> torch.Tensor:
    selected_patches = image_patch_tokens[image_indices]
    token_patch_similarity = torch.einsum("btd,bpd->btp", text_tokens, selected_patches)
    return token_patch_similarity.max(dim=-1).values


def _decode_bpe_piece(piece: str, tokenizer: SimpleTokenizer):
    ended = piece.endswith("</w>")
    bare_piece = piece.replace("</w>", "")
    decoded = bytearray([tokenizer.byte_decoder[c] for c in bare_piece]).decode("utf-8", errors="replace")
    return decoded.lower(), ended


def _caption_to_words(caption_ids_row: torch.Tensor, tokenizer: SimpleTokenizer):
    words = []
    current_word = ""
    current_positions = []
    for position, token_id in enumerate(caption_ids_row.tolist()):
        if token_id == 0:
            continue
        token_piece = tokenizer.decoder.get(token_id, "")
        if token_piece in {"<|startoftext|>", "<|endoftext|>", "<|mask|>"}:
            continue
        decoded_piece, ended = _decode_bpe_piece(token_piece, tokenizer)
        if not decoded_piece:
            continue
        current_word += decoded_piece
        current_positions.append(position)
        if ended:
            word = current_word.strip()
            if word:
                words.append((word, current_positions[:]))
            current_word = ""
            current_positions = []
    if current_word.strip() and current_positions:
        words.append((current_word.strip(), current_positions[:]))
    return words


def _extract_phrases_from_words(words, max_modifier_words: int):
    phrases = []
    for idx, (word, _) in enumerate(words):
        if word not in ANCHOR_NOUNS:
            continue
        start = idx
        modifier_budget = 0
        while start - 1 >= 0 and modifier_budget < max_modifier_words:
            prev_word = words[start - 1][0]
            if prev_word in STOPWORDS or prev_word in ANCHOR_NOUNS:
                break
            start -= 1
            modifier_budget += 1
        phrase_words = [item[0] for item in words[start: idx + 1]]
        token_positions = []
        for _, positions in words[start: idx + 1]:
            token_positions.extend(positions)
        phrase = {
            "text": " ".join(phrase_words),
            "token_positions": token_positions,
            "is_support": any(item in SUPPORT_NOUNS for item in phrase_words),
        }
        if not phrases or phrases[-1]["token_positions"] != token_positions:
            phrases.append(phrase)

    if not phrases:
        for word, positions in words[:3]:
            phrases.append({"text": word, "token_positions": positions, "is_support": word in SUPPORT_NOUNS})
    return phrases


def build_phrase_matrices(caption_ids: torch.Tensor, tokenizer: SimpleTokenizer, max_phrases: int = 8, max_modifier_words: int = 3):
    batch_size, text_length = caption_ids.shape
    phrase_token_mask = torch.zeros(batch_size, max_phrases, text_length, dtype=torch.bool, device=caption_ids.device)
    phrase_mask = torch.zeros(batch_size, max_phrases, dtype=torch.bool, device=caption_ids.device)
    support_mask = torch.zeros(batch_size, max_phrases, dtype=torch.bool, device=caption_ids.device)

    for batch_idx in range(batch_size):
        words = _caption_to_words(caption_ids[batch_idx].detach().cpu(), tokenizer)
        phrases = _extract_phrases_from_words(words, max_modifier_words)[:max_phrases]
        for phrase_idx, phrase in enumerate(phrases):
            phrase_mask[batch_idx, phrase_idx] = True
            if phrase["is_support"]:
                support_mask[batch_idx, phrase_idx] = True
            positions = torch.tensor(phrase["token_positions"], dtype=torch.long, device=caption_ids.device)
            phrase_token_mask[batch_idx, phrase_idx, positions] = True

    return phrase_token_mask, phrase_mask, support_mask


def aggregate_token_evidence_to_phrase(token_evidence: torch.Tensor, phrase_token_mask: torch.Tensor, phrase_mask: torch.Tensor) -> torch.Tensor:
    expanded_evidence = token_evidence.unsqueeze(1).expand(-1, phrase_token_mask.shape[1], -1)
    phrase_lengths = phrase_token_mask.float().sum(dim=-1).clamp_min(1.0)
    phrase_evidence = (expanded_evidence * phrase_token_mask.float()).sum(dim=-1) / phrase_lengths
    return phrase_evidence * phrase_mask.float()


def resolve_support_and_dominant_masks(phrase_mask: torch.Tensor, support_mask: torch.Tensor):
    support_exists = support_mask.any(dim=1, keepdim=True)
    support_base_mask = torch.where(support_exists, support_mask, phrase_mask)

    dominant_mask = phrase_mask & ~support_mask
    dominant_exists = dominant_mask.any(dim=1, keepdim=True)
    dominant_base_mask = torch.where(dominant_exists, dominant_mask, phrase_mask)
    return support_base_mask, dominant_base_mask
