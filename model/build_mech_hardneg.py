from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.mech_hardneg_utils import (
    build_valid_token_mask,
    compute_dominance,
    compute_pairwise_token_evidence,
    masked_mean,
    select_hard_negative_indices,
)
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights


class IRRAMechHardNeg(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size,
        )
        if not hasattr(self.base_model.visual, "num_y") or not hasattr(self.base_model.visual, "num_x"):
            raise RuntimeError("Mechanism-guided hard negatives currently require a ViT-based CLIP visual encoder.")
        self.embed_dim = base_cfg["embed_dim"]

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if "id" in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if "mlm" in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim,
                self.embed_dim // 64,
                batch_first=True,
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim,
                layers=args.cmt_depth,
                heads=self.embed_dim // 64,
            )
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict(
                    [
                        ("dense", nn.Linear(self.embed_dim, self.embed_dim)),
                        ("gelu", QuickGELU()),
                        ("ln", LayerNorm(self.embed_dim)),
                        ("fc", nn.Linear(self.embed_dim, args.vocab_size)),
                    ]
                )
            )
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split("+")]
        print(f"Training Model with {self.current_task} tasks")

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False,
        )[0]
        x = x.permute(1, 0, 2)
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def _compute_dehn_loss(self, image_feats, text_feats, caption_ids, pids):
        text_global = F.normalize(text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float(), p=2, dim=-1)
        image_global = F.normalize(image_feats[:, 0, :].float(), p=2, dim=-1)
        similarity = text_global @ image_global.t()

        hard_negative_indices, has_negative = select_hard_negative_indices(similarity.detach(), pids)
        if not has_negative.any():
            zero = similarity.new_tensor(0.0)
            return {
                "dehn_loss": zero,
                "dehn_rank_term": zero,
                "dehn_cov_term": zero,
                "dehn_pos_sim": zero,
                "dehn_neg_sim": zero,
                "dehn_pos_cov": zero,
                "dehn_neg_cov": zero,
                "dehn_neg_weight": zero,
                "dehn_neg_dominance": zero,
            }

        valid_token_mask = build_valid_token_mask(caption_ids)
        text_tokens = F.normalize(text_feats.float(), p=2, dim=-1)
        image_patch_tokens = F.normalize(image_feats[:, 1:, :].float(), p=2, dim=-1)

        batch_indices = torch.arange(text_feats.shape[0], device=text_feats.device)
        pos_token_evidence = compute_pairwise_token_evidence(text_tokens, image_patch_tokens, batch_indices)
        neg_token_evidence = compute_pairwise_token_evidence(text_tokens, image_patch_tokens, hard_negative_indices)

        pos_coverage = masked_mean(pos_token_evidence, valid_token_mask)
        neg_coverage = masked_mean(neg_token_evidence, valid_token_mask)
        pos_dominance = compute_dominance(pos_token_evidence, valid_token_mask)
        neg_dominance = compute_dominance(neg_token_evidence, valid_token_mask)

        pos_scores = similarity.diag()
        neg_scores = similarity[batch_indices, hard_negative_indices]

        active_mask = has_negative.float()
        rank_margin = F.relu(self.args.dehn_margin + neg_scores - pos_scores)
        coverage_margin = F.relu(self.args.dehn_cov_margin + neg_coverage - pos_coverage)
        dominance_gap = F.relu(neg_dominance - pos_dominance)
        score_surplus = F.relu(neg_scores - self.args.dehn_neg_score_floor)

        neg_weight = (
            1.0
            + self.args.dehn_score_weight * score_surplus
            + self.args.dehn_dominance_weight * dominance_gap
            + self.args.dehn_cov_weight * coverage_margin
        ).detach()

        denom = active_mask.sum().clamp_min(1.0)
        rank_loss = (rank_margin * neg_weight * active_mask).sum() / denom
        coverage_loss = (coverage_margin * active_mask).sum() / denom
        total_loss = (
            self.args.dehn_rank_loss_weight * rank_loss
            + self.args.dehn_cov_loss_weight * coverage_loss
        ) * self.args.dehn_loss_weight

        return {
            "dehn_loss": total_loss,
            "dehn_rank_term": rank_loss * self.args.dehn_loss_weight,
            "dehn_cov_term": coverage_loss * self.args.dehn_loss_weight,
            "dehn_pos_sim": (pos_scores * active_mask).sum() / denom,
            "dehn_neg_sim": (neg_scores * active_mask).sum() / denom,
            "dehn_pos_cov": (pos_coverage * active_mask).sum() / denom,
            "dehn_neg_cov": (neg_coverage * active_mask).sum() / denom,
            "dehn_neg_weight": (neg_weight * active_mask).sum() / denom,
            "dehn_neg_dominance": (neg_dominance * active_mask).sum() / denom,
        }

    def forward(self, batch):
        ret = {}

        images = batch["images"]
        caption_ids = batch["caption_ids"]
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({"temperature": 1 / logit_scale})

        if "itc" in self.current_task:
            ret.update({"itc_loss": objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if "sdm" in self.current_task:
            ret.update({"sdm_loss": objectives.compute_sdm(i_feats, t_feats, batch["pids"], logit_scale)})

        if "cmpm" in self.current_task:
            ret.update({"cmpm_loss": objectives.compute_cmpm(i_feats, t_feats, batch["pids"])})

        if "id" in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({"id_loss": objectives.compute_id(image_logits, text_logits, batch["pids"]) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)
            ret.update({"img_acc": (image_pred == batch["pids"]).float().mean()})
            ret.update({"txt_acc": (text_pred == batch["pids"]).float().mean()})

        if "mlm" in self.current_task:
            mlm_ids = batch["mlm_ids"]
            mlm_feats = self.base_model.encode_text(mlm_ids)
            x = self.cross_former(mlm_feats, image_feats, image_feats)
            x = self.mlm_head(x)

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch["mlm_labels"].reshape(-1)
            ret.update({"mlm_loss": objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            ret.update({"mlm_acc": (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()})

        ret.update(self._compute_dehn_loss(image_feats, text_feats, caption_ids, batch["pids"]))

        return ret


def build_model_mech_hardneg(args, num_classes=11003):
    model = IRRAMechHardNeg(args, num_classes)
    convert_weights(model)
    return model
