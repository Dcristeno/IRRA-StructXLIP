from collections import OrderedDict

import torch
import torch.nn as nn

from model import objectives
from .clip_model import LayerNorm, QuickGELU, Transformer, build_CLIP_from_openai_pretrained, convert_weights


PART_NAMES = ("upper", "lower", "shoes")


class IRRAPartAttrAlign(nn.Module):
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
            raise RuntimeError("Part-aware attribute alignment currently requires a ViT-based CLIP visual encoder.")
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
        self.current_task = [l.strip() for l in self.args.loss_names.split("+")]
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

    def _pool_image_parts(self, image_tokens):
        patch_tokens = image_tokens[:, 1:, :].float()
        num_y = self.base_model.visual.num_y
        num_x = self.base_model.visual.num_x
        patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], num_y, num_x, patch_tokens.shape[-1])

        upper_end = max(1, int(round(num_y * self.args.part_upper_ratio)))
        lower_end = max(upper_end + 1, int(round(num_y * self.args.part_lower_ratio)))
        lower_end = min(lower_end, num_y - 1)

        part_ranges = {
            "upper": (0, upper_end),
            "lower": (upper_end, lower_end),
            "shoes": (lower_end, num_y),
        }

        pooled = {}
        for part_name, (start, end) in part_ranges.items():
            pooled[part_name] = patch_tokens[:, start:end, :, :].mean(dim=(1, 2))
        return pooled

    def _compute_part_alignment(self, image_part_feats, batch):
        total = image_part_feats["upper"].new_tensor(0.0)
        valid_parts = 0
        per_part_losses = {}

        for part_name in PART_NAMES:
            part_mask = batch[f"{part_name}_mask"] > 0
            if part_mask.sum().item() < 2:
                per_part_losses[f"{part_name}_part_sdm"] = total.new_tensor(0.0)
                continue

            part_caption_ids = batch[f"{part_name}_caption_ids"]
            text_tokens = self.base_model.encode_text(part_caption_ids)
            text_feats = text_tokens[torch.arange(text_tokens.shape[0]), part_caption_ids.argmax(dim=-1)].float()

            part_loss = objectives.compute_sdm(
                image_part_feats[part_name][part_mask],
                text_feats[part_mask],
                batch["pids"][part_mask],
                self.logit_scale,
            )
            per_part_losses[f"{part_name}_part_sdm"] = part_loss
            total = total + part_loss
            valid_parts += 1

        if valid_parts == 0:
            return total, per_part_losses
        return total / valid_parts, per_part_losses

    def forward(self, batch):
        ret = {}

        images = batch["images"]
        caption_ids = batch["caption_ids"]
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        ret["temperature"] = 1 / self.logit_scale

        if "itc" in self.current_task:
            ret["itc_loss"] = objectives.compute_itc(i_feats, t_feats, self.logit_scale)

        if "sdm" in self.current_task:
            ret["sdm_loss"] = objectives.compute_sdm(i_feats, t_feats, batch["pids"], self.logit_scale)

        if "cmpm" in self.current_task:
            ret["cmpm_loss"] = objectives.compute_cmpm(i_feats, t_feats, batch["pids"])

        if "id" in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret["id_loss"] = objectives.compute_id(image_logits, text_logits, batch["pids"]) * self.args.id_loss_weight
            ret["img_acc"] = (torch.argmax(image_logits, dim=1) == batch["pids"]).float().mean()
            ret["txt_acc"] = (torch.argmax(text_logits, dim=1) == batch["pids"]).float().mean()

        if "mlm" in self.current_task:
            mlm_ids = batch["mlm_ids"]
            mlm_feats = self.base_model.encode_text(mlm_ids)
            x = self.cross_former(mlm_feats, image_feats, image_feats)
            x = self.mlm_head(x)
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch["mlm_labels"].reshape(-1)
            ret["mlm_loss"] = objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight
            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            ret["mlm_acc"] = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()

        image_part_feats = self._pool_image_parts(image_feats)
        part_loss, per_part_losses = self._compute_part_alignment(image_part_feats, batch)
        ret["part_attr_loss"] = part_loss * self.args.part_loss_weight
        for key, value in per_part_losses.items():
            ret[key] = value * self.args.part_loss_weight

        return ret


def build_model_part_attr(args, num_classes=11003):
    model = IRRAPartAttrAlign(args, num_classes)
    convert_weights(model)
    return model
