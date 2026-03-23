from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import objectives
from .clip_model import LayerNorm, QuickGELU, Transformer, build_CLIP_from_openai_pretrained, convert_weights


class IRRAColorAux(nn.Module):
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

        self.image_color_head = nn.Linear(self.embed_dim, args.color_num_classes)
        self.text_color_head = nn.Linear(self.embed_dim, args.color_num_classes)
        nn.init.normal_(self.image_color_head.weight.data, std=0.001)
        nn.init.constant_(self.image_color_head.bias.data, val=0.0)
        nn.init.normal_(self.text_color_head.weight.data, std=0.001)
        nn.init.constant_(self.text_color_head.bias.data, val=0.0)

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

    def _compute_color_loss(self, image_feats, text_feats, color_labels, has_color_label):
        valid = has_color_label > 0
        if not valid.any():
            zero = image_feats.new_tensor(0.0)
            return zero, zero, zero

        image_logits = self.image_color_head(image_feats[valid])
        text_logits = self.text_color_head(text_feats[valid])
        labels = color_labels[valid]

        image_loss = F.binary_cross_entropy_with_logits(image_logits, labels)
        text_loss = F.binary_cross_entropy_with_logits(text_logits, labels)
        loss = (image_loss + text_loss) / 2.0

        positive_mask = labels > 0
        if positive_mask.any():
            image_pos_acc = ((torch.sigmoid(image_logits) > self.args.color_score_thresh) == positive_mask).float()[positive_mask].mean()
            text_pos_acc = ((torch.sigmoid(text_logits) > self.args.color_score_thresh) == positive_mask).float()[positive_mask].mean()
        else:
            image_pos_acc = image_feats.new_tensor(0.0)
            text_pos_acc = image_feats.new_tensor(0.0)

        return loss, image_pos_acc, text_pos_acc

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

        color_loss, image_color_acc, text_color_acc = self._compute_color_loss(
            i_feats,
            t_feats,
            batch["color_labels"],
            batch["has_color_label"],
        )
        ret["color_loss"] = color_loss * self.args.color_loss_weight
        ret["img_color_acc"] = image_color_acc
        ret["txt_color_acc"] = text_color_acc

        return ret


def build_model_color(args, num_classes=11003):
    model = IRRAColorAux(args, num_classes)
    convert_weights(model)
    return model
