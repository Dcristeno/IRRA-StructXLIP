from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import objectives
from .clip_model import LayerNorm, QuickGELU, Transformer, build_CLIP_from_openai_pretrained, convert_weights


class IRRAStructXLIP(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size
        )
        self.embed_dim = base_cfg["embed_dim"]
        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.register_buffer("struct_logit_scale", torch.tensor(1.0 / args.struct_temperature, dtype=torch.float32))

        if "id" in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if "mlm" in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim,
                self.embed_dim // 64,
                batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim,
                layers=args.cmt_depth,
                heads=self.embed_dim // 64
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
                OrderedDict([
                    ("dense", nn.Linear(self.embed_dim, self.embed_dim)),
                    ("gelu", QuickGELU()),
                    ("ln", LayerNorm(self.embed_dim)),
                    ("fc", nn.Linear(self.embed_dim, args.vocab_size))
                ])
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
            need_weights=False
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

    def _mask_special_tokens(self, token_ids):
        valid = token_ids.ne(0)
        if valid.size(1) > 0:
            valid[:, 0] = False
        eot_indices = token_ids.argmax(dim=-1)
        for row, idx in enumerate(eot_indices.tolist()):
            valid[row, idx] = False
        return valid

    def _compute_struct_itc(self, image_feats, text_feats, logit_scale, pids):
        image_norm = F.normalize(image_feats, dim=-1)
        text_norm = F.normalize(text_feats, dim=-1)
        logits_per_image = logit_scale * image_norm @ text_norm.t()
        logits_per_text = logits_per_image.t()
        pid_matrix = pids.view(-1, 1)
        pos = torch.eq(pid_matrix, pid_matrix.t()).float()
        labels = pos / pos.sum(dim=1, keepdim=True)
        loss_i = -(F.log_softmax(logits_per_image, dim=1) * labels).sum(dim=1).mean()
        loss_t = -(F.log_softmax(logits_per_text, dim=1) * labels).sum(dim=1).mean()
        return (loss_i + loss_t) / 2

    def _compute_struct_local(self, edge_tokens, struct_text_tokens, struct_caption_ids):
        edge_tokens = F.normalize(edge_tokens[:, 1:, :].float(), dim=-1)
        struct_text_tokens = F.normalize(struct_text_tokens.float(), dim=-1)
        sim = torch.matmul(struct_text_tokens, edge_tokens.transpose(1, 2))
        token_best = sim.max(dim=-1).values
        valid_tokens = self._mask_special_tokens(struct_caption_ids)
        if valid_tokens.sum() == 0:
            return token_best.new_tensor(0.0)
        token_best = token_best.masked_fill(~valid_tokens, 0.0)
        denom = valid_tokens.sum(dim=1).clamp(min=1)
        return ((1.0 - token_best).sum(dim=1) / denom).mean()

    def _compute_struct_consistency(self, image_feats, edge_feats):
        image_feats = F.normalize(image_feats, dim=-1)
        edge_feats = F.normalize(edge_feats, dim=-1)
        return (1.0 - (image_feats * edge_feats).sum(dim=-1)).mean()

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

        edge_images = batch["edge_images"]
        struct_caption_ids = batch["struct_caption_ids"]
        edge_feats_all, struct_text_feats_all = self.base_model(edge_images, struct_caption_ids)
        edge_cls = edge_feats_all[:, 0, :].float()
        struct_text_cls = struct_text_feats_all[
            torch.arange(struct_text_feats_all.shape[0]),
            struct_caption_ids.argmax(dim=-1)
        ].float()

        if self.args.struct_loss_type == "sdm":
            ret["struct_global_loss"] = objectives.compute_sdm(
                edge_cls,
                struct_text_cls,
                batch["pids"],
                self.struct_logit_scale
            ) * self.args.struct_global_weight
        else:
            ret["struct_global_loss"] = self._compute_struct_itc(
                edge_cls,
                struct_text_cls,
                self.struct_logit_scale,
                batch["pids"]
            ) * self.args.struct_global_weight

        ret["struct_consistency_loss"] = self._compute_struct_consistency(i_feats, edge_cls) * self.args.struct_consistency_weight

        if self.args.struct_local_weight > 0:
            ret["struct_local_loss"] = self._compute_struct_local(
                edge_feats_all,
                struct_text_feats_all,
                struct_caption_ids
            ) * self.args.struct_local_weight

        return ret


def build_model_structxlip(args, num_classes=11003):
    model = IRRAStructXLIP(args, num_classes)
    convert_weights(model)
    return model
