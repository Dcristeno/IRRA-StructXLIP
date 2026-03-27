from collections import OrderedDict

import torch
import torch.nn as nn

from misc.multi_vector_utils import bidirectional_late_interaction, compute_distribution_matching_from_logits
from model import objectives
from .clip_model import LayerNorm, QuickGELU, Transformer, build_CLIP_from_openai_pretrained, convert_weights


class LearnedTokenAggregator(nn.Module):
    def __init__(self, embed_dim, num_vectors, num_heads):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_vectors, embed_dim) * (embed_dim ** -0.5))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_q = LayerNorm(embed_dim)
        self.ln_kv = LayerNorm(embed_dim)
        self.ln_attn = LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(embed_dim, embed_dim * 4)),
                    ("gelu", QuickGELU()),
                    ("fc2", nn.Linear(embed_dim * 4, embed_dim)),
                ]
            )
        )
        self.ln_ffn = LayerNorm(embed_dim)

    def forward(self, tokens, key_padding_mask=None):
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1).to(tokens.dtype)
        attended = self.attn(
            self.ln_q(queries),
            self.ln_kv(tokens),
            self.ln_kv(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.ln_attn(queries + attended)
        x = self.ln_ffn(x + self.ffn(x))
        return x


class IRRAMultiVector(nn.Module):
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

        self.image_local_aggregator = LearnedTokenAggregator(
            self.embed_dim,
            args.num_local_vectors,
            args.multi_vector_heads,
        )
        self.text_local_aggregator = LearnedTokenAggregator(
            self.embed_dim,
            args.num_local_vectors,
            args.multi_vector_heads,
        )

    def _set_task(self):
        self.current_task = [name.strip() for name in self.args.loss_names.split("+")]
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

    def _encode_image_tokens(self, image):
        return self.base_model.encode_image(image)

    def _encode_text_tokens(self, text):
        return self.base_model.encode_text(text)

    def _extract_global_text(self, text_tokens, caption_ids):
        return text_tokens[torch.arange(text_tokens.shape[0], device=caption_ids.device), caption_ids.argmax(dim=-1)].float()

    def _extract_local_text(self, text_tokens, caption_ids):
        key_padding_mask = caption_ids.eq(0)
        return self.text_local_aggregator(text_tokens, key_padding_mask=key_padding_mask).float()

    def _extract_local_image(self, image_tokens):
        return self.image_local_aggregator(image_tokens[:, 1:, :]).float()

    def encode_image(self, image):
        image_tokens = self._encode_image_tokens(image)
        return image_tokens[:, 0, :].float()

    def encode_text(self, text):
        text_tokens = self._encode_text_tokens(text)
        return self._extract_global_text(text_tokens, text)

    def encode_image_multi_vector(self, image):
        image_tokens = self._encode_image_tokens(image)
        return image_tokens[:, 0, :].float(), self._extract_local_image(image_tokens)

    def encode_text_multi_vector(self, text):
        text_tokens = self._encode_text_tokens(text)
        return self._extract_global_text(text_tokens, text), self._extract_local_text(text_tokens, text)

    def _compute_multi_vector_loss(self, text_local_feats, image_local_feats, pids):
        late_interaction_logits = self.logit_scale * bidirectional_late_interaction(text_local_feats, image_local_feats)
        return compute_distribution_matching_from_logits(late_interaction_logits, pids)

    def forward(self, batch):
        ret = {}

        images = batch["images"]
        caption_ids = batch["caption_ids"]
        image_tokens, text_tokens = self.base_model(images, caption_ids)
        image_global_feats = image_tokens[:, 0, :].float()
        text_global_feats = self._extract_global_text(text_tokens, caption_ids)
        image_local_feats = self._extract_local_image(image_tokens)
        text_local_feats = self._extract_local_text(text_tokens, caption_ids)

        ret["temperature"] = 1 / self.logit_scale

        if "itc" in self.current_task:
            ret["itc_loss"] = objectives.compute_itc(image_global_feats, text_global_feats, self.logit_scale)

        if "sdm" in self.current_task:
            ret["sdm_loss"] = objectives.compute_sdm(
                image_global_feats,
                text_global_feats,
                batch["pids"],
                self.logit_scale,
            )

        if "cmpm" in self.current_task:
            ret["cmpm_loss"] = objectives.compute_cmpm(image_global_feats, text_global_feats, batch["pids"])

        if "id" in self.current_task:
            image_logits = self.classifier(image_global_feats.half()).float()
            text_logits = self.classifier(text_global_feats.half()).float()
            ret["id_loss"] = objectives.compute_id(image_logits, text_logits, batch["pids"]) * self.args.id_loss_weight
            ret["img_acc"] = (torch.argmax(image_logits, dim=1) == batch["pids"]).float().mean()
            ret["txt_acc"] = (torch.argmax(text_logits, dim=1) == batch["pids"]).float().mean()

        if "mlm" in self.current_task:
            mlm_ids = batch["mlm_ids"]
            mlm_feats = self.base_model.encode_text(mlm_ids)
            x = self.cross_former(mlm_feats, image_tokens, image_tokens)
            x = self.mlm_head(x)
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch["mlm_labels"].reshape(-1)
            ret["mlm_loss"] = objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight
            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            ret["mlm_acc"] = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()

        ret["multi_vector_loss"] = self._compute_multi_vector_loss(
            text_local_feats,
            image_local_feats,
            batch["pids"],
        ) * self.args.multi_vector_loss_weight

        return ret


def build_model_multi_vector(args, num_classes=11003):
    model = IRRAMultiVector(args, num_classes)
    convert_weights(model)
    return model
