import logging

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

from misc.multi_vector_utils import bidirectional_late_interaction
from utils.metrics import rank


class EvaluatorMultiVector:
    def __init__(self, img_loader, txt_loader, global_score_weight=0.3, local_score_weight=1.0, chunk_size=256):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.global_score_weight = global_score_weight
        self.local_score_weight = local_score_weight
        self.chunk_size = chunk_size
        self.logger = logging.getLogger("IRRA.eval")
        self.latest_result = {}

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids = [], []
        qglobal, qlocal, gglobal, glocal = [], [], [], []

        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                global_feat, local_feat = model.encode_text_multi_vector(caption)
            qids.append(pid.view(-1))
            qglobal.append(global_feat.cpu())
            qlocal.append(local_feat.cpu())

        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                global_feat, local_feat = model.encode_image_multi_vector(img)
            gids.append(pid.view(-1))
            gglobal.append(global_feat.cpu())
            glocal.append(local_feat.cpu())

        return (
            torch.cat(qglobal, dim=0),
            torch.cat(qlocal, dim=0),
            torch.cat(gglobal, dim=0),
            torch.cat(glocal, dim=0),
            torch.cat(qids, dim=0),
            torch.cat(gids, dim=0),
        )

    def _compute_local_similarity(self, query_local, gallery_local):
        num_query = query_local.shape[0]
        num_gallery = gallery_local.shape[0]
        local_similarity = torch.empty(num_query, num_gallery, device=query_local.device)

        for q_start in range(0, num_query, self.chunk_size):
            q_end = min(q_start + self.chunk_size, num_query)
            q_chunk = query_local[q_start:q_end]
            for g_start in range(0, num_gallery, self.chunk_size):
                g_end = min(g_start + self.chunk_size, num_gallery)
                g_chunk = gallery_local[g_start:g_end]
                local_similarity[q_start:q_end, g_start:g_end] = bidirectional_late_interaction(q_chunk, g_chunk)

        return local_similarity

    def eval(self, model, i2t_metric=False):
        qglobal, qlocal, gglobal, glocal, qids, gids = self._compute_embedding(model)
        device = next(model.parameters()).device

        qglobal = F.normalize(qglobal.to(device), p=2, dim=1)
        gglobal = F.normalize(gglobal.to(device), p=2, dim=1)
        qlocal = F.normalize(qlocal.to(device), p=2, dim=-1)
        glocal = F.normalize(glocal.to(device), p=2, dim=-1)

        global_similarity = qglobal @ gglobal.t()
        local_similarity = self._compute_local_similarity(qlocal, glocal)
        similarity = self.global_score_weight * global_similarity + self.local_score_weight * local_similarity

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(["t2i", t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                similarity=similarity.t(),
                q_pids=gids,
                g_pids=qids,
                max_rank=10,
                get_mAP=True,
            )
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(["i2t", i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info("\n" + str(table))

        self.latest_result = {
            "t2i_R1": float(t2i_cmc[0]),
            "t2i_R5": float(t2i_cmc[4]),
            "t2i_R10": float(t2i_cmc[9]),
            "t2i_mAP": float(t2i_mAP),
            "t2i_mINP": float(t2i_mINP),
            "eval_global_score_weight": float(self.global_score_weight),
            "eval_local_score_weight": float(self.local_score_weight),
        }
        if i2t_metric:
            self.latest_result.update(
                {
                    "i2t_R1": float(i2t_cmc[0]),
                    "i2t_R5": float(i2t_cmc[4]),
                    "i2t_R10": float(i2t_cmc[9]),
                    "i2t_mAP": float(i2t_mAP),
                    "i2t_mINP": float(i2t_mINP),
                }
            )

        return t2i_cmc[0]
