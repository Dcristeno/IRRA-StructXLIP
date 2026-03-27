import torch
import torch.nn.functional as F


def bidirectional_late_interaction(text_local_feats, image_local_feats):
    text_local_feats = F.normalize(text_local_feats, p=2, dim=-1)
    image_local_feats = F.normalize(image_local_feats, p=2, dim=-1)

    similarity = torch.einsum("qld,gmd->qglm", text_local_feats, image_local_feats)
    text_to_image = similarity.max(dim=3).values.mean(dim=2)
    image_to_text = similarity.max(dim=2).values.mean(dim=2)
    return 0.5 * (text_to_image + image_to_text)


def compute_distribution_matching_from_logits(logits, pid, epsilon=1e-8):
    batch_size = logits.shape[0]
    pid = pid.reshape(batch_size, 1)
    labels = (pid - pid.t() == 0).float()
    labels_distribution = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)

    image_to_text_pred = F.softmax(logits.t(), dim=1)
    image_to_text_loss = image_to_text_pred * (
        F.log_softmax(logits.t(), dim=1) - torch.log(labels_distribution + epsilon)
    )

    text_to_image_pred = F.softmax(logits, dim=1)
    text_to_image_loss = text_to_image_pred * (
        F.log_softmax(logits, dim=1) - torch.log(labels_distribution + epsilon)
    )

    loss = torch.mean(torch.sum(image_to_text_loss, dim=1)) + torch.mean(torch.sum(text_to_image_loss, dim=1))
    return loss
