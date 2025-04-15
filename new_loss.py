import torch
from torch import nn
import torch.nn.functional as F

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, ignore_index=0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B*T, V), targets: (B*T,)
        B_T, V = logits.shape

        valid_mask = (targets != self.ignore_index)

        if self.label_smoothing > 0:
            confidence = 1.0 - self.label_smoothing
            smoothed_labels = torch.full_like(logits, self.label_smoothing / (V - 1))
            smoothed_labels.scatter_(1, targets.unsqueeze(1), confidence)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smoothed_labels * log_probs).sum(dim=1)
        else:
            loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)

        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.sum()


class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None, label_smoothing=0.0, ignore_index=0):
        super().__init__()
        self.weights = weights or {'ce': 1.0}
        self.ce_loss = MaskedCrossEntropyLoss(label_smoothing, ignore_index)

    def forward(self, logits, targets):
        # logits: (B*T, V), targets: (B*T,)
        ce = self.ce_loss(logits, targets)

        vocab_size = logits.size(-1)
        unique_classes = torch.unique(targets[targets != self.ce_loss.ignore_index])
        class_weight_factor = vocab_size / max(len(unique_classes), 1)

        total_loss = self.weights['ce'] * ce * class_weight_factor
        return total_loss
