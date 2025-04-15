import torch
from torch import nn
import torch.nn.functional as F

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, ignore_index=0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, T, V), targets: (B, T)
        B, T, V = logits.shape
        logits = logits.view(-1, V)
        targets = targets.view(-1)

        if self.label_smoothing > 0:
            # Label smoothing
            confidence = 1.0 - self.label_smoothing
            smoothed_labels = torch.full_like(logits, self.label_smoothing / (V - 1))
            smoothed_labels.scatter_(1, targets.unsqueeze(1), confidence)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smoothed_labels * log_probs).sum(dim=1)
        else:
            loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)

        loss = loss.view(B, T)
        mask = (targets.view(B, T) != self.ignore_index).float()
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()

class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, memory):
        # memory: (B, T, D)
        sim = self.cos(memory[:, :-1], memory[:, 1:])
        return 1 - sim.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None, label_smoothing=0.0, ignore_index=0):
        super().__init__()
        self.weights = weights or {'ce': 1.0, 'temp': 0.1}
        self.ce_loss = MaskedCrossEntropyLoss(label_smoothing, ignore_index)
        self.temp_loss = TemporalConsistencyLoss()

    def forward(self, logits, targets, memory=None):
        ce = self.ce_loss(logits, targets)
        #temp = self.temp_loss(memory)

        # ponderar por número de classes no batch
        vocab_size = logits.size(-1)
        classes_in_batch = torch.unique(targets[targets != 0])
        class_weight_factor = vocab_size / max(len(classes_in_batch), 1)

        #total_loss = self.weights['ce'] * ce + self.weights['temp'] * temp * class_weight_factor
        total_loss = self.weights['ce'] * ce * class_weight_factor
        
        return total_loss
