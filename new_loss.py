from torch import nn

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, targets):
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        
        # Calculate loss per token
        loss = self.ce(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss = loss.view(targets.shape)
        
        # Create mask (0 where target is padding, 1 otherwise)
        mask = (targets != 0).float()
        
        # Apply mask
        masked_loss = loss * mask
        
        # Return average over non-padded tokens
        return masked_loss.sum() / mask.sum()


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def forward(self, memory):
        """memory: (batch_size, seq_len, hidden_dim) from temporal encoder"""
        # Compare consecutive memory states
        seq1 = memory[:, :-1]
        seq2 = memory[:, 1:]
        
        # Calculate similarity between consecutive states
        sim = self.cos(seq1, seq2)
        
        # We want consecutive states to be similar
        loss = 1 - sim.mean()
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {'ce': 1.0, 'temp': 0.1}
        self.ce_loss = MaskedCrossEntropyLoss()
        self.temp_loss = TemporalConsistencyLoss()
        
    def forward(self, logits, targets, memory):
        ce = self.ce_loss(logits, targets)
        temp = self.temp_loss(memory)
        return self.weights['ce'] * ce + self.weights['temp'] * temp