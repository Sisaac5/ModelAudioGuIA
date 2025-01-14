import torch
from torch.nn.utils.rnn import pad_sequence

class CapsCollate:
    def __init__(self, pad_idx, batch_first=True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        # Separar imagens e legendas
        imgs = [item[0].unsqueeze(0) for item in batch]
        captions = [item[1] for item in batch]

        # Empilhar imagens em um tensor (batch_size, 3, 224, 224)
        imgs = torch.cat(imgs, dim=0)

        # Preencher as legendas até o mesmo comprimento
        captions_padded = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.pad_idx)

        return imgs, captions_padded