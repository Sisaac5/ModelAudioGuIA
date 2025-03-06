import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    # Separate frames and captions
    frames, captions = zip(*batch)
    
    # Stack frames
    frames = torch.stack(frames, dim=0)
    
    # Pad captions to the maximum length in the batch
    captions = pad_sequence(captions, batch_first=True, padding_value= 0)
    
    return frames, captions