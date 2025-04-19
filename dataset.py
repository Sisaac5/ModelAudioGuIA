import torch
from torch.utils.data import Dataset
import numpy as np
import os

class VideoDataset(Dataset):
    def __init__(self, dataframe, num_frames, npy_root, tokenizer, max_seq_len):
        self.dataframe = dataframe
        self.num_frames = num_frames
        self.npy_root = npy_root
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # read video features from .npy file
        frames = np.load(os.path.join(self.npy_root, self.dataframe['file_path'][idx]))
        # Pad frame features to max_seq_len
        if self.num_frames < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - self.num_frames, 512))
            frames = np.vstack((frames, pad))
        else:
            frames = frames[self.dataframe['selected_frames'][idx][:self.max_seq_len]]

        frames = torch.FloatTensor(frames)  # (seq_len, 512)

        max_frames = len(frames)
        timestamps = self.dataframe['selected_frames'][idx][:self.max_seq_len]
        normalized_timestamps = [ts / max_frames for ts in timestamps]
        timestamps = torch.tensor(normalized_timestamps, dtype=torch.float)
        #timestamps = torch.linspace(0, 1, self.num_frames)

        # Tokenize description
        desc = self.dataframe['description'][idx]
        tokens = self.tokenizer.encode_plus(
            desc,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)  # (max_seq_len)
        frames= frames.squeeze(1)  # (seq_len, 512)

        return frames,timestamps, input_ids