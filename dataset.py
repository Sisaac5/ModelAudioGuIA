import os
import torch
from torch.utils.data import Dataset
import numpy as np


class VideoCaptionDataset(Dataset):
    def __init__(self, root_dir, df, tokenizer ,transform=None, num_frames=10, max_lenght=20):
        """
        Dataset para Video Captioning.
        Args:
            root_dir (str): Diretório raiz contendo os vídeos ou diretórios de frames.
            df (DataFrame): DataFrame com colunas 'video' (caminhos dos vídeos) e 'caption' (legendas).
            vocab (obj): Vocabulário com métodos `stoi` e `numericalize`.
            transform (callable, optional): Transformações aplicadas aos frames.
            num_frames (int): Número de frames a serem carregados por vídeo.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = df
        self.captions = self.df["description"].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.max_length = max_lenght

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        path_movie_clips = self.df.iloc[idx]['file_path']
        frames = np.load(os.path.join(self.root_dir, path_movie_clips), allow_pickle=True)
  

        if(len(frames)==self.num_frames):
            selected_frames = frames
        elif(len(frames)<self.num_frames):
            selected_frames = np.zeros((self.num_frames, *frames.shape[1:]))
            selected_frames[:len(frames)] = frames
        else:
            selected_frames = frames[self.df.iloc[idx]['selected_frames']]
              
        video_tensor = torch.tensor(selected_frames, dtype=torch.float32).unsqueeze(0)  # (num_frames, C, H, W)

        # Processar a legenda
        caption = self.captions.iloc[idx]
        caption_tokens = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
     
        return video_tensor[0].squeeze(1), caption_tokens['input_ids'].squeeze(0)