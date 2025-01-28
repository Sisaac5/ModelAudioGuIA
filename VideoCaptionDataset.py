import os
import torch
from torch.utils.data import Dataset
import numpy as np


class VideoCaptionDataset(Dataset):
    def __init__(self, root_dir, df, movies , vocab, transform=None, num_frames=10):
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
        self.filter_df = df[df['movie'].isin(movies)]

        self.captions = self.filter_df["text"].reset_index(drop=True)
        self.vocab = vocab
        self.num_frames = num_frames
        

    def __len__(self):
        return len(self.filter_df)

    def get_frames(self, movie, start, end):
        movie_name = os.path.join(self.root_dir, f'{movie}.npy')

        df = np.load(movie_name)
        rows, cols = df.shape
        index_frame_start = int(start // 0.2)
        index_frame_end = int(end // 0.2)

        if index_frame_start > rows:
          new_quant_frames = index_frame_end - index_frame_start
          index_frame_start = rows - new_quant_frames
          index_frame_end = rows

          return df[index_frame_start:index_frame_end]

        elif index_frame_start == index_frame_end:
          return df[index_frame_start - 1:index_frame_end]

        return df[index_frame_start:index_frame_end]

    def __getitem__(self, idx):

        video_name = self.filter_df.iloc[idx]['movie']
        start = self.filter_df.iloc[idx]['start']
        end = self.filter_df.iloc[idx]['end']
        frames = self.get_frames(video_name, start, end)

        quant_frames = len(frames)

        ##TODO: usar clip
        
        frame_indices = torch.linspace(0, len(frames) - 1, steps=self.num_frames).long()
        selected_frames = frames[frame_indices]
        #Combinar frames em um tensor
        video_tensor = torch.tensor(selected_frames, dtype=torch.float32).unsqueeze(0)  # (num_frames, C, H, W)
           
        #Parar de retorna a média dos frames
        #video_tensor = torch.mean(video_tensor, dim=1)

        # Processar a legenda
        caption = self.captions.iloc[idx]
        caption_vec = [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return video_tensor[0], torch.tensor(caption_vec)