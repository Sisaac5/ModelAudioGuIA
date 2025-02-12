import os
import clip
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

# Carregar os filmes
data_path = './data'
captions_csv = 'mad-v2-ad-unnamed.csv'
npy_path = os.path.join(data_path, 'clips')

# Carregar dataset
df = pd.read_csv(os.path.join(data_path, captions_csv))
df = df.loc[df['movie'].isin([int(file.split('.')[0]) for file in os.listdir(npy_path)])]

# Salvar uma coluna nova 'movie_clip' e criar um CSV novo
df['movie_clip'] = df.apply(lambda row: f"{row['movie']}_{row.name}", axis=1)
df.to_csv("./data/mad-v2-ad-unnamed-plus.csv", index=False)

# Carregar o modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_frames(movie, start, end):
    movie_name = os.path.join(npy_path, f'{movie}.npy')
    df = np.load(movie_name, mmap_mode='r')
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

# Percorrer os arquivos de vídeos na pasta clips
for filme in tqdm(os.listdir(npy_path), desc="Processando filmes"):
    movie_id = int(filme.split('.')[0])
    movie_path = os.path.join(npy_path, filme)
    print(f"Processando filme: {movie_path}")
    filter_df = df.loc[df['movie'] == movie_id]
    
    for i, row in tqdm(filter_df.iterrows(), total=len(filter_df), desc=f"Processando {movie_id}"):
        frames = get_frames(movie_id, row['start'], row['end'])
        
        if frames.size == 0:
            continue
        
        # Obter embeddings do texto
        text_embedding = model.encode_text(clip.tokenize([row['text']], truncate=True).to(device))
        
        # Comparar embeddings dos frames com o texto
        frame_tensors = torch.from_numpy(frames).to(device)
        frame_tensors = frame_tensors.to(torch.float16)
        frame_similarities = (frame_tensors @ text_embedding.T).squeeze(1)
        best_frame_indices = np.argsort(frame_similarities.cpu().detach().numpy())[-10:]  # Seleciona os 10 melhores
        final_best_frames = frames[best_frame_indices]
        
        # Salvar os melhores frames finais
        np.save(os.path.join(npy_path, f"{row['movie_clip']}.npy"), final_best_frames)
