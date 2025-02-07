import os
import clip
import numpy as np
import pandas as pd

from PIL import Image

# Carregar os filmes
data_path = './data'
captions_csv = 'mad-v2-ad-unnamed.csv'
npy_path = os.path.join(data_path, 'clips')

df = pd.read_csv(os.path.join(data_path, captions_csv))
df = df.loc[df['movie'].isin([int(file.split('.')[0]) for file in os.listdir(npy_path)])]

# Salvar uma coluna nova 'movie_clip' e criar um csv novo
df['movie_clip'] = df.apply(lambda row: f"{row['movie']}_{row.name}", axis=1)
df.to_csv("./data/mad-v2-ad-unnamed-plus.csv", index=False)

# Filtrar o CSV


'''
1) Saber os files que está em /clips

2) Carregar o csv unnamed com os filmes

2.5) df['movie_clip'] = df.apply(lambda row: f"{row['movies']}_{row.name}", axis=1)
2.6) Salvar csv ( nome diferente )

3) Percorrer os files em /clips ( for filme in os.listdir(data/clips))
    4) Filtrar o csv unnamed ( df.loc[df['movies'] == filme] )
    5) Percorrer cada linha do csv do passo 4 ( for i, row in filter_df.iterrows() )
        6) Chamar get_frames(movie_name, row['start'], row['end'])
        7) Passar pelo CLIP (10 frames)
        8) Salvar a matriz resultante ( np.save(row['movie_clip'] + ".npy"), matriz_clip )
        
        
        
        
    for batch in train_loader:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[1])
        break
    exit()
'''

def get_frames(movie, start, end):
    movie_name = os.path.join(npy_path, f'{movie}.npy')

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