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