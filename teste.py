import pandas as pd

df = pd.read_csv('/home/arthur/tail/AudioGuIA/ModelAudioGuIA/data/mad-v2-ad-unnamed-plus.csv')

#prin each row in the dataframe
for index, row in df.iterrows():
    row['text_unnamed'] = row['text_unnamed'].replace(',', '')
    row['text_unnamed'] = row['text_unnamed'].replace('.', '')

df.to_csv('/home/arthur/tail/AudioGuIA/ModelAudioGuIA/data/mad-v2-ad-unnamed-plus.csv', index=False)