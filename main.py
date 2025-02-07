import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from VideoCaptionDataset import VideoCaptionDataset
from model import EncoderRNN, DecoderRNN, S2VTAttModel
from utils import CapsCollate
import pandas as pd
from vocabulary import Vocabulary
from torchvision import transforms as T
from rouge_score import rouge_scorer



if __name__ == "__main__":
        
    data_path = './data'
    captions_csv = 'mad-v2-ad-unnamed.csv' # troca pelo test se precisar
    npy_path = os.path.join(data_path, 'clips')

    ###HYPERPARAMETERS
    BATCH_SIZE = 32
    EMBED_SIZE = 300
    ATTENTION_DIM = 512
    ENCODER_DIM = 512
    DECODER_DIM = 512
    TEMPORAL_DIM = 512
    NUM_LAYERS = 2
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 1
    NUM_FRAMES = 10

    # Load dataframes
    df = pd.read_csv(os.path.join(data_path, captions_csv))
    df = df.loc[df['movie'].isin([int(file.split('.')[0]) for file in os.listdir(npy_path)])]
    
    # Split data
    movies = df['movie'].unique()
    
    np.random.seed(42)
    shuffled_movies = np.random.permutation(movies)

    n = len(shuffled_movies)
    train_size = int(0.6 * n)
    test_size = val_size = int(0.2 * n)

    train_movies = shuffled_movies[:train_size]
    test_movies = shuffled_movies[train_size:train_size + test_size]
    val_movies = shuffled_movies[train_size + test_size:]
    
    captions_train = df[df['movie'].isin(train_movies)]['text'].tolist()
    # Create a vocabulary
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocab(captions_train)
    os.makedirs(os.path.join(data_path, 'vocab'), exist_ok=True)
    vocab.save_vocab(path=os.path.join(data_path,'vocab','vocab.json'))

    # #Carregar vocabulário
    # vocab = Vocabulary.load_vocab(os.path.join(data_path,'vocab','vocab.json'))

    transforms = T.Compose([
        T.ToTensor()
    ])

    # Create datasets
    train_dataset = VideoCaptionDataset(npy_path, 
                                        df, train_movies,
                                        vocab,
                                        transform=transforms,
                                        num_frames=NUM_FRAMES
                                        )
    test_dataset = VideoCaptionDataset(npy_path, 
                                    df, 
                                    test_movies,
                                    vocab, 
                                    transform=transforms,
                                        num_frames=NUM_FRAMES
                                    )
    val_dataset = VideoCaptionDataset(npy_path, 
                                    df, 
                                    val_movies,
                                    vocab, 
                                    transform=transforms,
                                        num_frames=NUM_FRAMES
                                    )

    pad_idx = vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = EncoderRNN(
        dim_vid=512,
        dim_hidden=10,
        n_layers=NUM_LAYERS,
        bidirectional=False,
        rnn_cell='lstm'
    ).to(device)

    decoder = DecoderRNN(
        vocab_size=len(vocab),
        max_len=20,
        dim_hidden=10,
        dim_word=EMBED_SIZE,
        n_layers=NUM_LAYERS,
        rnn_cell='lstm',
        bidirectional=False,
        input_dropout_p=0.1,
        rnn_dropout_p=0.1
    ).to(device)

    model = S2VTAttModel(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} - Training")

        for batch_idx, (images, captions) in train_loader_tqdm:
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images, captions)
            targets = captions[:, 1:]
        
            loss = criterion(outputs.view(-1, len(vocab)), targets.reshape(-1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"\nEpoch {epoch} - Average Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        total_loss = 0
        all_rouge_scores = []
        epoch_dir = os.path.join('./models', f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        if epoch == 0:
            os.makedirs("./models/best", exist_ok=True)

        torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pth"))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        pred_vec, pred_vec_text, alp = [], [], []
        test_loader_tqdm = tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch} - Validation")
        
        with torch.no_grad():
            for images, captions in test_loader_tqdm:
                images, captions = images.to(device), captions.to(device)

                outputs, alph = model(images, captions)
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, len(vocab)), targets.reshape(-1))
                total_loss += loss.item()

                # Update tqdm description
                test_loader_tqdm.set_postfix(Loss=loss.item())

                for i, image in enumerate(images):
                  predicted_caption, seq_preds = model(image.unsqueeze(0), captions, mode='inference')
                  predicted_caption_text = " ".join([vocab.itos[token.item()] for token in seq_preds[0]])
                  
                  alp.append(alph)

                  gt_caption_text = ' '.join([
                      vocab.itos[token.item()]
                      for token in captions[i] if token.item() in vocab.itos
                  ])

                  rouge_score = scorer.score(gt_caption_text, predicted_caption_text)
                  all_rouge_scores.append(rouge_score)
                  pred_vec.append(predicted_caption)
                  pred_vec_text.append(predicted_caption_text)
            
        avg_rouge = {
            "rouge1": np.mean([score["rouge1"].fmeasure for score in all_rouge_scores]),
            "rouge2": np.mean([score["rouge2"].fmeasure for score in all_rouge_scores]),
            "rougeL": np.mean([score["rougeL"].fmeasure for score in all_rouge_scores]),
        }

        print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader):.4f}")
        print(f"ROUGE Scores: {avg_rouge}")