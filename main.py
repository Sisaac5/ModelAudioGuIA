import os
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from dataset import VideoCaptionDataset
from model import VideoCaptioningModel, EncoderRNN, DecoderRNN
from utils import custom_collate_fn
import pandas as pd
from vocabulary import Vocabulary
from torchvision import transforms as T
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torchsummary import summary

if __name__ == "__main__":
        
    data_path = './data'
    captions_csv = 'mad-v2-ad-unnamed-plus.csv'  # Change to test if needed
    npy_path = os.path.join(data_path, 'clips')

    ### HYPERPARAMETERS
    FEATURES_DIM = 512
    NUM_FRAMES = 10
    BATCH_SIZE = 32
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30
    MAX_LENGTH = 20
    ###

    # Load dataframes
    df = pd.read_csv(os.path.join(data_path, captions_csv))

    # Dividir entre treino e temp (validação + teste)
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)

    # Dividir temp entre validação e teste
    #df_val, df_test = train_test_split(df_temp, test_size=0.1, random_state=42, shuffle=True)

    # Obter as legendas
    captions_train = df_train['text'].tolist()
    # Create a vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # vocab = Vocabulary(freq_threshold=1)
    # vocab.build_vocab(captions_train)
    # os.makedirs(os.path.join(data_path, 'vocab'), exist_ok=True)
    # vocab.save_vocab(path=os.path.join(data_path,'vocab','vocab.json'))

    # # Load vocabulary
    # vocab = Vocabulary.load_vocab(os.path.join(data_path,'vocab','vocab.json'))

    transforms = T.Compose([
        T.ToTensor()
    ])

    # Create datasets
    train_dataset = VideoCaptionDataset(npy_path, 
                                        df_train,
                                        tokenizer,
                                        transform=transforms,
                                        num_frames=NUM_FRAMES
                                        )
    test_dataset = VideoCaptionDataset(npy_path,
                                    df_test,
                                    tokenizer, 
                                    transform=transforms,
                                        num_frames=NUM_FRAMES
                                    )
    # val_dataset = VideoCaptionDataset(npy_path, 
    #                                 df_val,                                     
    #                                 tokenizer,
    #                                 transform=transforms,
    #                                     num_frames=NUM_FRAMES
    #                                 )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
    )
    
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=4,
    #     shuffle=False,
    # )
    
    vocab_size = tokenizer.vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderRNN(FEATURES_DIM,
                          HIDDEN_SIZE,
                          n_layers=NUM_LAYERS,
                          rnn_dropout_p=DROPOUT
                          )
    decoder = DecoderRNN(EMBED_SIZE, 
                         HIDDEN_SIZE,                          
                         vocab_size, 
                         num_layers=NUM_LAYERS,
                          rnn_dropout_p=DROPOUT
                         )

    model = VideoCaptioningModel(encoder, 
                                 decoder
                                 ).to(device)
    
    # summary(model,input_size=[(1, NUM_FRAMES, 512, 512), (BATCH_SIZE, MAX_LENGTH)])
    # exit()
    optimizer = optim.Adam(model.parameters(),
                            lr=LEARNING_RATE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    best_val_loss = float('inf')
    # Training loop
    for epoch in range(NUM_EPOCHS):
        loss_train = 0
        model.train()
        for frames, captions in tqdm(train_loader):
            optimizer.zero_grad()
            frames = frames.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs = model(frames, captions[:, :-1])
            
            # Compute loss
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
            loss_train += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        loss_train = loss_train / len(train_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

        # Test loop
        model.eval()
        results = []
        test_loss = 0
        with torch.no_grad():
            for frames, captions in tqdm(test_loader):
                frames = frames.to(device)
                captions = captions.to(device)
                
                # Get hidden state from the encoder
                _, hidden = model.encoder(frames)
                
                # Generate captions using the decoder
                start_token_idx = tokenizer.cls_token_id
                predicted_captions = model.decoder.inference(hidden, max_len=20, start_token_idx=start_token_idx, device=device)
                outputs = model(frames, captions[:, :-1])

                test_loss += criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1)).item()
                # Convert token indices to words
                for i in range(predicted_captions.size(0)):              
                    predicted_caption = tokenizer.decode(predicted_captions[i], skip_special_tokens=True)
                    results.append({
                        "video_id": test_dataset.df.iloc[i]['movie_clip'],
                        "predicted_caption": predicted_caption,
                        "true_caption": tokenizer.decode(captions[i], skip_special_tokens=True)
                    })
            test_loss /= len(test_loader)
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Test Loss: {test_loss:.4f}')    
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            torch.save(model.state_dict(), os.path.join("/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models/best", 'best_model.pth'))
            print("Model saved! Epoch: ", epoch)

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(data_path, f'test_results_epoch{epoch}.csv'), index=False)
        print("Test results saved to test_results.csv")