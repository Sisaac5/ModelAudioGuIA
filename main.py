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
import uuid

if __name__ == "__main__":
        
    data_path = './data'
    captions_csv = '/home/arthur/tail/AudioGuIA/ModelAudioGuIA/data/annot-harry_potter-someone_path_plus_gender.csv'  # Change to test if needed
    npy_path = os.path.join('/home/arthur/tail/AudioGuIA/dataSet', 'Movies')

    ### HYPERPARAMETERS
    FEATURES_DIM = 512
    NUM_FRAMES = 10
    BATCH_SIZE = 32
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    NUM_LAYERS = 8
    DROPOUT = 0.4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    MAX_LENGTH = 20

    # Load dataframes
    df = pd.read_csv(os.path.join(data_path, captions_csv))

    # Dividir entre treino e temp (validação + teste)
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)

    # Dividir temp entre validação e teste
    #df_val, df_test = train_test_split(df_temp, test_size=0.1, random_state=42, shuffle=True)

    # Obter as legendas
    captions_train = df_train['description'].tolist()
    # Create a vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

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
    #gerar um uuid para salvar o modelo
    uuid = uuid.uuid4()
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
                        "video_id": test_dataset.df.iloc[i]['clipe'],
                        "predicted_caption": predicted_caption,
                        "true_caption": tokenizer.decode(captions[i], skip_special_tokens=True)
                    })
            test_loss /= len(test_loader)
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Test Loss: {test_loss:.4f}') 
        os.makedirs(os.path.join('/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models', str(uuid)), exist_ok=True)
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            torch.save(model.state_dict(), os.path.join(f"/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models/{str(uuid)}", 'best_model.pth'))
            #save log with loss, epoch , uuid, hyperparameters
            with open(os.path.join(f"/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models/{str(uuid)}", 'log.txt'), 'w') as f:
                f.write(f'Best model saved at epoch {epoch+1} with loss: {best_val_loss}\n')
                f.write(f'UUID: {uuid}\n')
                f.write(f'Dataset: {captions_csv}\n')
                f.write(f'Hyperparameters:\n')
                f.write(f'FEATURES_DIM: {FEATURES_DIM}\n')
                f.write(f'NUM_FRAMES: {NUM_FRAMES}\n')
                f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
                f.write(f'EMBED_SIZE: {EMBED_SIZE}\n')
                f.write(f'HIDDEN_SIZE: {HIDDEN_SIZE}\n')
                f.write(f'NUM_LAYERS: {NUM_LAYERS}\n')
                f.write(f'DROPOUT: {DROPOUT}\n')
                f.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
                f.write(f'NUM_EPOCHS: {NUM_EPOCHS}\n')
                f.write(f'MAX_LENGTH: {MAX_LENGTH}\n')

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(f'/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models/{str(uuid)}', f'test_results_{epoch}.csv'), index=False)
        
        print("Test results saved to test_results.csv")