import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from transformers import BertTokenizer
from model import EncoderLSTM, DecoderLSTM, VideoCaptioningModel
from dataset import VideoDataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

# Environment configurations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.backends.cudnn.logging = False

# Hyperparameters
INPUT_DIM = 512
HID_DIM = 512
N_LAYERS = 10
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
BATCH_SIZE = 32
MAX_SEQ_LEN = 20
NUM_EPOCHS = 20
CLIP = 1
LR = 1e-3
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

npy_root = '/home/arthur/tail/AudioGuIA/dataSet/Movies'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for frames, input_ids in progress_bar:
        frames = frames.to(device)
        input_ids = input_ids.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames, input_ids)
        
        outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
        targets = input_ids[:, 1:].reshape(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for frames, input_ids in progress_bar:
            frames = frames.to(device)
            input_ids = input_ids.to(device)
            
            outputs = model(frames, input_ids)
            
            outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            targets = input_ids[:, 1:].reshape(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, tokenizer, device, max_len=20):
    model.eval()
    all_references = []
    all_predictions = []
    cos_model = SentenceTransformer('all-MiniLM-L6-v2')
    results = {}
    index = 0
    with torch.no_grad():
        for frames, input_ids in dataloader:
            frames = frames.to(device)
            input_ids = input_ids.to(device)
            
            # Generate predictions
            encoder_outputs, hidden, cell = model.encoder(frames)
            predictions = input_ids[:, 0].unsqueeze(1)
            
            for _ in range(max_len-1):
                output, hidden, cell = model.decoder(predictions[:, -1].unsqueeze(1), 
                                     encoder_outputs, hidden, cell)
                next_token = output.argmax(1)
                predictions = torch.cat((predictions, next_token.unsqueeze(1)), dim=1)
            
            # Decode predictions and references
            pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) 
                          for ids in predictions.cpu().numpy()]
            ref_texts = [tokenizer.decode(ids, skip_special_tokens=True) 
                         for ids in input_ids.cpu().numpy()]
            
            text_pairs = {}
        
            for i, (pred, ref) in enumerate(zip(pred_texts, ref_texts), 1):
              text_pairs[f'text_{i}'] = {
                  'pred': pred,
                  'ref': ref
              }

            results[f'{index}'] = text_pairs
            index += 1

            # Tokenize for METEOR
            pred_texts_tokenized = [word_tokenize(pred) for pred in pred_texts]
            ref_texts_tokenized = [[word_tokenize(ref)] for ref in ref_texts]
            
            all_predictions.extend(pred_texts_tokenized)
            all_references.extend(ref_texts_tokenized)
    
    # Calculate metrics
    try:
        bleu4 = corpus_bleu(all_references, all_predictions)
        meteor = np.mean([meteor_score(ref, pred) for ref, pred in zip(all_references, all_predictions)])
        
        # For BERTScore and cosine similarity, we need the original strings
        pred_strings = [' '.join(pred) for pred in all_predictions]
        ref_strings = [' '.join(ref[0]) for ref in all_references]
        
        # BERTScore
        P, R, F1 = bert_score(pred_strings, ref_strings, lang='en')
        
        # Cosine similarity
        pred_embeds = cos_model.encode(pred_strings)
        ref_embeds = cos_model.encode(ref_strings)
        cosine_sim = np.diag(np.matmul(pred_embeds, ref_embeds.T))
        
        #save results
        with open('models/results.json', 'w') as f:
            json.dump(results, f, indent=4)
        return {
            'BLEU-4': bleu4,
            'METEOR': meteor,
            'BERTScore-F1': F1.mean().item(),
            'Cosine': np.mean(cosine_sim)
        }
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

if __name__ == "__main__":
    # Load and split data
    dataframe = pd.read_csv('data/annot-harry_potter-someone_path_plus_gender_selected_frames_20.csv')
    dataframe['selected_frames'] = dataframe['selected_frames'].apply(json.loads)
    
    # Split by movies
    movies = dataframe['movie'].unique()
    train_movies, test_movies = train_test_split(movies, test_size=TEST_RATIO, random_state=42)
    train_movies, val_movies = train_test_split(train_movies, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)
    
    train_df = dataframe[dataframe['movie'].isin(train_movies)].reset_index(drop=True)
    val_df = dataframe[dataframe['movie'].isin(val_movies)].reset_index(drop=True)
    test_df = dataframe[dataframe['movie'].isin(test_movies)].reset_index(drop=True)

    # Create datasets and dataloaders
    dataset_train = VideoDataset(train_df, num_frames=20, 
                               npy_root=npy_root,
                               tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
    dataset_val = VideoDataset(val_df, num_frames=20,
                             npy_root=npy_root,
                             tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
    dataset_test = VideoDataset(test_df, num_frames=20,
                              npy_root=npy_root,
                              tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)

    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    encoder = EncoderLSTM(input_size=INPUT_DIM, 
                          hidden_size=HID_DIM,
                          num_layers=N_LAYERS,
                          dropout=ENC_DROPOUT)
    decoder = DecoderLSTM(vocab_size=tokenizer.vocab_size, 
                          embed_size=HID_DIM,
                          hidden_size=HID_DIM,
                          encoder_hidden_size=HID_DIM,
                          num_layers=N_LAYERS,
                          dropout=DEC_DROPOUT)
    
    model = VideoCaptioningModel(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pt')

    # Final evaluation
    model.load_state_dict(torch.load('models/best_model.pt'))
    test_metrics = evaluate(model, test_loader, tokenizer, device)
    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
