import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from transformers import BertTokenizer
from dataset import VideoDataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from model import SceneDescriptionModel
from datetime import datetime
from new_loss import MultiTaskLoss

# Environment configurations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.backends.cudnn.logging = False

# Hyperparameters
INPUT_DIM = 512
HID_DIM = 512
N_LAYERS = 8
NHEAD = 16
BATCH_SIZE = 32
MAX_SEQ_LEN = 20
NUM_EPOCHS = 30
CLIP = 1
LR = 1e-4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
npy_root = '/home/arthur/tail/AudioGuIA/dataSet/Movies'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def generate_description(model, visual_features, timestamps, tokenizer, device, max_len=30):
    model.eval()
    with torch.no_grad():
        visual_features = visual_features.unsqueeze(0).to(device)
        timestamps = timestamps.unsqueeze(0).to(device)
        memory = model(visual_features, timestamps)
        
        generated = torch.tensor([[tokenizer.cls_token_id]], device=device)
        
        for i in range(max_len):
            text_embeddings = model.bert.embeddings(generated)
            text_embeddings = model.bert_proj(text_embeddings)
            text_embeddings = model.pos_encoder(text_embeddings)
            
            tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            output = model.text_decoder(
                tgt=text_embeddings,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            logits = model.output_layer(output[:, -1:, :])
            next_token = logits.argmax(-1)
            
            if next_token.item() == tokenizer.sep_token_id:
                break
                
            generated = torch.cat([generated, next_token], dim=1)
        
        return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

def evaluate_and_save(model, dataset, output_path, device):
    model.eval()
    results = []
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        frames, timestamps, text_ids = dataset[idx]  # Modified to include timestamps
        true_text = tokenizer.decode(text_ids.tolist(), skip_special_tokens=True)
        
        try:
            pred_text = generate_description(model, frames, timestamps, tokenizer, device)
            results.append({
                "id": idx,
                "true_description": true_text,
                "predicted_description": pred_text,
                "frames_path": dataset.dataframe.iloc[idx]['file_path']
            })
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Load and split data
    dataframe = pd.read_csv('data/annotations-someone_path_plus_gender_without_fantasy_selected_frames_20.csv')
    dataframe['selected_frames'] = dataframe['selected_frames'].apply(json.loads)
    dataframe= dataframe[dataframe['id']==10]
    # Split by movies
    movies = dataframe['movie'].unique()
    train_movies, test_movies = train_test_split(movies, test_size=TEST_RATIO, random_state=42)
    train_movies, val_movies = train_test_split(train_movies, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)
    
    train_df = dataframe[dataframe['movie'].isin(train_movies)].reset_index(drop=True)
    val_df = dataframe[dataframe['movie'].isin(val_movies)].reset_index(drop=True)
    test_df = dataframe[dataframe['movie'].isin(test_movies)].reset_index(drop=True)

    # Create datasets
    train_dataset = VideoDataset(
        dataframe=train_df,
        num_frames=20,
        npy_root=npy_root,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN
    )
    
    val_dataset = VideoDataset(
        dataframe=val_df,
        num_frames=20,
        npy_root=npy_root,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN
    )
    
    test_dataset = VideoDataset(
        dataframe=test_df,
        num_frames=20,
        npy_root=npy_root,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = SceneDescriptionModel(
        input_dim=INPUT_DIM,
        hidden_dim=HID_DIM,
        num_layers=N_LAYERS,
        nhead=NHEAD,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion = MultiTaskLoss(weights={'ce': 1.0, 'temp': 0.1}, label_smoothing=0.1, 
                              ignore_index=tokenizer.pad_token_id)
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for frames, timestamps, text in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            frames, timestamps, text = frames.to(device), timestamps.to(device), text.to(device)
            
            optimizer.zero_grad()
            logits = model(frames, timestamps, text)
            loss = criterion.forward(logits.reshape(-1, logits.size(-1)), text[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for frames, timestamps, text in val_loader:
                frames, timestamps, text = frames.to(device), timestamps.to(device), text.to(device)
                logits = model(frames, timestamps, text)
                loss = criterion.forward(logits.reshape(-1, logits.size(-1)), text[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print("Saved best model")

    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    
    # Test evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_path = os.path.join(output_dir, f'test_results_{timestamp}.json')
    evaluate_and_save(model, test_dataset, test_output_path, device)
    
    print(f"Test results saved to {test_output_path}")
    
    # Example generation
    sample_idx = 0
    sample_frames, sample_text = test_dataset[sample_idx]
    sample_true_text = tokenizer.decode(sample_text.tolist(), skip_special_tokens=True)
    sample_pred_text = generate_description(model, sample_frames, tokenizer, device)
    
    print("\nExample Generation:")
    print(f"True Description: {sample_true_text}")
    print(f"Predicted Description: {sample_pred_text}")