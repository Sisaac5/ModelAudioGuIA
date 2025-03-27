import os
import torch
import pandas as pd
from dataset import VideoCaptionDataset
from model import VideoCaptioningModel, EncoderRNN, DecoderRNN
from torchvision import transforms as T
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
model_path = "/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models/best/best_model.pth"

### HYPERPARAMETERS
FEATURES_DIM = 512
NUM_FRAMES = 20
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_LENGTH = 10

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
vocab_size = tokenizer.vocab_size

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model
encoder = EncoderRNN(FEATURES_DIM, HIDDEN_SIZE, n_layers=NUM_LAYERS, rnn_dropout_p=DROPOUT)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, vocab_size, num_layers=NUM_LAYERS, rnn_dropout_p=DROPOUT)
model = VideoCaptioningModel(encoder, decoder).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Define transformations
transforms = T.Compose([
    T.ToTensor()
])

#npy path
npy_path = '/home/arthur/tail/AudioGuIA/dataSet/Movies/0017_Pianist/0017_Pianist_00.02.06.770-00.02.11.041/0017_Pianist_00.02.06.770-00.02.11.041_ViT-B_32.npy'
frames = np.load(npy_path)
print(frames.shape)
selected_frames = frames.squeeze(1)

video_tensor = torch.tensor(selected_frames, dtype=torch.float32)

# Perform inference item by item
with torch.no_grad():
  video_tensor = video_tensor.to(device).unsqueeze(0)
  _, hidden = model.encoder(video_tensor)
  start_token_idx = tokenizer.cls_token_id
  predicted_caption = model.decoder.inference(hidden, max_len=30, start_token_idx=start_token_idx, device=device)
  predicted_caption_text = tokenizer.decode(predicted_caption[0], skip_special_tokens=True)

print(predicted_caption_text)