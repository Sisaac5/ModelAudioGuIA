import os
import torch
import pandas as pd
from dataset import VideoCaptionDataset
from model import VideoCaptioningModel, EncoderRNN, DecoderRNN
from torchvision import transforms as T
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

data_path = './data'
npy_path = os.path.join(data_path, 'clips')
model_path = "/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models/best/best_model.pth"

### HYPERPARAMETERS
FEATURES_DIM = 512
NUM_FRAMES = 10
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_LENGTH = 20
###

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

# Load test dataset
test_csv = 'mad-v2-ad-unnamed-plus.csv'  # Adjust if needed
df = pd.read_csv(os.path.join(data_path, test_csv))
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

test_dataset = VideoCaptionDataset(npy_path, df_test, tokenizer, transform=transforms, num_frames=NUM_FRAMES)

# Perform inference item by item
results = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        frames, _ = test_dataset[i]
        frames = frames.to(device).unsqueeze(0)  # Add batch dimension
        _, hidden = model.encoder(frames)
        start_token_idx = tokenizer.cls_token_id
        predicted_caption = model.decoder.inference(hidden, max_len=MAX_LENGTH, start_token_idx=start_token_idx, device=device)
        
        predicted_caption_text = tokenizer.decode(predicted_caption[0], skip_special_tokens=True)
        results.append({
            "video_id": df_test.iloc[i]['movie_clip'],
            "predicted_caption": predicted_caption_text
        })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(data_path, 'inference_results.csv'), index=False)
print("Inference results saved to inference_results.csv")
