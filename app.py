import os
import torch
import clip
import numpy as np
import imageio.v3 as iio
import streamlit as st
import imageio
from PIL import Image
from torchvision import transforms as T
from transformers import BertTokenizer
from dataset import VideoCaptionDataset
from model import VideoCaptioningModel, EncoderRNN, DecoderRNN
import tempfile
import cv2


#Configuração do modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

#Configuração do modelo de Captioning
model_path = "/home/arthur/tail/AudioGuIA/ModelAudioGuIA/models/best/best_model_primeiro_dataset.pth"
FEATURES_DIM = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_LENGTH = 20

# Tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
vocab_size = tokenizer.vocab_size

# Carregar modelo de legendas
encoder = EncoderRNN(FEATURES_DIM, HIDDEN_SIZE, n_layers=NUM_LAYERS, rnn_dropout_p=DROPOUT)
decoder = DecoderRNN(FEATURES_DIM, HIDDEN_SIZE, vocab_size, num_layers=NUM_LAYERS, rnn_dropout_p=DROPOUT)
model = VideoCaptioningModel(encoder, decoder).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

def display_gif(gif_path):
    """Exibe um GIF animado no Streamlit usando HTML."""
    gif_html = f'<img src="data:image/gif;base64,{gif_to_base64(gif_path)}" style="width:100%;" loop>'
    st.markdown(gif_html, unsafe_allow_html=True)

def gif_to_base64(gif_path):
    """Converte um GIF para base64 para exibição na web."""
    import base64
    with open(gif_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Função para extrair frames do vídeo
def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        num_frames = total_frames  # Ajusta caso o vídeo tenha poucos frames
    
    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

# Função para converter os frames em embeddings CLIP e salvar como .npy
def process_frames(frames):
    frame_features = []
    
    for frame in frames:
        image = Image.fromarray(frame)
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input).cpu().numpy()
        
        frame_features.append(image_features)
    
    return np.stack(frame_features)

# Função para rodar inferência no modelo de legendas
def generate_caption(video_tensor):
    with torch.no_grad():
        _, hidden = model.encoder(video_tensor)
        start_token_idx = tokenizer.cls_token_id
        predicted_caption = model.decoder.inference(hidden, max_len=30, start_token_idx=start_token_idx, device=device)
        return tokenizer.decode(predicted_caption[0], skip_special_tokens=True)

# Função para criar GIF a partir dos frames
def create_gif(frames, gif_path):
    images = [Image.fromarray(frame) for frame in frames]
    imageio.mimsave(gif_path, images, duration=0.1, loop=0)  # loop=0 faz o GIF rodar infinitamente

# Streamlit UI
st.title("Video Captioning com Streamlit")
uploaded_video = st.file_uploader("Faça upload de um vídeo", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

    st.video(video_path)  # Mostrar o vídeo original

    # Etapa 1: Extrair frames
    st.write("📸 Extraindo frames...")
    frames = extract_frames(video_path)

    # Etapa 2: Processar frames no CLIP
    st.write("🖼️ Extraindo características visuais...")
    npy_features = process_frames(frames).squeeze(1)

    # Etapa 3: Passar no modelo de legendas
    st.write("📝 Gerando legenda...")
    video_tensor = torch.tensor(npy_features, dtype=torch.float32).to(device).unsqueeze(0)
    caption = generate_caption(video_tensor)

    # Etapa 4: Criar GIF
    gif_path = ".temp/temp_video.gif"
    create_gif(frames, gif_path)

    # Exibir GIF e legenda
    display_gif(gif_path)
    st.markdown(f"**Legenda Predita:** {caption}")


