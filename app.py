import streamlit as st
import torch
import tempfile
import os
import numpy as np
from PIL import Image
from inference import video_to_caption
from model import SceneDescriptionModel
from transformers import BertTokenizer
import clip
import re
import base64  # <-- adicionado para exibir GIF com HTML

def capitalize_first_letter(text):
    return re.sub(r'(^|(?<=[\.\!\?]\s))([a-zá-úà-úâ-ûãõç])',
                  lambda m: m.group(1) + m.group(2).upper(),
                  text)

@st.cache_resource
def load_model_and_tokenizer(model_params, max_seq_len, model_path, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SceneDescriptionModel(**model_params, max_seq_len=max_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    return model, tokenizer, clip_model, clip_preprocess

st.set_page_config(page_title="Gerador de Descrições de Cenas", layout="centered")
st.title("🎬 Gerador de Descrições de Cenas com CLIP + Transformer")
st.write("Faça upload de um vídeo curto e receba uma descrição gerada automaticamente.")

model_params = {
    'input_dim': 512,
    'hidden_dim': 512,
    'num_layers': 4,
    'nhead': 8
}

max_seq_len = 20
max_frames = 40
model_path = "./best_model.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, tokenizer, clip_model, clip_preprocess = load_model_and_tokenizer(model_params, max_seq_len, model_path, device)
uploaded_file = st.file_uploader("📁 Faça upload de um vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    ext = os.path.splitext(uploaded_file.name)[1]

    if st.button("🔍 Gerar descrição"):

        with st.spinner("Processando o vídeo..."):
            # Salvar vídeo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmp_video_path = tmpfile.name

            try:
                frames_selecionados, description = video_to_caption(
                    video_path=tmp_video_path,
                    model=model,
                    device=device,
                    max_seq_len=max_seq_len,
                    max_frames=max_frames,
                    tokenizer=tokenizer,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess
                )
                description = capitalize_first_letter(description)
                
                st.success("✅ Descrição gerada:")
                st.markdown(f"> {description}")

                if frames_selecionados:
                    pil_images = [
                        Image.fromarray((frame * 255).astype(np.uint8)) if frame.max() <= 1
                        else Image.fromarray(frame.astype(np.uint8))
                        for frame in frames_selecionados
                    ]

                    # Salvar como GIF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as gif_file:
                        pil_images[0].save(
                            gif_file.name,
                            save_all=True,
                            append_images=pil_images[1:],
                            duration=150,
                            loop=0
                        )
                        gif_path = gif_file.name

                    with open(gif_path, "rb") as f:
                        gif_bytes = f.read()
                        b64_gif = base64.b64encode(gif_bytes).decode("utf-8")
                        st.markdown(f'<img src="data:image/gif;base64,{b64_gif}" width="100%" />', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Erro: {e}")
            finally:
                os.remove(tmp_video_path)
