import streamlit as st
import torch
import tempfile
import os
import numpy as np
from PIL import Image
from inference import video_to_caption

st.set_page_config(page_title="Gerador de Descrições de Cenas", layout="centered")
st.title("🎬 Gerador de Descrições de Cenas com CLIP + Transformer")
st.write("Faça upload de um vídeo curto e receba uma descrição gerada automaticamente.")

# Parâmetros do modelo (de acordo com o treino)
model_params = {
    'input_dim': 512,
    'hidden_dim': 512,
    'num_layers': 8,
    'nhead': 16
}
max_seq_len = 20
max_frames = 40
model_path = "results/best_model.pt"  # ajuste conforme necessário
device = 'cuda' if torch.cuda.is_available() else 'cpu'

uploaded_file = st.file_uploader("📁 Faça upload de um vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("🔍 Gerar descrição"):
        with st.spinner("Processando o vídeo..."):
            # Salvar vídeo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmp_video_path = tmpfile.name

            try:
                frames_selecionados, description = video_to_caption(
                    video_path=tmp_video_path,
                    model_path=model_path,
                    device=device,
                    max_seq_len=max_seq_len,
                    max_frames=max_frames,
                    model_params=model_params
                )

                st.success("✅ Descrição gerada:")
                st.markdown(f"> {description}")

                if frames_selecionados:
                    pil_images = [
                        Image.fromarray((frame * 255).astype(np.uint8)) if frame.max() <= 1
                        else Image.fromarray(frame.astype(np.uint8))
                        for frame in frames_selecionados
                    ]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as gif_file:
                        pil_images[0].save(
                            gif_file.name,
                            save_all=True,
                            append_images=pil_images[1:],
                            duration=150,
                            loop=0
                        )
                        gif_path = gif_file.name

                    st.image(gif_path, caption="📸 Frames selecionados", use_column_width=True)

            except Exception as e:
                st.error(f"❌ Erro: {e}")
            finally:
                os.remove(tmp_video_path)
