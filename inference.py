import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import BertTokenizer
from model import SceneDescriptionModel
import clip

def extract_frames(video_path, num_frames=40):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames, preprocess):
    processed = []
    for frame in frames:
        img = preprocess(frame)
        processed.append(img)
    return torch.stack(processed)  # (num_frames, 3, H, W)

def select_frames(features, frames_num):
    features = features / features.norm(dim=-1, keepdim=True)
    k_values = range(2, min(10, len(features)))
    silhouette_scores = []
    best_kmeans = None
    best_score = -1
    best_k = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features.cpu().numpy())
        score = silhouette_score(features.cpu().numpy(), cluster_labels)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_kmeans = kmeans
            best_k = k

    cluster_labels = best_kmeans.labels_
    clustered_images = {i: [] for i in range(best_k)}
    for i, label in enumerate(cluster_labels):
        clustered_images[label].append(i)

    selected_images = []
    clusters_order = list(range(best_k))
    current_cluster_index = 0

    while len(selected_images) < min(frames_num, len(cluster_labels)):
        try:
            cluster_index = clusters_order[current_cluster_index % best_k]
            selected_images.append(clustered_images[cluster_index].pop(0))
            current_cluster_index += 1
        except IndexError:
            clusters_order = [c for c in clusters_order if clustered_images[c]]
            best_k = len(clusters_order)
            if not clusters_order:
                break
            current_cluster_index = 0

    selected_images.sort()
    return selected_images

def video_to_caption(video_path, model_path, device='cuda', max_seq_len=20, max_frames=40,
                     model_params=None):
    
    # Carregar modelo CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # Extrair frames
    frames = extract_frames(video_path, num_frames=max_frames)
    if len(frames) == 0:
        raise ValueError("Nenhum frame extraído do vídeo.")
    frames_tensor = preprocess_frames(frames, clip_preprocess).to(device)  # (N, 3, 224, 224)
    
    # Extrair features com CLIP
    with torch.no_grad():
        features = clip_model.encode_image(frames_tensor).float()  # (N, 512)
    
    # Selecionar frames representativos
    selected_idxs = select_frames(features, frames_num=max_seq_len)
    selected_features = features[selected_idxs]  # (max_seq_len, 512)
    frames_selecionados = frames[selected_idxs]
    
    # Normalizar timestamps    
    normalized_timestamps = [ts / max_frames for ts in selected_idxs]
    timestamps = torch.tensor(normalized_timestamps, dtype=torch.float, device=device)
            
    # Carregar modelo de descrição
    model = SceneDescriptionModel(
        **model_params,  # input_dim, hidden_dim, num_layers, nhead
        max_seq_len=max_seq_len
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Gerar descrição
    with torch.no_grad():
        selected_features = selected_features.unsqueeze(0)  # (1, seq_len, 512)
        timestamps = timestamps.unsqueeze(0)  # (1, seq_len)
        memory = model(selected_features, timestamps)
        generated = torch.tensor([[tokenizer.cls_token_id]], device=device)
        for _ in range(max_seq_len):
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
            
        description = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        
    return frames_selecionados, description

if __name__ == "__main__":
    import sys

    model_params = {
        'input_dim': 512,
        'hidden_dim': 512,
        'num_layers': 8,
        'nhead': 16
    }
    video_path = sys.argv[1]
    model_path = sys.argv[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    desc = video_to_caption(video_path, model_path, device=device, max_seq_len=20, max_frames=40, model_params=model_params)
    print("Descrição gerada:", desc)