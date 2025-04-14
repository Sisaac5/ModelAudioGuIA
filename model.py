import torch
import torch.nn as nn
import math
from transformers import BertTokenizer, BertModel

class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, timestamps):
        """Encode timestamps into sinusoidal embeddings.
        
        Args:
            timestamps: (batch_size, seq_len) tensor of timestamps
            
        Returns:
            embeddings: (batch_size, seq_len, d_model) tensor of time encodings
        """
        pe = torch.zeros(*timestamps.shape, self.d_model).to(timestamps.device)
        pe[..., 0::2] = torch.sin(timestamps.unsqueeze(-1) * self.div_term)
        pe[..., 1::2] = torch.cos(timestamps.unsqueeze(-1) * self.div_term)
        return pe

class SceneDescriptionModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=4, nhead=8, max_seq_len=30):
        super(SceneDescriptionModel, self).__init__()
        
        # Feature projection
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal encoding components
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.time_encoder = TimeEncoding(hidden_dim)
        
        # Transformer encoder for temporal features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer decoder for text generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # BERT embeddings with projection
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_proj = nn.Linear(768, hidden_dim)
        
        # Output layer
        self.vocab_size = self.bert.config.vocab_size
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)
        
        self.max_seq_len = max_seq_len
        
    def forward(self, visual_features, timestamps, text_input=None):
        """
        Args:
            visual_features: (batch_size, seq_len, input_dim)
            timestamps: (batch_size, seq_len) tensor of frame timestamps
            text_input: (batch_size, text_seq_len) - token ids for teacher forcing
        Returns:
            logits: (batch_size, text_seq_len-1, vocab_size)
        """
        # Process visual features with temporal encoding
        visual_features = self.feature_proj(visual_features)
        
        # Add positional and time encodings
        visual_features = self.pos_encoder(visual_features)
        time_emb = self.time_encoder(timestamps)
        visual_features = visual_features + time_emb
        
        # Encode temporal features
        memory = self.temporal_encoder(visual_features)
        
        if text_input is not None:
            # Prepare decoder input (shift right)
            decoder_input = text_input[:, :-1]
            
            # Get embeddings
            text_embeddings = self.bert.embeddings(decoder_input)
            text_embeddings = self.bert_proj(text_embeddings)
            text_embeddings = self.pos_encoder(text_embeddings)
            
            # Create masks
            tgt_seq_len = decoder_input.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(visual_features.device)
            tgt_padding_mask = (decoder_input == 0)
            
            # Decode
            output = self.text_decoder(
                tgt=text_embeddings,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            logits = self.output_layer(output)
            return logits
        else:
            return memory

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    # Keep existing implementation
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)
        return x + pe.to(x.device)