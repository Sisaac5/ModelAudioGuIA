import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Attention(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size + encoder_hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, encoder_hidden_size)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).expand(-1, seq_len, -1)
        energy = torch.tanh(self.W(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 
                           hidden_size, 
                           num_layers, 
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.dropout(outputs)  # Apply dropout to all LSTM outputs
        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_hidden_size, 
                 num_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout_embed = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size + encoder_hidden_size, 
                           hidden_size, 
                           num_layers, 
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size, encoder_hidden_size)
        self.dropout_attn = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout_fc = nn.Dropout(dropout)
    
    def forward(self, x, encoder_outputs, hidden, cell):
        embedded = self.embedding(x)
        embedded = self.dropout_embed(embedded)  # Dropout on embeddings
        
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = self.dropout_attn(context).unsqueeze(1)  # Dropout on attention context
        
        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output = self.dropout_fc(output)  # Dropout before final layer
        output = self.fc(output.squeeze(1))
        
        return output, hidden, cell

class VideoCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, frames, input_ids, teacher_forcing_ratio=0.9):
        batch_size = input_ids.size(0)
        target_len = input_ids.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(frames)
        
        input = input_ids[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, encoder_outputs, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = input_ids[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs