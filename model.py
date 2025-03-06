import torch
import torch.nn as nn

class VideoCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(VideoCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, captions):
        # Encode video features
        encoder_output, hidden = self.encoder(vid_feats)
        
        # Decode captions
        outputs = self.decoder(captions, hidden)
        return outputs

class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden

import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, rnn_dropout_p=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout= rnn_dropout_p)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, captions, hidden):
        """
        Forward pass for training.
        Args:
            captions: Ground truth captions of shape (batch_size, caption_length).
            hidden: Initial hidden state from the encoder of shape (num_layers, batch_size, hidden_size).
        Returns:
            outputs: Predicted word scores of shape (batch_size, caption_length, vocab_size).
        """
        # Embed captions
        embeddings = self.embed(captions)  # (batch_size, caption_length, embed_size)
        
        # Pass through RNN
        rnn_out, hidden = self.rnn(embeddings, hidden)  # rnn_out: (batch_size, caption_length, hidden_size)
        
        # Predict next word
        outputs = self.linear(rnn_out)  # (batch_size, caption_length, vocab_size)
        return outputs

    def inference(self, hidden, max_len=20, start_token_idx=None, device=None):
        """
        Inference mode: Generate captions token-by-token.
        Args:
            hidden: Initial hidden state from the encoder of shape (num_layers, batch_size, hidden_size).
            max_len: Maximum length of the generated caption.
            start_token_idx: Index of the <SOS> token.
            device: Device to use (e.g., 'cuda' or 'cpu').
        Returns:
            captions: Generated captions of shape (batch_size, max_len).
        """
        batch_size = hidden.size(1)
        
        # Initialize the first input token as <SOS>
        inputs = torch.full((batch_size, 1), start_token_idx, dtype=torch.long).to(device)  # (batch_size, 1)
        
        # Store the generated captions
        captions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)
        
        for t in range(max_len):
            # Embed the input token
            embeddings = self.embed(inputs)  # (batch_size, 1, embed_size)
            
            # Pass through RNN
            rnn_out, hidden = self.rnn(embeddings, hidden)  # rnn_out: (batch_size, 1, hidden_size)
            
            # Predict the next word
            outputs = self.linear(rnn_out.squeeze(1))  # (batch_size, vocab_size)
            
            # Get the most likely next token
            _, next_token = torch.max(outputs, dim=1)  # (batch_size)
            
            if all(number == 102 for number in next_token):
                break
            # Store the predicted token
            captions[:, t] = next_token
            
            # Update the input for the next step
            inputs = next_token.unsqueeze(1)  # (batch_size, 1)
        
        return captions
