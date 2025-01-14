from torch import nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size, num_features, attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size, attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size, num_features, attention_dim)
        attention_scores = self.A(combined_states)  # (batch_size, num_features, 1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size, num_features)

        alpha = F.softmax(attention_scores, dim=1)  # (batch_size, num_features)
        attention_weights = features * alpha.unsqueeze(2)  # (batch_size, num_features, encoder_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size, encoder_dim)
        return alpha, attention_weights


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeds = self.embedding(captions)
        h, c = self.init_hidden_state(features)
        seq_length = len(captions[0]) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.fcn.out_features).to(features.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def init_hidden_state(self, encoder_out):
        if (len(encoder_out.shape)==3):
          mean_encoder_out = encoder_out.mean(dim=1)
        else:
          mean_encoder_out = encoder_out
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c


    def generate_caption(self,features,max_len=20,vocab=None):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        #starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to("cuda")
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha,context = self.attention(features, h)

            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)

            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            #save the generated word
            captions.append(predicted_word_idx.item())

            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        return [vocab.itos[idx] for idx in captions],alphas



class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, temporal_dim, num_layers, drop_prob=0.3):
        super().__init__()
        #self.encoder = EncoderCNN()
        #self.temporal_encoder = TemporalEncoder(encoder_dim * 64, temporal_dim, num_layers)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            drop_prob=drop_prob
        )

    def forward(self, video_frames, captions):
        #features = self.encoder(video_frames)  # (batch_size, num_frames, 49, 2048)
        #temporal_features = self.temporal_encoder(features)  # (batch_size, num_frames, temporal_dim)

        outputs = self.decoder(video_frames, captions)
        return outputs
