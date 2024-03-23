import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class AttentionNetwork(nn.Module):
    def __init__(self, embedding_size, sequence, hidden_units=512):
        super().__init__()
        self.hidden_dim = hidden_units
        self.emb_dim = embedding_size
        self.encoder = nn.LSTM(embedding_size, hidden_units, num_layers=1, bidirectional=True)
        self.sequence_embedding = return_seq_emb(sequence)
        self.fc1 = nn.Linear(hidden_units, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        # New linear layer to project sequence embedding to 32 dimensions
        self.fc_seq = nn.Linear(1024, 32)
        # Updated fc4 layer to accept concatenated input of 64 dimensions (32 from fc3 + 32 from fc_seq)
        self.fc4 = nn.Linear(64, 1)

    @staticmethod
    def attention_layer(encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        attention_wights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        alphas = F.softmax(attention_wights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), alphas.unsqueeze(2)).squeeze(2)
        return alphas, new_hidden

    def forward(self, x):
        output, (encoder_hidden, cell_state) = self.encoder(x)
        bidirectional_sum_initial = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        bidirectional_sum_initial = bidirectional_sum_initial.permute(1, 0, 2)
        bidirectional_sum = (encoder_hidden[-2, :, :] + encoder_hidden[-1, :, :]).unsqueeze(0)
        alphas, attn_out = self.attention_layer(bidirectional_sum_initial, bidirectional_sum)
        x = F.relu(self.fc1(attn_out))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Project sequence embedding to 32 dimensions
        seq_emb_proj = F.relu(self.fc_seq(self.sequence_embedding))
        # Adjust seq_emb_proj to have the same batch size as x
        seq_emb_proj = seq_emb_proj.unsqueeze(0)  # Add batch dimension
        seq_emb_proj = seq_emb_proj.expand(x.size(0), -1)  # Repeat to match x's batch size
        # Concatenate fc3 output with projected sequence embedding
        x = torch.cat((x, seq_emb_proj), dim=1)
        x = self.fc4(x)
        return x


def return_seq_emb(sequence):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    encoded_input = tokenizer(sequence, return_tensors='pt')  # Use the function parameter `seq`

    with torch.no_grad():
        outputs = model(**encoded_input)
    # Get the mean embedding across the sequence length dimension, resulting in a single 1024-dim vector
    sequence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Ensure it's squeezed to [1024]
    return sequence_embedding
