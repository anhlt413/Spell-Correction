import torch
import torch.nn as nn
from utils import pad_sents, max_length_char
from vocab import Subword_vocab
class Subword(nn.Module):
    def __init__(self,
                 num_layers=3,
                 d_model=256,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0.3,
                 max_length=512,
                 sub_path="sub_vocab.txt" ):
        super(Subword, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.sub_path = sub_path
        self.subword_vocab = Subword_vocab.from_corpus(self.sub_path)
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.dropout = dropout
        self.model_embedding = nn.Embedding(len(self.subword_vocab), self.d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)
        self.TransformerLayer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.Transformer = nn.TransformerEncoder(self.TransformerLayer, self.num_layers, self.norm)
        self.pos_embed = nn.Embedding(self.max_length, self.d_model)

    def forward(self, source, mask, pos):
        # mask used for transformer
        pos_embed = self.pos_embed(pos) # b, length, d_model
        source_padded = pad_sents(source, "<pad>") # (batch, length_sen)
        # length_char = [[len(word) for word in sent] for sent in source_padded]
        length_char = [list(map(len, sent)) for sent in source_padded]
        m = max_length_char(length_char)
        # indices = torch.tensor([[[self.subword_vocab[c] for c in word] + (m - len(word)) * [0] for word in sent] for sent in source_padded]).cuda()
        indices = torch.tensor([[[self.subword_vocab[c] for c in word] + (m - len(word)) * [0] for word in sent] for sent in source_padded]).to(self.device)
        inp = self.model_embedding(indices)
        sum_char = torch.sum(inp, dim=-2)
        # length_char = torch.tensor(length_char).unsqueeze(-1).cuda()
        length_char = torch.tensor(length_char).unsqueeze(-1).to(self.device)
        inp = sum_char/length_char + pos_embed
        # inp = sum_char / length_char
        inp = inp.permute(1,0,2) # (src_len, b, dim)
        context_rep = self.Transformer(inp, src_key_padding_mask=mask) # Tensor: (src_len, b, embedded_dim)
        return context_rep

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embedding.weight.device

