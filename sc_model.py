import sys
import torch
import torch.nn as nn
import torch.nn.utils
from utils import handle_punctuation, norm_text, check_and_reduce, check_number_in_word
from subword_model import Subword
from vocab import Vocab


class SC(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers=6,
                 d_model=512,
                 nhead=16,
                 hidden_dim=128,
                 dim_feedforward=1024,
                 dropout=0.3,
                 activation=nn.ReLU(),
                 max_length=512,
                 sub_num_layers=3,
                 sub_d_model=256,
                 sub_nhead=8,
                 sub_dim_feedforward=512,
                 sub_dropout=0.3,
                 sub_path="sub_vocab.txt"):
        super(SC, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.sub_num_layers = sub_num_layers
        self.sub_d_model = sub_d_model
        self.sub_nhead = sub_nhead
        self.sub_dim_feedforward = sub_dim_feedforward
        self.sub_dropout = sub_dropout
        self.sub_path = sub_path
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Embedding(self.max_length, self.d_model)
        self.subword = Subword(self.sub_num_layers, self.sub_d_model, self.sub_nhead, self.sub_dim_feedforward, self.sub_dropout, self.max_length, self.sub_path)
        self.model_embedding = nn.Embedding(len(vocab), self.d_model - self.subword.d_model, padding_idx=0)
        self.TransformerLayer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.Transformer = nn.TransformerEncoder(self.TransformerLayer, self.num_layers, self.norm)
        self.linear1 = nn.Linear(self.d_model, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, 1)
        self.linear2 = nn.Linear(self.d_model, self.d_model*2)
        self.output2 = nn.Linear(self.d_model*2, len(vocab))
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout)
    def forward(self, source):
        """ Take a mini-batch of source and target sentences
        @param source (List[List[str]]): list of source sentence tokens
        @returns output(tensor) shape batch, seg, rep
        """
        # source_lengths = [len(s) for s in source]
        source_lengths = list(map(len, source))

        source_padded = self.vocab.to_input_tensor(source, device= self.device) # Tensor: (src_len, b)
        X = self.model_embedding(source_padded) # Tensor: (src_len, b, embedded_dim1)
        sequence_len = X.shape[0]
        b = X.shape[1]
        # pos = [[i for i in range(sequence_len)] for j in range(X.shape[1])] # b,src_len
        pos = torch.tensor(list(range(sequence_len)) * b, device=self.device).reshape(b, -1)
        # pos = torch.tensor(pos, device = self.device)
        mask = source_padded.permute(1,0) == self.vocab['<pad>'] # Tensor: (b, src_len)
        # char_embed = self.subword(source, mask) # Tensor: (src_len, b, embedded_dim2)
        char_embed = self.subword(source, mask, pos)
        X = torch.cat((X,char_embed), dim = 2) # Tensor: (src_len, b, embedded_dim)
        pos_embed = self.pos_embed(pos)
        X = pos_embed.permute(1, 0, 2) + X
        context_rep = self.Transformer(X, src_key_padding_mask = mask) # Tensor: (src_len, b, d_model)
        check = self.linear1(context_rep) # Tensor: (src_len, b, hidden_dim)
        check = self.activation(check)
        check = self.dropout_layer(check)
        check = self.output1(check) # Tensor: (src_len, b, 1)
        check = self.sigmoid(check)
        check = check.permute(1,0,2).squeeze() # Tensor: (b, src_len)
        if len(source) > 1:
            check.data.masked_fill_(mask, 0)
        correct = self.linear2(context_rep)
        correct = self.activation(correct)
        correct = self.dropout_layer(correct)
        correct = self.output2(correct)
        correct = correct.permute(1,0,2) # Tensor: (b, src_len, len(vocab))
        return check, source_lengths, correct

    def evaluate_accuracy(self, data_check, data_correct, label):
        total_error = torch.sum(label).item()
        check,_,correct = self.forward(data_check)
        c = torch.round(check)
        y = torch.argmax(correct, dim=-1)
        y_true = self.vocab.to_input_tensor(data_correct, device=self.device)
        y_true = y_true.permute(1, 0)
        total_correct = ((y_true == y).float() * c).sum().item()
        return total_correct, total_error

    def accuracy(self, data_check, data_correct, label):
        correct, error = self.evaluate_accuracy(data_check, data_correct, label)
        return (correct/error)*100

    def predict(self, data):
        self.eval()
        data = norm_text(data)
        data = handle_punctuation(data)
        raw = data.split()
        data = [data.lower().split()]
        data = [list(map(check_and_reduce, data[0]))]
        c = torch.ones(5)
        i = 0
        while (sum(c).item() != 0 and i < 5):
            i += 1
            check, _, correct = self.forward(data)
            c = torch.round(check)
            y = torch.argmax(correct, dim=-1)
            try:
                for i in range(len(c)):
                    if c[i] == 1:
                        data[0][i] = self.vocab.id2word[y[0][i].item()]
                if i == 5:
                    break
            except:
                data[0][0] = self.vocab.id2word[y[0][0].item()] if c.item() == 1 else data[0][0]
                break
        new = data[0]
        for i in range(len(raw)):
            if raw[i][0].isupper():
                new[i] = new[i].capitalize()
            if check_number_in_word(raw[i]):
                new[i] = raw[i]
        return " ".join(new)

    def checking(self, source):
        """ Take a mini-batch of source and target sentences
        @param source (List[List[str]]): list of source sentence tokens
        @returns output(tensor) shape batch, seg, rep
        """
        # source_lengths = [len(s) for s in source]
        source_lengths = list(map(len, source))

        source_padded = self.vocab.to_input_tensor(source, device= self.device) # Tensor: (src_len, b)
        X = self.model_embedding(source_padded) # Tensor: (src_len, b, embedded_dim1)
        sequence_len = X.shape[0]
        b = X.shape[1]
        # pos = [[i for i in range(sequence_len)] for j in range(X.shape[1])] # b,src_len
        pos = torch.tensor(list(range(sequence_len)) * b, device=self.device).reshape(b, -1)
        # pos = torch.tensor(pos, device = self.device)
        mask = source_padded.permute(1,0) == self.vocab['<pad>'] # Tensor: (b, src_len)
        # char_embed = self.subword(source, mask) # Tensor: (src_len, b, embedded_dim2)
        char_embed = self.subword(source, mask, pos)
        X = torch.cat((X,char_embed), dim = 2) # Tensor: (src_len, b, embedded_dim)
        pos_embed = self.pos_embed(pos)
        X = pos_embed.permute(1, 0, 2) + X
        context_rep = self.Transformer(X, src_key_padding_mask = mask) # Tensor: (src_len, b, d_model)
        check = self.linear1(context_rep) # Tensor: (src_len, b, hidden_dim)
        check = self.activation(check)
        check = self.dropout_layer(check)
        check = self.output1(check) # Tensor: (src_len, b, 1)
        check = self.sigmoid(check)
        check = check.permute(1,0,2).squeeze() # Tensor: (b, src_len)
        return check, source_lengths

    # def predict_index(self, data):
    #     self.eval()
    #     data = norm_text(data)
    #     raw = word_tokenize(data)
    #     data = [word_tokenize(data.lower())]
    #     data = [list(map(check_and_reduce, data[0]))]
    #     check, _ = self.checking(data)
    #     c = torch.round(check).squeeze()
    #     try:
    #         error = list((c == 1).nonzero().squeeze())
    #     except:
    #         error = list((c == 1).nonzero())
    #     return [e.item() for e in error]


    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embedding.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = SC(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(num_layers = self.num_layers,
                         d_model = self.d_model,
                         nhead = self.nhead,
                         hidden_dim = self.hidden_dim,
                         dim_feedforward= self.dim_feedforward,
                         dropout= self.dropout,
                         activation= self.activation,
                         max_length = self.max_length,
                         sub_num_layers=self.sub_num_layers,
                         sub_d_model=self.sub_d_model,
                         sub_nhead=self.sub_nhead,
                         sub_dim_feedforward=self.sub_dim_feedforward,
                         sub_dropout=self.sub_dropout,
                         sub_path=self.sub_path),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
