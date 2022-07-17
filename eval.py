import pandas as pd
from vocab import Vocab
import utils
from sc_model import SC
import torch.nn as nn
import torch
from utils import pad_sents
from torch.utils.data import DataLoader
import argparse
import os

def length(data):
    return torch.LongTensor([len(seq) for seq in data]).to(device)



def collate_fn(batch):
    x1,x2, y = zip(*batch)
    y = pad_sents(y, "0")
    for i in range(len(y)):
        y[i] = list(map(int, y[i]))
    return x1, x2, torch.Tensor(y).to(device)


def read_data(filename):
    data = pd.read_csv(filename)
    return list(data.iloc[:, 0]), list(data.iloc[:, 1]), list(data.iloc[:, 2])


def loss_function(model, x1, x2, y):
    check, lengths, correct = model(x1)

    loss = loss_check(check, y) / sum(lengths)

    label = y.float().reshape(-1).squeeze()

    source_padded = model.vocab.to_input_tensor(x2, device=model.device).permute(1, 0).reshape(-1).squeeze()

    correct = correct.reshape(source_padded.shape[0], -1)

    l_correct = loss_correct(correct, source_padded) @ label / sum(label)

    #         loss = loss +  l_correct
    loss = loss + l_correct
    return loss


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Load data")
    model_save_path = "model.bin"
    model = SC.load(model_save_path).to(device)

    data = pd.read_csv("data_train.csv")
    eval_check = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 0])))
    eval_correct = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 1])))
    eval_label = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 2])))

    #eval_label = pad_sents(eval_label, "0")
    #for i in range(len(eval_label)):
        #eval_label[i] = list(map(int, eval_label[i]))
    #eval_label = torch.Tensor(eval_label).cuda()


    # train
    data_eval = list(zip(eval_check, eval_correct, eval_label))
    eval_loader = DataLoader(data_eval, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)

    model.eval()
    with torch.no_grad():
        correct = 0
        error = 0
        for eval_check, eval_correct, eval_label in eval_loader:
            c, e = model.evaluate_accuracy(eval_check, eval_correct, eval_label)
            correct +=c
            error +=e
        print(correct)
        print(error)
        score = (correct/error)*100
    print("Acc ", score)

