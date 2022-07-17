import pandas as pd
from create_data import create
from vocab import Vocab
import utils
from sc_model import SC
import torch.nn as nn
import torch
from utils import pad_sents, check_and_reduce_text, norm_text
from torch.utils.data import DataLoader
import argparse
import os
from preprocess.aug_dataset_v6 import create_data
import os
from tqdm import tqdm


def length(data):
    return torch.LongTensor([len(seq) for seq in data]).to(device)

def compare(check, correct):
    x = check.strip().lower().split()
    y = correct.strip().lower().split()
    label = []
    for i in range(len(x)):
        if x[i] == y[i]:
            label.append("0")
        else:
            label.append("1")
    return  " ".join(label)


def parse_args():
    par = argparse.ArgumentParser()
    par.add_argument("--first_train", type=int, default=1, help="Create new model to train or load it from a path")
    par.add_argument("--epochs", type=int, default=500, help="The number of epochs")
    par.add_argument("--path_model", type=str, default="model.bin", help="Path to save or load model")
    par.add_argument("--batch_size", type=int, default=64, help="Batch size")
    par.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    par.add_argument("--training_path", type=str, default="data_train.csv", help="Path to file csv training data")
    par.add_argument("--dev_path", type=str, default="data_eval.csv", help="Path to file csv including dev data")
    par.add_argument("--vocab_path", type=str, default="vocab.txt", help="Path to file txt including vocab")
    return par.parse_args()


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
    args = parse_args()
    model_save_path = args.path_model
    if args.first_train == 1:
        v = Vocab.from_corpus(args.vocab_path, 30000, 1)
        print("Create new model")
        model = SC(v).to(device)
    else:
        model = SC.load(model_save_path).to(device)
    # data = pd.read_csv(args.training_path)
    # read_data
    # data_check = list(map(lambda x: x.strip().lower().split(),list(data.iloc[:, 0])))
    # for line in list(data.iloc[:, 0]):
    #     data_check.append(line.strip().lower().split())
    # data_correct = list(map(lambda x: x.strip().lower().split(),list(data.iloc[:, 1])))
    # for line in list(data.iloc[:, 1]):
    #     data_correct.append(line.strip().lower().split())
    # data_label = list(map(lambda x: x.strip().lower().split(),list(data.iloc[:, 2])))
    # for line in list(data.iloc[:, 2]):
    #     data_label.append(line.strip().lower().split())
    print("Load data eval", args.dev_path)
    data = pd.read_csv(args.dev_path)
    eval_check = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 0])))
    eval_correct = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 1])))
    eval_label = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 2])))

    #eval_label = pad_sents(eval_label, "0")
    #for i in range(len(eval_label)):
        #eval_label[i] = list(map(int, eval_label[i]))
    #eval_label = torch.Tensor(eval_label).cuda()
    # param
    print("Batch size", args.batch_size)
    batch_size = args.batch_size
    print("Lr:", args.lr)
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_check = nn.BCELoss(reduction="sum")
    loss_correct = nn.CrossEntropyLoss(reduction="none")
    lr_decay = 0.5
    print("Epochs:", args.epochs)
    epochs = args.epochs
    patience = 0
    k = 0

    # train
    data_eval = list(zip(eval_check, eval_correct, eval_label))
    eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    min_loss = 0
    pa = 0
    for epoch in range(epochs):
        print("Load data train")
        data = create(args.training_path)
        data_check = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 0])))
        # for line in list(data.iloc[:, 0]):
        #     data_check.append(line.strip().lower().split())
        data_correct = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 1])))
        # for line in list(data.iloc[:, 1]):
        #     data_correct.append(line.strip().lower().split())
        data_label = list(map(lambda x: x.strip().lower().split(), list(data.iloc[:, 2])))
        data = list(zip(data_check, data_correct, data_label))
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        print("Done Preb")
        model.train()
        total_loss = 0
        for x1, x2, y in tqdm(loader):
            optimizer.zero_grad()

            loss = loss_function(model, x1, x2, y)
            loss.backward()
            optimizer.step()

        # Early stop and decay lr
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                error = 0
                for eval_check, eval_correct, eval_label in eval_loader:
                    c, e = model.evaluate_accuracy(eval_check, eval_correct, eval_label)
                    correct +=c
                    error +=e
                score = correct/error *100
            if epoch == 0:
                best_score = score
                print("Save model with acc = {:.2f} %".format(best_score))
                model.save(model_save_path)
            if score > best_score:
                patience = 0
                best_score = score
                print("Save model with acc = {:.2f} %".format(best_score))
                model_save_path = "model" + "{:.2f}".format(best_score) + ".bin"
                model.save(model_save_path)

            else:
                patience += 1
            if patience == 5:
                # decay lr, and restore from previously best checkpoint
                lr = optimizer.param_groups[0]['lr'] * lr_decay
                if lr <= 0.000001:
                    break
                print('Load previously best model and decay learning rate to %f' % lr)
                model = SC.load(model_save_path).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # reset patience
                patience = 0

