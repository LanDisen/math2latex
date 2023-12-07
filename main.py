import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import os
import time
import argparse
from data_preprocess.build_vocab import build_vocab
# from model import Model
from exp.Exp import Exp
from utils.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ResnetTransformer", help="")
    parser.add_argument("--n_epochs", type=int, default=10, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--lr", type=float, default=0.01, help="")
    parser.add_argument("--dim", type=int, default=256, help="") # d_model
    parser.add_argument("--n_heads", type=int, default=4, help="")
    parser.add_argument("--n_layers", type=int, default=4, help="")
    parser.add_argument("--dropout", type=float, default=0.2, help="")
    parser.add_argument("--img_size", type=int, default=224, help="")
    parser.add_argument("--max_len", type=int, default=200, help="")
    parser.add_argument("--seed", type=int, default=2023, help="")
    parser.add_argument('--sample', type=bool, default=False, help='')
    # parser.add_argument("--train")
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Args:")
    print(args)
    set_seed(args.seed) # 设置随机种子
    # device = args.device
    # # 建立词表
    # vocab = build_vocab("./data_preprocess/vocab.txt")
    # num_classes = len(vocab.word2idx) # 所有token数量
    # model = Model(args.dim, args.n_heads, args.max_len).to(device)
    exp = Exp(args=args)
    if args.sample:
        exp.sample()
    else:
        exp.train()
        print("over!")

    # exp.train(model, args.n_epochs, args.batch_size, args.lr)
    # test(model, device)
    