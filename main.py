import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import os
import time
import argparse
from data_preprocess.build_vocab import build_vocab
from exp.Exp import Exp
from utils.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pure", help="[pure, mix]")
    parser.add_argument("--model", type=str, default="ResnetTransformer", help="")
    parser.add_argument("--n_epochs", type=int, default=30, help="")
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--dim", type=int, default=256, help="") # d_model
    parser.add_argument("--n_heads", type=int, default=4, help="")
    parser.add_argument("--n_layers", type=int, default=3, help="")
    parser.add_argument("--dropout", type=float, default=0.2, help="")
    parser.add_argument("--img_size", type=int, default=224, help="") # image size
    parser.add_argument("--max_len", type=int, default=500, help="")
    parser.add_argument("--seed", type=int, default=2023, help="")
    parser.add_argument("--device_id", type=int, default=0, help="")
    # modes
    parser.add_argument('--sample', type=bool, default=False, help='') # True: sampling
    parser.add_argument('--dev', type=bool, default=False, help='')
    parser.add_argument('--test', type=bool, default=False, help='') # True: labeling for test set
    parser.add_argument("--pretrain", type=bool, default=False, help="") # MLM pretrain
    parser.add_argument("--finetune", type=bool, default=False, help="")
    # obj detection
    parser.add_argument('--multi_sample', type=bool, default=False, help='')  # True: sampling
    parser.add_argument("--yolo", type=bool, default=False, help="")
    # path config
    parser.add_argument('--vocab_path', type=str, default="./vocab/vocab_plus.txt", help='')
    args = parser.parse_args()
    args.device = 'cuda:' + str(args.device_id) if torch.cuda.is_available() else 'cpu'
    if args.task == "mix":
        # include Chinese
        args.vocab_path = "./vocab/vocab_plus_cn.txt"
    print("Args:")
    print(args)
    set_seed(args.seed) # set random seed

    setting = "{}_{}_d{}_nh{}_nl{}_ep{}".format(
        args.model,
        args.task, # [pure, mix]
        args.dim,
        args.n_heads,
        args.n_layers,
        args.n_epochs
    )
    args.setting = setting
    # experiment
    exp = Exp(args=args)
    if args.sample:
        exp.sample()
    elif args.multi_sample:
        exp.multi_sample()
    elif args.yolo:
        exp.yolo_dev()
    elif args.test:
        # labeling for test set
        exp.test()
        print("over!")
    elif args.dev:
        # temp
        checkpoint_path = './checkpoints/train/' + args.setting + '/model.pth'
        exp.model.load_state_dict(torch.load(checkpoint_path))
        exp.dev()
        print("over")
    elif args.pretrain:
        exp.pretrain()
        print("over!")
    elif args.finetune:
        exp.finetune()
        print("over!")
    else:
        exp.train()
        print("over!")
    