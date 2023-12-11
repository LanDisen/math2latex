import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
import warnings
from data_loader import ImageLabelDataset, get_data_loader
from data_preprocess.build_vocab import build_vocab
from utils.utils import remove_ignored, adjust_lr
from utils.metrics import bleu_score, exact_match_score, edit_distence, overall_score
from models import ResnetTransformer

warnings.filterwarnings('ignore')

class Exp:
    def __init__(self, args):
        self.args = args 
        # add models
        self.model_dict = {
            "ResnetTransformer": ResnetTransformer
        }
        self.vocab = build_vocab(args.vocab_path)
        with open(args.vocab_path, 'r', encoding='utf-8') as f:
            self.latex = f.read().split()
        self.device = args.device
        self.model = self.model_dict[args.model].Model(self.args).to(self.device)
        if args.task == "pure":
            train_mean = 0.930882
            train_std = 0.178370
        elif args.task == "mix":
            train_mean = 0.917530
            train_std = 0.188238
        self.transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
        # data loader
        dataset_path = "./dataset/data_" + args.task + "/"
        self.train_loader = get_data_loader(
            ImageLabelDataset(dataset_path, 
                              self.transform, self.vocab, self.latex,
                              flag="train"),
            batch_size=args.batch_size)
        self.dev_loader = get_data_loader(
            ImageLabelDataset(dataset_path,
                              self.transform, self.vocab, self.latex,
                              flag="dev"),
            batch_size=args.batch_size)
        # self.test_loader = get_data_loader(
        #     ImageLabelDataset(dataset_path, self.transform, self.vocab, self.latex, "test"),
        #     batch_size=1)

    # train, dev, test
    def dev(self):
        print("dev start")
        ignored = [self.vocab('<start>'), self.vocab('<unk>'), self.vocab('<end>'), self.vocab('<pad>')]
        total_loss = []
        preds = []
        trues = []
        data_loader = self.dev_loader
        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab('<pad>')) # 损失函数
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for i, (images, labels, lengths) in enumerate(data_loader):
                images = images.float().to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images, labels, lengths)
                # labels = labels.float()
                pred = outputs.detach().cpu()
                loss = loss_func(pred, labels.cpu())
                total_loss.append(loss)
                pred = self.model.predict(images)
                pred = remove_ignored(pred, ignored) # 去除<start>等
                # flatten
                for p in pred:
                    preds.append(p)
                for label in labels:
                    trues.append(label)
                # iteration
                if (i + 1) % 100 == 0:
                    print(f"\titer: {i + 1}, speed: {(time.time() - start_time) / (i + 1):.6f}s/iter")
                # preds.append(pred.detach())
                # trues.append(labels)
                # temp
                # if i + 1 >= 3: 
                #     break
        print(f"dev time: {time.time() - start_time}")
        avg_loss = np.average(total_loss)
        predictions = [self.model.tokens(pred, self.vocab) for pred in preds]
        trues = [self.model.tokens(label.tolist(), self.vocab) for label in trues]
        # metrics
        score1 = bleu_score(trues, predictions)
        score2 = edit_distence(trues, predictions)
        score3 = exact_match_score(trues, predictions)
        overall = (score1 + score2 + score3) / 3
        print(f"avg_loss: {avg_loss:.6f}")
        print(f"bleu_score: {score1:.6f}")
        print(f"edit_distence: {score2:.6f}")
        print(f"exact_match_score: {score3:.6f}")
        print(f"overall_score: {overall:.6f}")
        metrics = [score1, score2, score3, overall]
        self.model.train()
        print("dev end")
        return avg_loss, metrics

    def train(self):
        train_loader = self.train_loader
        n_epochs = self.args.n_epochs
        batch_size = self.args.batch_size
        lr = self.args.lr
        # test_loader = get_data_loader(ImageLabelDataset("./dataset/data/", "test"), batch_size=batch_size)
        print("train data size:", self.train_loader.__len__() * batch_size)
        print("dev data size:", self.dev_loader.__len__() * batch_size)
        print("train start")
        vocab = self.vocab
        # print("test data size:", len(test_loader))
        optimizer = optim.Adam(self.model.parameters(), lr=lr) # 优化器
        loss_func = nn.CrossEntropyLoss(ignore_index=vocab('<pad>')) # 损失函数
        epoch_losses = []
        self.model.train()
        for epoch in range(1, n_epochs + 1):
            train_loss = []
            start_time = time.time()
            print(f"epoch: {epoch} start")
            for i, (images, labels, lengths) in enumerate(train_loader):
                # break
                optimizer.zero_grad()
                images = images.to(self.device) # [B, C, H, W]
                labels = labels.to(self.device) # [B, L]
                outputs = self.model(images, labels[:, :-1], lengths) # [B, NC, L]
                # labels = labels.float()
                loss = loss_func(outputs, labels[:, 1:])
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                if self.args.task == "pure" and (i + 1) % 10 == 0:
                    print(f"\titer: {i + 1}, loss: {loss.item():.6f}, speed: {(time.time() - start_time) / (i + 1):.6f}s/iter")
                elif self.args.task == "mix" and (i + 1) % 100 == 0:
                    print(f"\titer: {i + 1}, loss: {loss.item():.6f}, speed: {(time.time() - start_time) / (i + 1):.6f}s/iter")
            
            epoch_time = time.time() - start_time
            avg_loss = np.average(train_loss)
            epoch_losses.append(avg_loss)
            print(f"\tepoch: {epoch} over, time: {epoch_time:.6f}, train loss: {avg_loss:.6f}")
            # adjust_lr(optimizer, epoch, self.args) # 根据epoch调整学习率

            # test_loss, test_metrics = dev(self.model, self.device, test_loader)
            # print(f"epoch: {epoch}, train loss: {avg_loss}, dev loss: {dev_loss}, test loss: {test_loss}")

        print("train end")
        # save loss
        loss_dir = "./img/loss/" + self.args.setting
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(range(1, n_epochs + 1), epoch_losses)
        plt.savefig(loss_dir + "/loss.png")
        # save model's parameters
        print("saving model...")
        # TODO early stopping
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #     torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        checkpoint_dir = "./checkpoints/" + self.args.setting
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = checkpoint_dir + "/model.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        # 验证集dev
        dev_loss, dev_metrics = self.dev() # 后续改为每10个epoch验证一次

    def test(self):
        # data_loader = get_data_loader(ImageLabelDataset("./dataset/data_pure/", "test"),
        #                             batch_size=1)
        test_image_dir = "./dataset/data_" + self.args.task + "/test/images"
        test_label_dir = "./dataset/data_" + self.args.task + "/test/labels/"
        images = sorted(os.listdir(test_image_dir))
        vocab = self.vocab
        ignored = [vocab('<start>'), vocab('<unk>'), vocab('<end>'), vocab('<pad>')]
        preds = []
        checkpoint_path = './checkpoints/' + self.args.setting + '/model.pth'
        self.model.load_state_dict(torch.load(checkpoint_path)) # load model
        self.model.eval()
        print("test start")
        start_time = time.time()
        with torch.no_grad():
            i = 0
            for fname in images:
                image = Image.open(test_image_dir + "/" + fname).convert("L")
                image = self.transform(image).float().to(self.device)
                image = image.unsqueeze(0) # [B=1, C=1, H, W]
                pred = self.model.predict(image)
                pred = remove_ignored(pred, ignored) # 去除<start>等
                # type(pred)=list [B=1, L(不等长)]
                tokens = self.model.tokens(pred[0], self.vocab)
                # tokens写入文件
                idx = fname.split(".")[0]
                with open(test_label_dir + idx + ".txt", "w", encoding="utf-8") as f:
                    for token in tokens:
                        f.write(token)
                        # if token == "\\\\":
                        #     f.write("\n")
                i += 1
                if i % 1000 == 0:
                    print(f"\titer: {i}, speed: {(time.time() - start_time) / i:.6f}s/iter")
        print(f"test time: {time.time() - start_time}")
        print("test end")

    def sample(self):
        ignored = [self.vocab('<start>'), 
                   self.vocab('<unk>'), 
                   self.vocab('<end>'), 
                   self.vocab('<pad>')] # 忽略的词
        checkpoint_path = './checkpoints/' + self.args.setting + '/model.pth'
        self.model.load_state_dict(torch.load(checkpoint_path))
        # 0: a_n, 3: x^2 + y^2 -
        # 2: a_{k+1}+..., 4: 6y+5=0, 16: DC=2BD
        # image_path = "./dataset/data_pure/train/images/" + "3.png" 
        # image_path = "./dataset/data_pure/test/images/" + "2.png"
        image_path = "./dataset/data_mix/test/images/" + "353.png"
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        self.model.eval()
        with torch.no_grad():
            # image: [C, H, W]
            images = image.unsqueeze(0).to(self.device) # [B=1, C, H, W]
            pred = self.model.predict(images)
            pred = remove_ignored(pred, ignored)
            tokens = self.model.tokens(pred[0], self.vocab)
            sentence = ""
            for token in tokens:
                # token间不加空格
                sentence += token
            print(sentence)