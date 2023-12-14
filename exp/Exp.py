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
from models import ResnetTransformer, Yolo
from layers.MLM import MLM # pretrain
# yolo
from layers.yolo.Infer import yoloInfer, BaseTransform, vis  
import cv2
import random
from cnocr import CnOcr
from data_loader import FMM_func

warnings.filterwarnings('ignore')

class Exp:
    def __init__(self, args):
        self.args = args 
        # add models
        self.model_dict = {
            "ResnetTransformer": ResnetTransformer,
        }
        self.vocab = build_vocab(args.vocab_path)
        if self.args.pretrain or self.args.finetune:
            self.vocab.add_word("<mask>")
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
        # mask language model
        self.mlm = MLM(self.vocab, mask_prob=0.15).to(self.device)
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
        predictions = [self.model.tokens(pred, self.vocab).replace(" ", "") for pred in preds]
        trues = [self.model.tokens(label.tolist(), self.vocab).replace(" ", "") for label in trues]
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
        train_start_time = time.time()
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

        print(f"train end, total time: {time.time() - train_start_time}")
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
            # images = image.unsqueeze(0).to(self.device) # [B=1, C, H, W]
            # pred = self.model.predict(images)
            # pred = remove_ignored(pred, ignored)
            # tokens = self.model.tokens(pred[0], self.vocab)
            # sentence = ""
            # for token in tokens:
            #     # token间不加空格
            #     sentence += token
            sentence = self.model.sample(image)
            print(sentence)
    
    # 多个要识别latex的内容
    def multi_sample(self):
        # yolo的部分
        # 设定输入的大小, 按照训练是的大小来
        input_size = 416
        input_size = [input_size, input_size]

        # 加载数据集
        # root_path = "/root/autodl-tmp/resource"
        root_path = "./dataset/data_mix/test/images/"

        # 加载模型
        print("loading models......")
        net = Yolo.myYOLO(self.device, input_size=input_size, num_classes=1, trainable=False)

        # 定义模型的位置
        # trained_model = './checkpoints/yolo/model_resnet50_s416_lc130.pth'  好像没有原来的好
        trained_model = './checkpoints/yolo/yolo_model.pth'
        net.load_state_dict(torch.load(trained_model, map_location=self.device))
        net.to(self.device).eval()

        # 设置识别框置信度
        visual_threshold = 0.5
        # 随机一张图片
        random.seed(2023)
        random_int = random.randint(0, 50656)
        # pic_path = os.path.join(root_path, "PngImages", str(random_int) + ".png")
        pic_name = "70431.png"
        pic_path = os.path.join(root_path, pic_name)
        print("img_path is: ", pic_path)
        img = cv2.imread(pic_path)
        h, w, _ = img.shape

        # to tensor
        transform = BaseTransform(input_size)
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(self.device)

        # forward
        bboxes, scores, cls_inds = net(x)
        # 设置宽度缩放系数
        longer = 0.1
        # 获取scale
        scale = np.array([[w * (1 - longer) , h, w * (1 + longer), h]])
        # 将box放缩到原来的大小
        bboxes *= scale

        # img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names)
        # math2latex部分, 将识别到的box图片送进识别模型
        # 先转成灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > visual_threshold:
                ignored = [self.vocab('<start>'), 
                        self.vocab('<unk>'), 
                        self.vocab('<end>'), 
                        self.vocab('<pad>')] # 忽略的词
                checkpoint_path = './checkpoints/train/' + self.args.setting + '/model.pth'
                self.model.load_state_dict(torch.load(checkpoint_path))
                image = Image.fromarray(img[int(ymin):int(ymax), int(xmin):int(xmax)])
                image = self.transform(image)
                self.model.eval()
                with torch.no_grad():
                    # image: [C, H, W]
                    images = image.unsqueeze(0).to(self.device) # [B=1, C, H, W]
                    # images = images.half()
                    pred = self.model.predict(images)
                    pred = remove_ignored(pred, ignored)
                    tokens = self.model.tokens(pred[0], self.vocab)
                    sentence = ""
                    for token in tokens:
                        # token间不加空格
                        sentence += token
                    print(sentence)
                    vis(img, box)
                    # cv2.putText(img, sentence, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.15, (0, 0, 0), 1)
                        
        cv2.imwrite(f'./dataset/out/{pic_name}', img)

    def pretrain(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr) # 优化器
        loss_func = nn.CrossEntropyLoss(ignore_index=self.vocab('<pad>')) # 损失函数
        epoch_losses = []
        self.model.train()
        pretrain_start_time = time.time()
        print("pretrain start")
        for epoch in range(1, self.args.n_epochs + 1):
            train_loss = []
            start_time = time.time()
            print(f"epoch: {epoch} start")
            for i, (images, labels, lengths) in enumerate(self.train_loader):
                optimizer.zero_grad()
                images = images.to(self.device) # [B, C, H, W]
                labels = labels.to(self.device) # [B, L]
                mask_labels = self.mlm(labels)
                outputs = self.model(images, mask_labels[:, :-1], lengths) # [B, NC, L]
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
            print(f"\tepoch: {epoch} over, time: {epoch_time:.6f}, pretrain loss: {avg_loss:.6f}")
            # adjust_lr(optimizer, epoch, self.args) # 根据epoch调整学习率

            # test_loss, test_metrics = dev(self.model, self.device, test_loader)
            # print(f"epoch: {epoch}, train loss: {avg_loss}, dev loss: {dev_loss}, test loss: {test_loss}")

        print(f"pretrain end, total time: {time.time() - pretrain_start_time}")
        # save loss
        loss_dir = "./img/loss/" + self.args.setting
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(range(1, self.args.n_epochs + 1), epoch_losses)
        plt.savefig(loss_dir + "/pretrain_loss.png")
        # save model's parameters
        print("saving model...")
        # TODO early stopping
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #     torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        checkpoint_dir = "./checkpoints/" + self.args.setting
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = checkpoint_dir + "/pretrain_model.pth"
        torch.save(self.model.state_dict(), checkpoint_path)

    def finetune(self):
        checkpoint_path = './checkpoints/' + self.args.setting + '/pretrain_model.pth'
        self.model.load_state_dict(torch.load(checkpoint_path))
        print("finetune start")
        self.train() 
        print("finetune end")

    def yolo_dev(self):
        print("yolo_dev start")
        path = "./dataset/data_mix/"
        flag = "dev"
        images = sorted(os.listdir(path + flag + "/images"))
        labels = sorted(os.listdir(path + flag + "/labels"))
        # load models
        model_args = self.args
        model_args.model = "ResnetTransformer"
        model_args.task = "pure"
        model_args.vocab_path = "./vocab/vocab_plus.txt"
        model_setting = "{}_{}_d{}_nh{}_nl{}_ep{}".format(
            model_args.model,
            model_args.task, # [pure, mix]
            model_args.dim,
            model_args.n_heads,
            model_args.n_layers,
            model_args.n_epochs
        )
        model_args.setting = model_setting
        # ResnetTransformer
        model = self.model_dict[model_args.model].Model(model_args).to(self.device)
        model_path = "./checkpoints/train/ResnetTransformer_pure_d256_nh4_nl3_ep30/model.pth"
        model.load_state_dict(torch.load((model_path)))
        ocr = CnOcr()
        yolo_model = Yolo.myYOLO(self.device, input_size=[416, 416], num_classes=1, trainable=False).to(self.device)
        yolo_path = "./checkpoints/yolo/yolo_model.pth"
        yolo_model.load_state_dict(torch.load(yolo_path))
        model.eval()
        yolo_model.eval()
        preds = []
        trues = []
        # load images and labels
        for k in range(len(labels)):
            label_path = path + flag + "/labels/" + labels[k]
            with open(label_path) as f:
                l = [line.strip() for line in f.readlines()] # 去除所有的"\n"
            label = ""
            for line in l:
                label += line
            trues.append(label.replace(" ", ""))
            tokens = FMM_func(self.latex, label)
            label_ids = []
            label_ids.append(self.vocab('<start>'))
            label_ids.extend([self.vocab(token) for token in tokens])
            label_ids.append(self.vocab('<end>'))
            label = torch.Tensor(label_ids) # 转换tensor
            # process image
            image_path = path + flag + "/images/" + images[k]
            image = np.array(Image.open(image_path).convert("L"))
            # yolo box
            box_dict = yolo_model.split_text_formula(image_path)
            text_list = box_dict["word"]
            formula_list = box_dict["formula"]
            i, j = 0, 0
            sentence = ""
            while i < len(text_list) or j < len(formula_list):
                if i < len(text_list):
                    box1 = text_list[i] # [xmin, ymin, xmax, ymax]
                else:
                    box1 = np.array([10000, 0, 0, 0])
                if j < len(formula_list):
                    box2 = formula_list[j]
                else:
                    box2 = np.array([10000, 0, 0, 0])
                if box1[0] < box2[0]:
                    # text
                    box = box1
                    i += 1
                else:
                    # formula
                    box = box2
                    j += 1
                # split image
                cur_image = image[:, int(box[0]): int(box[2])+1]
                if i > j:
                    tokens = ocr.ocr_for_single_line(cur_image)["text"].replace(" ", "")
                else:
                    cur_image = Image.fromarray(image).convert('L')
                    cur_image = self.transform(cur_image).to(self.device)
                    tokens = model.sample(cur_image)
                sentence += tokens
            preds.append(sentence)
        
        score1 = bleu_score(trues, preds)
        score2 = edit_distence(trues, preds)
        score3 = exact_match_score(trues, preds)
        overall = (score1 + score2 + score3) / 3
        print(f"bleu_score: {score1:.6f}")
        print(f"edit_distence: {score2:.6f}")
        print(f"exact_match_score: {score3:.6f}")
        print(f"overall_score: {overall:.6f}")
        print("yolo_dev end")
