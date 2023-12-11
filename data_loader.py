import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
import numpy as np
import math
import os
import time
import random
from PIL import Image
import nltk
from data_preprocess.build_vocab import build_vocab

def FMM_func(user_dict, sentence):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子
    """
    # 词典中最长词长度
    max_len = max([len(item) for item in user_dict])
    start = 0
    token_list = []
    while start != len(sentence):
        index = start+max_len
        if index>len(sentence):
            index = len(sentence)
        for i in range(max_len):
            if (sentence[start:index] in user_dict) or (len(sentence[start:index])==1):
                token_list.append(sentence[start:index])
                # print(sentence[start:index], end='/')
                start = index
                break
            index += -1
    return token_list

class ImageLabelDataset(Dataset):
    '''
    加载图片数据集
    Args:
        path: 数据集存放的路径
        flag: 数据加载类型(训练集、验证集、测试集)
    '''
    def __init__(self, path, transform, vocab, latex, flag="train"):
        super().__init__()
        self.path = path
        self.flag = flag # 训练, 验证或测试
        self.images = sorted(os.listdir(path + flag + "/images"))
        self.labels = sorted(os.listdir(path + flag + "/labels"))
        # 词表
        self.vocab = vocab
        self.latex = latex
        # 预处理
        # train_mean = 0.941592
        # train_std = 0.166602
        # self.transform = transforms.Compose([
        #     # transforms.RandomCrop(224, pad_if_needed=True), # padding
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(train_mean, train_std),
        # ])
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        '''
        返回图片以及对应的文字公式混合LaTeX文本标签
        '''
        images = Image.open(self.path + self.flag + "/images/" + self.images[index]).convert("L")
        # images = images.resize((Image.ANTIALIAS))
        images = self.transform(images)
        # 图片归一化处理
        with open(self.path + self.flag + "/labels/" + self.labels[index]) as f:
            # 分段函数有多行
            # labels = f.readline()
            l = [line.strip() for line in f.readlines()] # 去除所有的"\n"
        labels = ""
        for line in l:
            labels += line
        # 文本labels转tensor
        # nltk.download('punkt')
        # tokens = nltk.tokenize.word_tokenize(labels) # 分词
        tokens = FMM_func(self.latex, labels)
        label_ids = []
        label_ids.append(self.vocab('<start>'))
        label_ids.extend([self.vocab(token) for token in tokens])
        label_ids.append(self.vocab('<end>'))
        labels = torch.Tensor(label_ids) # 转换tensor
        return images, labels
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, labels_ids = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(label_ids) for label_ids in labels_ids]
    targets = torch.zeros(len(labels_ids), max(lengths)).long()
    for i, label_ids in enumerate(labels_ids):
        end = lengths[i]
        targets[i, :end] = label_ids[:end]        
    return images, targets, lengths

def get_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, 
                      shuffle=True, drop_last=True,
                      collate_fn=collate_fn)