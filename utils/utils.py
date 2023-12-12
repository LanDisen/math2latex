import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
import numpy as np
import math
import os
import time
import random
from typing import Union
from PIL import Image
import nltk
from data_preprocess.build_vocab import build_vocab
# Metrics


def set_seed(seed):
    '''设置随机种子'''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def adjust_lr(optimizer, epoch, args):
    '''学习率调整'''
    lr_adjust = {epoch: args.lr * (0.8 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    
def triangular_mask(size):
    r'''
    上三角mask用于transformer decoders
    Example:
        size=4, return: Tensor.
        [[0, -inf, -inf, -inf], 
         [0, 0   , -inf, -inf],
         [0, 0   , 0   , -inf],
         [0, 0   , 0   , 0   ]]
    '''
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask

def pad_mask(batch):
    b = len(batch)
    max_len = max(batch)
    mask = torch.full((b, max_len), 1)
    for i, l in enumerate(batch):
        mask[i, :l] = 0
    mask = mask.bool()
    return mask

def find_first(x: Tensor, element: Union[int, float], dim: int = 1) -> Tensor:
    """Find the first occurence of element in x along a given dimension.

    Args:
        x: The input tensor to be searched.
        element: The number to look for.
        dim: The dimension to reduce.

    Returns:
        Indices of the first occurence of the element in x. If not found, return the
        length of x along dim.

    Usage:
        >>> first_element(Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])

    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/9

        I fixed an edge case where the element we are looking for is at index 0. The
        original algorithm will return the length of x instead of 0.
    """
    mask = x == element
    found, indices = ((mask.cumsum(dim) == 1) & mask).max(dim)
    indices[(~found) & (indices == 0)] = x.shape[dim]
    return indices

def remove_ignored(sentences, ignored_words):
    '''
    return: list
    '''
    B = sentences.shape[0] # batch size
    ret_sentences = []
    for i in range(B):
        ret_sentences.append([token for token in sentences[i].tolist() if token not in ignored_words])
    return ret_sentences # list shape: [B, L]
