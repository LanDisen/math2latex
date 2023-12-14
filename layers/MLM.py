import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math
import numpy as np
import nltk

class MLM(nn.Module):
    '''Mask Language Model'''
    def __init__(self, vocab, mask_prob=0.15):
        super().__init__()
        vocab.add_word('<mask>')
        self.vocab = vocab
        self.mask_prob = mask_prob
    
    def mask_ids(self, labels):
        # labels: [B, L]
        # device = labels.device
        # labels = labels.to("cpu")
        mask_labels = labels.clone() # 避免原地修改
        random_matrix = torch.rand(mask_labels.shape).to(labels.device)
        # 确保起始符和结束符不被mask
        mask = (random_matrix < self.mask_prob) & (labels != self.vocab('<pad>')) & (labels != self.vocab('<start>')) & (labels != self.vocab('<end>'))
        mask_labels[mask] = self.vocab("<mask>")
        # labels = labels.to(device)
        return mask_labels
    
    def forward(self, labels):
        return self.mask_ids(labels)