import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
import math
from data_preprocess.build_vocab import build_vocab
from layers.Embed import PositionalEncoding1D, PositionalEncoding2D
from utils.utils import triangular_mask, pad_mask, find_first
from transformers import ViTForImageClassification, ViTConfig
import math

class Encoder(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # local_path = "F:/googlevit-base-patch16-224"  # 预训练模型文件夹路径
        self.dim = dim
        vit_config = ViTConfig(hidden_size=dim, 
                               num_channels=1, 
                               num_hidden_layers=3,
                               num_attention_heads=4,
                               hidden_dropout_prob=dropout)
        # vit_path = "./checkpoints/googlevit-base-patch16-224"
        self.vit_model = ViTForImageClassification(config=vit_config)
        self.vit_model.classifier = nn.Identity()  # 去除分类器
        # print(self.vit_model)
        # self.linear_layer = nn.Linear(768, dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, C=1, H=224, W=224]
        # x是输入图片的张量，B: batch_size, C:通道(灰度图是1, RGB是3), H:图片高度, W:图片宽度
        # x = torch.cat([x]*3, dim=1)
        # x = x.repeat(1, 3, 1, 1)
        hidden_states = self.vit_model(x, output_hidden_states=True)["hidden_states"]
        x = hidden_states[-1]
        x = x.permute(1, 0, 2).contiguous()
        # x = self.linear_layer(x)
        x = x[1:] # remove CLS token
        return x  # [H*W//1024, B, D]


class Decoder(nn.Module):
    r'''
    Transformer Decoder
    Args: 
        dim:
        num_classes:
        max_len
    '''
    def __init__(self, dim, n_heads, n_layers, max_len, num_classes, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.max_len = max_len + 2  # 包括起始符和终止符
        self.num_classes = num_classes
        # 词嵌入
        self.embedding = nn.Embedding(num_classes, self.dim)  # word2vec
        # 一维文本位置编码
        self.pe = PositionalEncoding1D(self.dim, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            dim, n_heads, dim*4, dropout)
        self.decode = nn.TransformerDecoder(decoder_layer, n_layers)
        self.mask = triangular_mask(self.max_len)

    def forward(self, x_encoded: torch.Tensor, y: torch.Tensor, length=None):
        # x_encoded: [H*W//1024, B, D]
        # y: [B, L]
        y = y.permute(1, 0)  # [L, B]
        # y需要是LongTensor类型才能进行词嵌入
        y = self.embedding(y) * math.sqrt(self.dim)  # [L, B, D]
        y = self.pe(y)  # 添加位置编码
        L = y.shape[0]
        # mask
        y_mask = self.mask[:L, :L].type_as(x_encoded)  # [L, L]
        if length is not None:
            padding_mask = pad_mask(length)[:, :y_mask.shape[1]].to(y.device)
        else:
            padding_mask = None
        output = self.decode(y, x_encoded,
                             tgt_mask=y_mask,
                             tgt_key_padding_mask=padding_mask)
        return output  # [L, B, D]


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.max_len = args.max_len
        self.n_layers = args.n_layers
        self.vocab = build_vocab(args.vocab_path)
        self.num_classes = len(self.vocab)
        self.encoder = Encoder(self.dim, args.dropout)
        self.decoder = Decoder(self.dim, self.n_heads, n_layers=self.n_layers,
                               max_len=self.max_len,
                               num_classes=self.num_classes,
                               dropout=args.dropout)
        self.head = nn.Linear(self.dim, self.num_classes)

    def forward(self, x, y, length=None):
        # x: [B, C=1, H, W] images
        # y: [B, L] word embeddings
        x = self.encoder(x)  # [H*W//1024, B, D]
        output = self.decoder(x, y, length)  # [L, B, D]
        output = self.head(output)  # [L, B, num_classes]
        output = output.permute(1, 2, 0)  # [B, num_classes, L]
        # output = output.permute(1, 0, 2)
        return output

    def predict(self, images):
        # images: [B, C, H, W]
        B = images.shape[0]  # batch size
        output_indices = torch.full((B, self.max_len), self.vocab(
            "<pad>")).type_as(images).long()  # all zero
        output_indices[:, 0] = self.vocab("<start>")  # 第一个设置为起始符
        has_ended = torch.full((B,), False).bool()

        x_encoded = self.encoder(images)  # [H*W//1024, B, D]
        for L in range(1, self.max_len):
            y = output_indices[:, :L]
            logits = self.head(self.decoder(x_encoded, y)
                               )  # [L, B, num_classes]
            outputs = torch.argmax(logits, dim=-1)
            output_indices[:, L] = outputs[-1:]

            # Early stopping
            has_ended |= (output_indices[:, L] == self.vocab(
                "<end>")).type_as(has_ended)
            if torch.all(has_ended):
                break

        # <end>后面都是<pad>
        eos_positions = find_first(output_indices, self.vocab("<end>"))
        for i in range(B):
            j = int(eos_positions[i].item()) + 1
            output_indices[i, j:] = self.vocab("<pad>")
        return output_indices  # [B, max_len]

    def tokens(self, idx, vocab):
        '''将模型输出的索引列表根据词表转换为词tokens'''
        # idx: [max_len]
        tokens = []
        for index in idx:
            token = vocab.idx2word[index]
            if token == '<end>':
                break
            if token == '<pad>' or token == '<start>':
                continue
            tokens.append(token)
        return tokens
