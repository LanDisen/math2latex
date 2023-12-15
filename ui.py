import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
from ttkbootstrap import Style
import matplotlib.pyplot as plt
import io
import re
from models.ResnetTransformer import Model
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
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from utils.utils import remove_ignored
from exp.Exp import Exp
from data_preprocess.build_vocab import build_vocab
from models import ResnetTransformer  

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def connect_image(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size
    # Create a new image with the combined width and the height of the tallest image
    new_width = width1 + width2
    new_height = max(height1, height2)
    new_image = Image.new("RGB", (new_width, new_height), 'white')
    # Paste the two images onto the new image
    if height1 > height2:
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (width1, int((height1 - height2) / 2)))
    else:
        new_image.paste(image1, (0, int((height2 - height1) / 2)))
        new_image.paste(image2, (width1, 0))

    return new_image


def str_to_image(str):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(0.5, 0.5, str, fontsize=30, verticalalignment='center', horizontalalignment='center')
    renderer = fig.canvas.get_renderer()
    bbox = ax.texts[0].get_window_extent(renderer=renderer)
    img_size = (bbox.width / 100 + 0.1, bbox.height / 100 + 0.1)
    fig.set_size_inches(img_size)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img = Image.open(img_buf)
    img = img.crop((2, 2, img.size[0] - 2, img.size[1] - 2))  # 裁去边缘
    return img


def latex_to_image(str_latex):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(0.5, 0.5, '$' + str_latex + '$', fontsize=30, verticalalignment='center', horizontalalignment='center')
    renderer = fig.canvas.get_renderer()
    bbox = ax.texts[0].get_window_extent(renderer=renderer)
    img_size = (bbox.width / 100 + 0.1, bbox.height / 100 + 0.1)
    fig.set_size_inches(img_size)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img = Image.open(img_buf)
    img = img.crop((2, 2, img.size[0] - 2, img.size[1] - 2))  # 裁去边缘
    return img


# 将Latex公式字符串转化为图片
def string_to_latex(str_latex):
    substring = r"\\text\s*{\s*([^}]+)\s*}"  # 用正则表达式来表示字符串中的文本
    extracted = r'\\text\s*\{(.*?)\}'
    new_string = re.sub(substring, "#", str_latex)  # 将字符串中的文本去除
    latex_list = re.split("#", new_string)
    i = 1
    img = Image.new("RGB", (0, 0), 'white')
    print(latex_list)
    if len(latex_list[0]) != 0:
        latex_list[0] = re.sub('boldsymbol', 'mathbf', latex_list[0])
        img = connect_image(img, latex_to_image(latex_list[0]))
    while True:
        new_string = re.search(substring, str_latex)
        if new_string is not None:
            new_string = re.search(extracted, new_string.group())
            str_latex = re.sub(substring, "", str_latex, 1)
            img = connect_image(img, str_to_image(new_string.group(1)))
            if len(latex_list[i]) != 0:
                latex_list[i] = re.sub('boldsymbol', 'mathbf', latex_list[i])
                img = connect_image(img, latex_to_image(latex_list[i]))
            i = i + 1
        else:
            break

    return img


# 放大图片
def resize_image(img, max_width_length, max_height_length):
    # 获取原尺寸
    width, height = img.size
    # 计算放大后的尺寸
    scale_factor = min(max_width_length / width, max_height_length / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # 放大图像
    resized_img = img.resize((new_width, new_height))
    return resized_img


class ImageTextRecognitionApp:
    # 屏幕尺寸
    screen_width = 0
    screen_height = 0

    def __init__(self, root, args):
        self.root = root
        self.root.title("Math2LaTeX")
        self.screen_width = root.winfo_screenwidth()  # 获取屏幕宽度
        self.screen_height = root.winfo_screenheight()  # 获取屏幕高度
        self.root.geometry(str(int(self.screen_width / 2)) + "x" + str(int(self.screen_height / 1.5)) + "+200+50")
        # self.root.config(background="whitesmoke")

        self.style = Style(theme='lumen')

        # load model
        # 图像预处理
        if args.task == "pure":
            train_mean = 0.930882
            train_std = 0.178370
        elif args.task == "mix":
            train_mean = 0.917530
            train_std = 0.188238
        self.transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
        self.vocab = build_vocab(args.vocab_path)
        self.model = Model(args).to(args.device)
        model_path = "E://models/math2latex/ResnetTransformer_mix_d256_nh4_nl3_ep30/model.pth"
        print("loading model......")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # load model end

        self.upload_button = ttk.Button(root, text="上传", command=self.upload_image)
        # self.upload_button.grid(row=0, column=0, pady=10)
        self.upload_button.place(x=self.screen_width / 50, y=self.screen_height / 100)

        self.convert_button = ttk.Button(root, text="转换", command=self.convert_image)
        # self.convert_button.grid(row=0, column=1, pady=10, sticky='w')
        self.convert_button.place(x=self.screen_width / 15, y=self.screen_height / 100)

        self.image_label = ttk.Label(root, text="待识别图片", font=("微软雅黑", 10))
        # self.image_label.grid(row=1, column=0, pady=5, sticky='n')
        self.image_label.place(x=self.screen_width / 200, y=self.screen_height / 15)

        self.text_display_label = ttk.Label(root, text="原字符串", font=("微软雅黑", 10))
        # self.text_display_label.grid(row=2, column=0, pady=5, sticky='n')
        self.text_display_label.place(x=self.screen_width / 200, y=self.screen_height / 2.8)

        self.latex_display_label = ttk.Label(root, text="Latex公式", font=("微软雅黑", 10))
        # self.latex_display_label.grid(row=3, column=0, pady=5, sticky='n')
        self.latex_display_label.place(x=self.screen_width / 200, y=self.screen_height / 2.4)

        self.image_label = ttk.Label(root)
        # self.image_label.grid(row=1, column=1, pady=5, sticky='nw')
        self.image_label.place(x=self.screen_width / 15, y=self.screen_height / 15)

        self.text_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=int(self.screen_width / 30),
                                                      height=int(self.screen_height / 600), font=("微软雅黑", 11))
        # self.text_display.grid(row=2, column=1, pady=10, sticky='nw')
        self.text_display.place(x=self.screen_width / 15, y=self.screen_height / 2.9)

        self.latex_label = ttk.Label(root)
        # self.latex_label.grid(row=3, column=1, pady=5, sticky='nw')
        self.latex_label.place(x=self.screen_width / 15, y=self.screen_height / 2.4)

        self.image_path = None



        # root.grid_rowconfigure(0, weight=1)
        # root.grid_rowconfigure(1, weight=20)
        # root.grid_rowconfigure(2, weight=3)
        # root.grid_rowconfigure(3, weight=5)
        # root.grid_columnconfigure(0, weight=2)
        # root.grid_columnconfigure(1, weight=3)

    # 上传图片
    def upload_image(self):
        file_path = filedialog.askopenfilename(title="选择图片文件", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.display_image()

    # 显示图片
    def display_image(self):
        image = Image.open(self.image_path)
        # screen_width = root.winfo_screenwidth()
        # screen_height = root.winfo_screenheight()
        image.thumbnail((self.screen_width / 2.4, self.screen_height / 3))
        if image.size[0] < self.screen_width / 4 and image.size[1] < self.screen_height / 5:
            image = resize_image(image, self.screen_width / 4, self.screen_height / 5)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    # 识别图片内容
    def convert_image(self):
        if self.image_path:
            print("converting...")
            image = Image.open(self.image_path).convert('L')
            input_tensor = self.transform(image).unsqueeze(0).to(args.device)
            with torch.no_grad():
                output = self.model.predict(input_tensor)

            # 在这里进行处理，得到 LaTeX 字符串
            ignored = [self.vocab('<start>'), 
                    self.vocab('<unk>'), 
                    self.vocab('<end>'), 
                    self.vocab('<pad>')] # 忽略的词
            pred = remove_ignored(output, ignored)
            tokens = self.model.tokens(pred[0], self.vocab)
            sentence = ""
            for token in tokens:
                sentence += token + " "
            # 示例中假设输出是一个张量
            sentence = sentence.replace(" ", "")
            print(sentence)
            str = sentence
            # image = Image.open(self.image_path)
            # str = predict(image)
            self.text_display.delete(1.0, tk.END)  # 清空Text窗口内容
            self.text_display.insert(tk.END, str)

            img = string_to_latex(str)
            img.thumbnail((self.screen_width / 2.4, self.screen_height / 10))
            photo = ImageTk.PhotoImage(img)
            self.latex_label.config(image=photo)
            self.latex_label.image = photo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mix", help="[pure, mix]")
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
    root = tk.Tk()
    app = ImageTextRecognitionApp(root, args)
    root.mainloop()
