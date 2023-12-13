import argparse

import numpy as np
import torch
from models.ResnetTransformer import Model


def convert(export_onnx_path, device, model, batch_size, input_shape):
    # set the model to inference mode
    # model.eval()
    x = torch.randn(batch_size,*input_shape, requires_grad=True)

    x = x.to(device)
    x = x.half()

    torch.onnx.export(
        model,
        x,
        export_onnx_path,
        opset_version=10,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}}
    )


SET_MODELS = ["ResnetTransformer"]  # 需要转换的模型们

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pure", help="[pure, mix]")
    parser.add_argument("--model", type=str, default="ResnetTransformer", help="")
    parser.add_argument("--n_epochs", type=int, default=30, help="")
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--dim", type=int, default=256, help="")  # d_model
    parser.add_argument("--n_heads", type=int, default=4, help="")
    parser.add_argument("--n_layers", type=int, default=3, help="")
    parser.add_argument("--dropout", type=float, default=0.2, help="")
    parser.add_argument("--img_size", type=int, default=224, help="")  # image size
    parser.add_argument("--max_len", type=int, default=500, help="")
    parser.add_argument("--seed", type=int, default=2023, help="")
    parser.add_argument('--sample', type=bool, default=False, help='')  # True: sampling
    parser.add_argument('--test', type=bool, default=False, help='')  # True: labeling for test set
    parser.add_argument('--to_onnx', type=bool, default=False, help='')
    # path config
    parser.add_argument('--vocab_path', type=str, default="./vocab/vocab_plus.txt", help='')
    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.task == "mix":
        # 词表包括中文
        args.vocab_path = "./vocab/vocab_plus_cn.txt"

    if "ResnetTransformer" in SET_MODELS:
        if args.device == 'cuda:0':
            print("cuda")
            torch.cuda.empty_cache()
        torch_model = Model(args)
        # 开启半精度, 可以加快运行速度、减少GPU占用，并且只有不明显的accuracy损失。
        torch_model.half()
        torch_model.to(args.device).eval()
        torch_model.load_state_dict(torch.load("checkpoints/ResnetTransformer_mix_d128_nh4_nl2_ep30/model.pth"))  #加载.pth文件
        export_onnx_file = "../checkpoints/ResnetTransformer_mix_d128_nh4_nl2_ep30/model.onnx"
        batch_size = 1
        input_shape = (3, 224, 224)   # 模型的输入，根据训练时数据集的输入
        convert(export_onnx_file, args.device, torch_model, batch_size, input_shape)

