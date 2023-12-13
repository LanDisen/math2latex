import torch
import numpy as np
import onnxruntime
import json
import time

# 加载数据
# 打开合并后的json文件
with open(r'../merged.json', 'r') as file:
    data = json.load(file)

# 初始化空的数组来存储mmwave_data和kinect_data
mmwave_data_array = np.empty((0, 3))

# 从json中提取mmwave_data和kinect_data，添加在数组中
for obj in data:
    for item in obj:
        mmwave_data = item['mmwave_data']
        mmwave_data_array = np.vstack((mmwave_data_array, mmwave_data))

# print(mmwave_data_array.shape)
print(f"总数据量:{len(mmwave_data_array)}")

mmwave_data_array = mmwave_data_array.reshape(len(mmwave_data_array), 1, 3)
mmwave_data_tensor = torch.as_tensor(mmwave_data_array)
mmwave_data_tensor = mmwave_data_tensor.permute(1, 0, 2)
mmwave_data_tensor = mmwave_data_tensor.squeeze(0)
data_loader = mmwave_data_tensor.numpy().astype(np.float32)


class OnnxInfer:
    def __init__(self, onnx_model_path):

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

    def infer(self, data):
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        # 执行推理
        result = self.ort_session.run([output_name], {input_name: data})

        # 输出推理结果
        print(result)


if __name__ == "__main__":
    # 加载导出的ONNX模型
    onnx_model_path = 'MLP.onnx'  # 替换为你的ONNX模型路径
    onnx_infer = OnnxInfer(onnx_model_path)
    # 模拟ros发布消息
    for i in range(len(data_loader)):
        onnx_infer.infer([data_loader[i]])
        time.sleep(1)