import torch
import torch.nn as nn
from layers.yolo.Modules import SPP, SAM, BottleneckCSP, Conv
from layers.yolo.Resnet import resnet18, resnet50
import numpy as np
from layers.yolo import Tools
from layers.yolo.Infer import yoloInfer, BaseTransform, vis  
import cv2
import random
from PIL import Image

class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 hr=False):
        super(myYOLO, self).__init__()
        # 定义类内参数
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        self.grid_cell = self.create_grid(input_size)
        self.input_size = input_size
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()

        # resnet18
        self.backbone = resnet18(pretrained=True)

        # neck
        self.SPP = nn.Sequential(
            Conv(512, 256, k=1),
            SPP(),
            BottleneckCSP(256 * 4, 512, n=1, shortcut=False)
        )
        self.SAM = SAM(512)
        self.conv_set = BottleneckCSP(512, 512, n=3, shortcut=False)

        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # w = input_size
        # h = input_size
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 2).to(self.device)

        return grid_xy

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_boxes(self, pred):
        """
        input box :  [tx, ty, tw, th]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2

        return output

    # 非极大抑制（NMS）
    def nms(self, dets, scores):
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)  # bbox大小
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        """
        bbox_pred: (HxW, 4), bsize = 1
        prob_pred: (HxW, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds

    def forward(self, x, target=None):
        # backbone
        _, _, C_5 = self.backbone(x)

        # head
        C_5 = self.SPP(C_5)
        C_5 = self.SAM(C_5)
        C_5 = self.conv_set(C_5)

        # pred
        prediction = self.pred(C_5)
        prediction = prediction.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
        B, HW, C = prediction.size()

        # Divide prediction to obj_pred, txtytwth_pred and cls_pred
        # [B, H*W, 1]
        conf_pred = prediction[:, :, :1]
        # [B, H*W, num_cls]
        cls_pred = prediction[:, :, 1: 1 + self.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = prediction[:, :, 1 + self.num_classes:]

        # 测试
        if not self.trainable:
            with torch.no_grad():
                all_conf = torch.sigmoid(conf_pred)[0]  # batch_size只有一
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)

                all_conf = all_conf.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                return bboxes, scores, cls_inds
        else:
            conf_loss, cls_loss, txtytwth_loss, total_loss = Tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target)

            return conf_loss, cls_loss, txtytwth_loss, total_loss
        
    def split_text_formula(self, image_path):
         # yolo的部分
        # 设定输入的大小, 按照训练是的大小来
        input_size = 416
        input_size = [input_size, input_size]

        # 加载数据集
        # root_path = "/root/autodl-tmp/resource"
        root_path = "./dataset/data_mix/test/images/"

        # 加载模型
        # print("loading models......")
        # net = Yolo.myYOLO(self.device, input_size=input_size, num_classes=1, trainable=False)
        # # 定义模型的位置
        # # trained_model = './checkpoints/yolo/model_resnet50_s416_lc130.pth'  好像没有原来的好
        # trained_model = './checkpoints/yolo/model_80.pth'
        # net.load_state_dict(torch.load(trained_model, map_location=self.device))
        # net.to(self.device).eval()

        # 设置识别框置信度
        visual_threshold = 0.4
        # # 随机一张图片
        # random.seed(2023)
        # random_int = random.randint(0, 50656)
        # pic_path = os.path.join(root_path, "PngImages", str(random_int) + ".png")
        # pic_name = "15893.png"
        # pic_path = os.path.join(root_path, pic_name)
        # print("img_path is: ", pic_path)

        # image: [C=1, H=224, W=224] Tensor 
        img = cv2.imread(image_path)
        # image = image.permute(1, 2, 0) # [H, W, 1]
        # img = image.cpu().numpy()
        h, w, _ = img.shape
        # to tensor
        transform = BaseTransform(input_size)
        # numpy to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(self.device)

        # forward
        bboxes, scores, cls_inds = self.forward(x)
        # 设置宽度缩放系数
        longer = 0.1
        # 获取scale
        scale = np.array([[w * (1 - longer) , h, w * (1 + longer), h]])
        # 将box放缩到原来的大小
        bboxes *= scale

        # math2latex部分, 将识别到的box图片送进识别模型
        # 返回的字典
        ret_dic = {"word":[], "formula":[] }  
        for i, box in enumerate(bboxes):
            if scores[i] > visual_threshold:
                # ret_dic["formula"].append([int(xmin), int(ymin),  int(xmax), int(ymax)])
                ret_dic["formula"].append(box)
        rightbox = ret_dic["formula"]
        rightbox = sorted(rightbox, key=lambda x: x[0]) 
        last_x_max = 0
        for i, box in enumerate(rightbox):
            xmin, ymin, xmax, ymax = box
            ret_dic["word"].append(np.array([last_x_max ,ymin, xmin ,ymax]))
            last_x_max = xmax
            if i == len(rightbox) - 1 and xmax < w:
               ret_dic["word"].append(np.array([xmax ,ymin, w ,ymax]))    
        # print(ret_dic)      
        # 返回图片原始坐标 
        if len(ret_dic["formula"]) == 0:
            ret_dic["word"].append(np.array([0, 0, w, h]))
        return ret_dic