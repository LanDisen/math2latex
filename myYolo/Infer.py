import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time
from Model import myYOLO


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean, std):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x /= 255.
    x -= mean
    x /= std
    return x


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels


def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names):
    for i, box in enumerate(bboxes):
        cls_indx = cls_inds[i]
        xmin, ymin, xmax, ymax = box
        if scores[i] > thresh:
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
            cv2.rectangle(img, (int(xmin), int(abs(ymin) - 20)), (int(xmax), int(ymin)),
                          class_colors[int(cls_indx)], -1)
            mess = '%s' % (class_names[int(cls_indx)])
            cv2.putText(img, mess, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def test(net, device, testset, transform, thresh, class_colors=None, class_names=None):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")

        # scale each detection back up to the image
        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names)
        cv2.imshow('detection', img_processed)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)


def infer(net, device, root_path, transform, thresh, class_colors=None, class_names=None):
    while True:
        random_int = random.randint(0, 50656)
        pic_path = os.path.join(root_path, "PngImages", str(random_int) + ".png")
        img = cv2.imread(pic_path)
        h, w, _ = img.shape

        # to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        # 转为numpy形式
        # bboxes = bboxes.cpu().numpy()
        # scores = bboxes.cpu().numpy()
        # cls_inds = cls_inds.cpu().numpy()

        # scale each detection back up to the image
        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names)
        cv2.imshow('detection', img_processed)
        key = cv2.waitKey(0)
        if key == ord('q'):  # 如果按下 'q' 键，则退出循环
            break
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)


if __name__ == '__main__':
    # cuda
    if torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda:0")
    else:
        print("cpu")
        device = torch.device("cpu")

    # 设定输入的大小, 按照训练是的大小来
    input_size = 416
    input_size = [input_size, input_size]

    # 加载数据集
    print("加载数据集......")
    data_dir = "../../resource/yolo"
    num_classes = 1
    # dataset = VOCDetection(root=data_dir, img_size=input_size[0], transform=None)

    class_colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
                    range(num_classes)]

    # 加载模型
    print("加载模型......")
    net = myYOLO(device, input_size=input_size, num_classes=num_classes, trainable=False)

    # 定义模型的位置
    trained_model = "./out/model_80.pth"
    net.load_state_dict(torch.load(trained_model, map_location=device))
    net.to(device).eval()
    print("加载模型完成")

    # 设置置信度
    visual_threshold = 0.4
    VOC_CLASSES = (  # always index 0
        'math',)

    # 开始推理
    print("---------------------------------------------")
    print("开始推理")
    # test(net=net,
    #      device=device,
    #      testset=dataset,
    #      transform=BaseTransform(input_size),
    #      thresh=visual_threshold,
    #      class_colors=class_colors,
    #      class_names=VOC_CLASSES,
    #      )

    infer(net=net,
         device=device,
         root_path=data_dir,
         transform=BaseTransform(input_size),
         thresh=visual_threshold,
         class_colors=class_colors,
         class_names=VOC_CLASSES,
    )
