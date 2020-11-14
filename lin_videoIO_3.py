# 该代码实现功能：实时识别
from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import datasets

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from utils.utils import rescale_boxes
from utils.datasets import pad_to_square, resize

from tqdm import tqdm

from ALL_sign_data.model import Lenet5, my_resnt18, FashionCNN
from ALL_sign_data.resnet import ResNet18

import json

sign_classes = 115
# classes_weights_path = "ALL_sign_data/model_acc_90__epoch_4.pt"
classes_weights_path = "ALL_sign_data/checkpoints/model_acc_97__class_115_epoch_10.pt"
# os.makedirs("output", exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/changshu_18_during", help="path to dataset 待检测文件所在的文件夹")
    parser.add_argument("--model_def", type=str, default="config/ALL_DATA.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_13.pth", help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_33.pth", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="data/ALL_DATA.names", help="path to class label file")
    parser.add_argument("--class_path", type=str, default="ALL_sign_data/ALL_data_in_2_train/names.txt", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    # torch.Tensor分配到的设备的对象，包括：设备类型+设备序号
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 开始时间
    start_t = time.time()

    # 创建“检测”模型：模型定义文件 + 图片大小
    # 深度学习框架 Darknet 用C和CUDA编写的开源神经网络框架
    # 把模型放到 GPU 上，功能：检测
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path))
    model.eval() 

    # 加载“分类”模型，设置模型参数，放到 GPU 上，转换为 eval 模式
    # model_class = FashionCNN(sign_classes)
    # ResNet18：封装好的模型（属于经典网络），功能：分类
    model_class = ResNet18(sign_classes)    
    model_class.load_state_dict(torch.load(classes_weights_path))
    # model_class.load_state_dict(torch.load(classes_weights_path, map_location=device))
	
    model_class.to(device)
    model_class.eval()
    print("\n创建两个模型，用时："+str(time.time()-start_t)+"\n")

    # 从类别标签文件中，提取类别，得到一个 class 数组
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # IO
    # 输入，输出：mp4
    input_video_path = "lin_input.mp4"
    output_video_path = "lin_output.avi"
    output_result_path = "lin_output.json"

    input = cv2.VideoCapture(0)
    fps = int(input.get(cv2.CAP_PROP_FPS))
    width = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))
    train_results = {"imgs" : {}}

    # 循环读取图片
    nums = 0
    while True:
        ret, frame = input.read()
        nums += 1    
        if not ret:
            break

        print("---------------读取第"+str(nums)+"帧")
        
        frame_start_t = time.time()
        # cv2 读取图片转换为 PIL 格式 转换为 Tensor 
        img = torchvision.transforms.ToTensor()(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).convert(mode="RGB"))

        input_imgs, _ = pad_to_square(img, 0)
        # Resize
        input_imgs = resize(input_imgs, opt.img_size).unsqueeze(0)

        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        tensor_t = time.time()
        print("转换为 Tensor 用时："+str(time.time()-frame_start_t))

        # 开始检测 
        with torch.no_grad():
            detections = model(input_imgs.to(device))
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
        detect_t = time.time()
        print("进行物体检测用时："+str(detect_t-tensor_t))
        
        # 处理每一帧的检测结果
        objects = []  
        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, img.shape[1:])
            
            unique_labels = detections[:, -1].cpu().unique()
            # 元组分解为fig和ax两个变量
            fig, ax = plt.subplots()
            # img_copy =Image.open(img_path)
            img_copy = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            # ax.imshow(img_copy)
            # 检测物体的个数
            j = 0
            
            # 对每一个检测到的物体进行操作
            for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections): 
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_w = x2 - x1
                box_h = y2 - y1
                

                min_sign_size = 10
                if box_w >= min_sign_size and box_h >= min_sign_size:

                    # 对PIL.Image进行多个变换，裁剪等
                    crop_sign_org = img_copy.crop((x1, y1, x2, y2)).convert(mode="RGB")
                    test_transform = torchvision.transforms.Compose([ 
                        torchvision.transforms.Resize((28, 28), interpolation=2),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                        ])
                    crop_sign_input = test_transform(crop_sign_org).unsqueeze(0)

                    # 对检测到的物体进行分类
                    with torch.no_grad():
                        pred_class = model_class(crop_sign_input.to(device))
                    sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]
                    cls_pred = sign_type
                    cls_pred_type = classes[int(cls_pred)]

                    # 绘制检测框
                    if True and classes[int(cls_pred)] != "zo":     
                        cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 0, 255), 2)
                        cv2.putText(frame, cls_pred_type, (x1,y1+50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, 4)
                        objects.append({'category': classes[int(cls_pred)], 'score': 848.0, 'bbox': {'xmin': x1, 'ymin': y1, 'ymax': y2, 'xmax': x2}})
        classify_t = time.time()
        print("进行物体分类和图片标注，用时："+str(classify_t-detect_t))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break
        # 写每一帧
        output.write(frame)
        # 更新：该帧对应的检测结果
        train_results["imgs"][nums] = {"objects": objects}
        
        frame_end_t = time.time()
        print("该帧总用时："+str(frame_end_t-frame_start_t))


    input.release()
    cv2.destroyAllWindows()
    end_t = time.time()
    print("\n\n该视频总用时："+str(time.time()-start_t)+"  \n处理帧数："+str(nums))
    # 输出：所有帧的检测结果写到 json 文件中去
    with open(output_result_path, "w") as file_object:
        json.dump(train_results, file_object)