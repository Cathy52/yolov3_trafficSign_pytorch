from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

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

os.environ['CUDA_VISIBLE_DEVICES']='0'
sign_classes = 115
classes_weights_path = "ALL_sign_data/checkpoints/model_acc_97__class_115_epoch_10.pt"


if __name__ == "__main__":
    if(True):
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_folder", type=str, default="data/changshu_18_during", help="path to dataset")
        parser.add_argument("--model_def", type=str, default="config/ALL_DATA.cfg", help="path to model definition file")
        parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_33.pth", help="path to weights file")
        parser.add_argument("--class_path", type=str, default="ALL_sign_data/ALL_data_in_2_train/names.txt", help="path to class label file")
        parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
        parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
        opt = parser.parse_args()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs("output", exist_ok=True)
        # 读取所有的类别：读取文件
        classes = load_classes(opt.class_path)  # Extracts class labels from file
        # 构建了一个Float型的张量
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    print("--------1.加载检测模型：使用Draknet构建检测模型+加载权重文件")
    t1 = time.time()
    # 定义检测用的模型
    if(True):
        # Set up model # 加载权重文件 # 不启用 BatchNormalization 和 Dropout
        # 深度学习框架 Darknet 用C和CUDA编写的开源神经网络框架
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        print("    构建模型")
        t11 = time.time()
        print(t11-t1)
        model.load_state_dict(torch.load(opt.weights_path))
        print("    加载权重文件")
        t12 = time.time()
        print(t12-t11)
        model.eval()  # Set in evaluation mode 
    print("    检测结束")
    print(time.time()-t12)

    # 1.识别模型构建
    print("--------2.识别模型：ResNet18初始化+load_state_dict+to(gpu)+eval")
    t2 = time.time()
    # 定义分类用的模型
    if(True):
        # model_class = FashionCNN(sign_classes)
        model_class = ResNet18(sign_classes)
        model_class.load_state_dict(torch.load(classes_weights_path))
        model_class.to(device)
        model_class.eval()
    print("    识别模型构建结束")
    print(time.time()-t2)

    crop_dirs = ["image_for_detect/Tinghua100K"]    
    for dir_ in crop_dirs:
        train_results = {"imgs" : {}}
        opt.image_folder = dir_
        names = os.listdir(opt.image_folder)
        nums = 0
        for name in tqdm(names[:]):
            # 3.处理图片
            print(name)
            print("--------3.检测前简单处理图片：读取转换为tensor，resize-0，")
            t3 = time.time()

            img_id = name.split(".")[0]
            img_path = os.path.join(opt.image_folder, name)
            if not os.path.isfile(img_path):
                continue

            if not (img_path.endswith(".jpg")  or img_path.endswith(".png") or img_path.endswith(".ppm")  ):  #  
                continue

            # Extract image as PyTorch tensor
            img = torchvision.transforms.ToTensor()(Image.open(img_path).convert(mode="RGB"))
            print("    读取转换成tensor变量")
            t31 = time.time()
            print(t31-t3)
          
            input_imgs, _ = pad_to_square(img, 0)
            # Resize
            input_imgs = resize(input_imgs, opt.img_size).unsqueeze(0)

            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))
            print("    读取，处理图片完毕")
            t32 = time.time()
            print(t32-t31)

            # 4.检测图片
            print("--------4.检测图片")
            t4 = time.time()
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs.to(device))
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]

            print("    检测完毕")
            print(time.time()-t4)

            if  detections is not None:  #  one image
                # 5.识别
                print("--------5.识别图片")
                t5 = time.time()

                objects = []  #  save the results of a image detection
                detections = rescale_boxes(detections, opt.img_size, img.shape[1:])
                unique_labels = detections[:, -1].cpu().unique()
                fig, ax = plt.subplots()
                img_copy =Image.open(img_path) 
                j = 0
                
                print("    识别前处理：提取图片所有的检测框内")
                t51 = time.time()
                print(t51-t5)

                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections): #  one object in a image
                    t51 = time.time()
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    min_sign_size = 10

                    if box_w >= min_sign_size and box_h >= min_sign_size:                        
                        crop_sign_org = img_copy.crop((x1, y1, x2, y2)).convert(mode="RGB")

                        # #### to class  ###############
                        test_transform = torchvision.transforms.Compose([ 
                            torchvision.transforms.Resize((28, 28), interpolation=2),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                            ])

                        crop_sign_input = test_transform(crop_sign_org).unsqueeze(0)

                        print("    识别前对每一个检测框简单处理")
                        t52 = time.time()
                        print(t52-t51)

                        with torch.no_grad():
                            print("    开始识别")
                            t53 = time.time()
                            print(t53-t52)
                            pred_class = model_class(crop_sign_input.to(device))
                            print("    识别结束")
                            t54 = time.time()
                            print(t54-t53)
                        sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]
                        cls_pred = sign_type
                       
                        if True and classes[int(cls_pred)] != "zo":     
                            print("    识别后开始对每一个检测框进行标记")
                            t55 = time.time()
                            color = "r"
                            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                            ax.add_patch(bbox)
                            # Add label
                            plt.text(
                                x1,
                                y2 + 50,
                                s=classes[int(cls_pred)],
                                color="white",
                                verticalalignment="top",
                                bbox={"color": color, "pad": 0},
                            )
                            
                            pad_sign_path_png = "ALL_sign_data/pad-all/" + classes[int(cls_pred)] + ".png"
                            pad_sign_path_jpg = "ALL_sign_data/pad-all/" + classes[int(cls_pred)] + ".jpg"
                            if  os.path.isfile(pad_sign_path_png):
                                pad_sign = Image.open(pad_sign_path_png)
                            elif os.path.isfile(pad_sign_path_jpg):
                                pad_sign = Image.open(pad_sign_path_jpg)
                            else:
                                pad_sign = Image.new("RGB", (100, 100), (255, 255, 255))

                            img_copy.paste(crop_sign_org.resize((100, 100)), (0, j * 100) )
                            img_copy.paste(pad_sign.resize((100, 100)), (100, j * 100) )
                            j += 1
                            
                            #  save predict results to a json file: my_train_results.json
                            objects.append({'category': classes[int(cls_pred)], 'score': 848.0, 'bbox': {'xmin': x1, 'ymin': y1, 'ymax': y2, 'xmax': x2}})
                # 对每一个检测框进行操作                
                    # 不处理太小的检测框
                        # 不标记某些检测框
                            print("    该检测框标记结束")
                            t56 = time.time()
                            print(t56-t55)

                # Save generated image with detections
                nums += 1
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                ax.imshow(img_copy)
                try:
                    dir__ = dir_.split("/")[-1]
                    plt.savefig(f"output/{dir__ + str(nums).zfill(5)}.png", bbox_inches="tight", pad_inches=0.0,)
                except:
                    continue
            train_results["imgs"][img_id] = {"objects": objects}
            print("    该图片保存结束")
            print(time.time()-t56)
            print("    该图片保存用时")
            print(time.time()-t3)
            
        print("--------6.所有识别和检测结束，开始写输出文件")
        t6 = time.time()
    # 将结果写到json文件中
    if(True):
        file_name = "output/Tinghua100K_result_for_test.json"
        with open(file_name, "w") as file_object:
            json.dump(train_results, file_object)
