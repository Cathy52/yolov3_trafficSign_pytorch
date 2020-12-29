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
    # 准备：设置设备，所有分类，构建张量
    t1 = time.time() 

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
        print("------")
        print(time.time())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(time.time())

        os.makedirs("output", exist_ok=True)
        # 读取所有的类别
        classes = load_classes(opt.class_path)  # Extracts class labels from file
        print("------")
        print(time.time())
        # 构建了一个Float型的张量
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        print("--模型检测----")
        print(time.time())

    # 定义检测用的模型
    if(True):
        # Set up model # 加载权重文件 # 不启用 BatchNormalization 和 Dropout
        # 深度学习框架 Darknet 用C和CUDA编写的开源神经网络框架
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        print("------")
        print(time.time())
        model.load_state_dict(torch.load(opt.weights_path))
        print("------")
        print(time.time())
        model.eval()  # Set in evaluation mode 
        print("---模型识别---")
        print(time.time())

    # 定义分类用的模型
    if(True):
        # model_class = FashionCNN(sign_classes)
        model_class = ResNet18(sign_classes)
        print("------")
        print(time.time())
        model_class.load_state_dict(torch.load(classes_weights_path))
        print("------")
        print(time.time())
        model_class.to(device)
        print("------")
        print(time.time())
        model_class.eval()

    crop_dirs = ["image_for_detect/Tinghua100K"]
    
    # 每一张图片进行识别
    for dir_ in crop_dirs:
        print("---开始---")
        print(time.time())
        train_results = {"imgs" : {}}

        opt.image_folder = dir_

        # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        names = os.listdir(opt.image_folder)
        nums = 0
        for name in tqdm(names[:]):
            print("----图片开始--")
            print(time.time())

            img_id = name.split(".")[0]
            img_path = os.path.join(opt.image_folder, name)
            if not os.path.isfile(img_path):
                continue

            if not (img_path.endswith(".jpg")  or img_path.endswith(".png") or img_path.endswith(".ppm")  ):  #  
                continue

            # Extract image as PyTorch tensor
            img = torchvision.transforms.ToTensor()(Image.open(img_path).convert(mode="RGB"))
            print("--读取完毕，转换成了tensor变量----")
            print(time.time())
          
            input_imgs, _ = pad_to_square(img, 0)
            # Resize
            input_imgs = resize(input_imgs, opt.img_size).unsqueeze(0)

            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))
            print("---开始检测---")
            print(time.time())

            # Get detections
            with torch.no_grad():
                detections = model(input_imgs.to(device))
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]

            print("---检测完毕---")
            print(time.time())

            if  detections is not None:  #  one image
                print("---检测结果不为空---")
                print(time.time())

                objects = []  #  save the results of a image detection
                detections = rescale_boxes(detections, opt.img_size, img.shape[1:])
                unique_labels = detections[:, -1].cpu().unique()
                fig, ax = plt.subplots()
                img_copy =Image.open(img_path) 
                j = 0
                
                print("------")
                print(time.time())

                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections): #  one object in a image                
                    print("---开始分析一张图片的每一个检测的区域---")
                    print(time.time())
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    min_sign_size = 10

                    if box_w >= min_sign_size and box_h >= min_sign_size:                        
                        print("--如果区域足够大，切割图片，做一种处理----")
                        print(time.time())
                        crop_sign_org = img_copy.crop((x1, y1, x2, y2)).convert(mode="RGB")

                        # #### to class  ###############
                        test_transform = torchvision.transforms.Compose([ 
                            torchvision.transforms.Resize((28, 28), interpolation=2),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
                            ])

                        crop_sign_input = test_transform(crop_sign_org).unsqueeze(0)

                        print("---处理好了---")
                        print(time.time())

                        with torch.no_grad():
                            print("---开始识别---")
                            print(time.time())
                            pred_class = model_class(crop_sign_input.to(device))
                            print("---识别结束---")
                            print(time.time())
                        sign_type  = torch.max(pred_class, 1)[1].to("cpu").numpy()[0]
                        # #### to class  ###############
                        cls_pred = sign_type

                        print("cls_pred_type = ", classes[int(cls_pred)])

                        print("---以下：开始保存切割后的图片---")
                        print(time.time())
                        # #############
                        # save crop image
                        # #############                        
                        if classes[int(cls_pred)] != "zo":
                            # save  crop image #############
                            save_dir = "img_crop_2_classification_Tinghua_weights_11/" + classes[int(cls_pred)]
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            name = dir_.split("/")[-2] + "_" + dir_.split("/")[-1] + str(int(random.random() * 100000000))
                            print("save path:", save_dir, str(name) + ".jpg")
                            #  save crop sign
                            crop_sign_org.save(os.path.join(save_dir, str(name) + ".jpg"))
                        
                        print("--保存结束，开始标记----")
                        print(time.time())
                       
                        if True and classes[int(cls_pred)] != "zo":     
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
                        print("--标记结束，开始保存图片----")
                        print(time.time())

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
            print("--保存结束，开始写输出文本文件----")
            print(time.time())

    # 将结果写到json文件中
    if(True):
        file_name = "output/Tinghua100K_result_for_test.json"
        with open(file_name, "w") as file_object:
            json.dump(train_results, file_object)
    
    print("--结束 END----")
    print(time.time())