# encoding:utf-8
# 该代码实现功能：输入视频，输出标注后的视频（仅含中文的提示）
# 效果同4，但是标注内容不一致
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

from PIL import Image, ImageDraw, ImageFont
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
    # 对应标志的map
    map = {
        'i1': '步行',
        'i10': '向右转弯',
        'i11': '向左和向右转弯',
        'i12': '向左转弯',
        'i13': '直行',
        'i14': '直行和向右转弯',
        'i15': '直行和向左转弯',
        'i16': '允许掉头',
        'i17': '残疾人通道',
        'i18': '直行和向右行驶',
        'i19': '服务区',
        'i2': '非机动车行驶',
        'i3': '环岛行驶',
        'i4': '机动车行驶',
        'i5': '靠道路右侧行驶',
        'i6': '靠道路左侧行驶',
        'i7': '立交直行和右转弯行驶',
        'i8': '立交执行和左转弯行驶',
        'i9': '鸣喇叭',
        'il100': '最低限速100',
        'il110': '最低限速110',
        'il50': '最低限速50',
        'il60': '最低限速60',
        'il70': '最低限速70',
        'il80': '最低限速80',
        'il90': '最低限速90',
        'ip': '人行横道',
        'ipp': '停车位',
        'p1': '禁止超车',
        'p10': '禁止机动车驶入',
        'p11': '禁止鸣喇叭',
        'p12': '禁止二轮摩托车驶入',
        'p13': '禁止载货汽车和拖拉机驶入',
        'p14': '禁止直行',
        'p15': '禁止人力车进入',
        'p16': '禁止人力货运三轮车驶入',
        'p17': '禁止人力客运三轮车驶入',
        'p18': '禁止三轮机动车通行',
        'p19': '禁止向右转弯',
        'p2': '禁止畜力车进入',
        'p20': '禁止向左向右转弯',
        'p21': '禁止直行和向右转弯',
        'p22': '禁止三轮机动车通行',
        'p23': '禁止左转',
        'p24': '禁止小型客车右转',
        'p25': '禁止小型客车进入',
        'p26': '禁止载货汽车驶入',
        'p27': '禁止运输危险物品车辆输入',
        'p28': '禁止直行和向左转弯',
        'p29': '禁止拖拉机进入',
        'p3': '禁止大型客车驶入',
        'p30': '禁止攀爬',
        'p31': '禁止明火',
        'p32': '慢行',
        'p4': '禁止电动三轮车驶入',
        'p5': '禁止掉头',
        'p6': '禁止非机动车驶入',
        'p7': '禁止载货汽车左转',
        'p8': '禁止汽车拖、挂车驶入',
        'p9': '禁止行人进入',
        'pa': '限制轴重6t',
        'pa10': '限制轴重10t',
        'pb': '禁止通行',
        'pc': '停车检查',
        'pd': '海关',
        'pdc': '海关',
        'pe': '会车让行',
        'pg': '减速让行',
        'ph3.5': '限制高度3.5m',
        'pl1': '限制质量1.5t',
        'pl10': '限制速度10',
        'pl100': '限制速度100',
        'pl110': '限制速度110',
        'pl120': '限制速度120',
        'pl130': '限制速度130',
        'pl15': '限制速度15',
        'pl2': '限制高度2.8m',
        'pl20': '限制质量20t',
        'pl25': '限制质量25t',
        'pl3': '限制质量3t',
        'pl30': '限制速度30',
        'pl35': '限制速度35',
        'pl4': '限制高度4.5m',
        'pl40': '限制速度40',
        'pl5': '限制速度5',
        'pl50': '限制速度50',
        'pl55': '限制质量55t',
        'pl60': '限制速度60',
        'pl7': '限制质量7.5t',
        'pl70': '限制速度70',
        'pl8': '限制质量8t',
        'pl80': '限制速度80',
        'pl90': '限制速度90',
        'pm10': '限制质量10t',
        'pn': '禁止停车',
        'pne': '禁止驶入',
        'pnl': '禁止长时停车',
        'pr0': '解除禁止超车',
        'pr100': '解除限制速度100',
        'pr20': '解除限制速度20',
        'pr30': '解除限制速度30',
        'pr40': '解除限制速度40',
        'pr50': '解除限制速度50',
        'pr60': '解除限制速度60',
        'pr70': '解除限制速度70',
        'pr80': '解除限制速度80',
        'ps': '停车让行',
        'pw3': '限制宽度3m',
        'w1': '傍山险路',
        'w10': '反向弯路',
        'w11': '反向弯路2',
        'w12': '过水路面',
        'w13': '十字交叉',
        'w14': '十字交叉路口',
        'w15': 'Y形交叉路口',
        'w16': 'Y形交叉路口',
        'w17': 'Y形交叉路口',
        'w18': '左侧变窄',
        'w19': 'Y形交叉路口',
        'w2': '傍山险路2',
        'w20': 'T形交叉',
        'w21': 'T形交叉',
        'w22': 'T形交叉',
        'w23': '环形交叉路口',
        'w24': '连续弯路',
        'w25': '连续下坡',
        'w26': '路面不平',
        'w27': '注意雨雪天',
        'w28': '路面低洼',
        'w29': '路面高凸',
        'w3': '村庄',
        'w30': '慢行',
        'w31': '上坡路',
        'w32': '工地路段',
        'w33': '十字平面交叉',
        'w34': '事故易发路段',
        'w35': '双向交通',
        'w36': '注意野生动物',
        'w37': '隧道',
        'w38': '隧道开车灯',
        'w39': '驼峰桥',
        'w4': '堤坝路',
        'w40': '无人看守铁道路口',
        'w41': '下坡路',
        'w42': '向右急转弯',
        'w43': '向左急转弯',
        'w44': '易滑标志',
        'w45': '注意信号灯',
        'w46': '有人看守铁道路口',
        'w47': '右侧变窄',
        'w48': '右侧绕行',
        'w49': '窄桥',
        'w5': '堤坝路',
        'w50': '注意保持车距',
        'w51': '注意不利气象条件',
        'w52': '注意残疾人',
        'w53': '注意潮汐车道',
        'w54': '注意雾天',
        'w55': '注意儿童',
        'w56': '注意非机动车',
        'w57': '注意行人',
        'w58': '注意左侧合流',
        'w59': '注意右侧合流',
        'w6': '丁字平面交叉',
        'w60': '注意横风',
        'w61': '注意路面结冰',
        'w62': '注意落石',
        'w63': '注意危险',
        'w64': '注意牲畜',
        'w65': '左侧绕行',
        'w66': '左右绕行',
        'w67': '注意前方车辆排队',
        'w68': '注意测速',
        'w69': '注意摔落',
        'w7': '渡口',
        'w8': '两侧变窄',
        'w9': '注意落石',
    }
    
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/changshu_18_during", help="path to dataset")
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
    model_class.load_state_dict(torch.load(classes_weights_path, map_location=device))
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
    output_video_path_2 = "lin_output.avi"
    output_result_path = "lin_output.json"

    input = cv2.VideoCapture(input_video_path)
    fps = int(input.get(cv2.CAP_PROP_FPS))
    width = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))
    train_results = {"imgs" : {}}

    # 循环读取图片
    nums = 0
    # NEW 变量：typeSet period lastTypeSet
    period = 0
    typeDict = dict()
    lastTypeDict = dict()
    # NEW 中文字体
    # 定义宋体路径
    fontpath = 'simsun.ttc'
    # 创建字体对象，并且指定字体大小
    font = ImageFont.truetype(fontpath, 50)    
    while True:
        ret, frame = input.read()
        nums += 1    
        if not ret:
            break

        print("---------------读取第"+str(nums)+"帧")
        
        frame_start_t = time.time()
        # cv2 读取图片转换为 PIL 格式 转换为 Tensor
        # img = torchvision.transforms.ToTensor()(Image.open(img_path).convert(mode="RGB"))
        frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        img = torchvision.transforms.ToTensor()(frame_pil.convert(mode="RGB"))
        # NEW 创建一个可用来对其进行draw的对象
        draw = ImageDraw.Draw(frame_pil)

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
                        # cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 0, 255), 2)
                        # cv2.putText(frame, cls_pred_type, (x1,y1+50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, 4)
                        # NEW 更新 typeDict
                        typeChinese = map[classes[int(cls_pred)]]
                        if typeChinese in typeDict:
                            typeDict[typeChinese] = typeDict[typeChinese]+1
                        else:
                            typeDict[typeChinese] = 1
                        objects.append({'category': map[classes[int(cls_pred)]], 'score': 848.0, 'bbox': {'xmin': x1, 'ymin': y1, 'ymax': y2, 'xmax': x2}})
        classify_t = time.time()
        print("进行物体分类和图片标注，用时："+str(classify_t-detect_t))
        # NEW 绘制标注，更新list
        j = 0
        print(lastTypeDict)
        for key in lastTypeDict:
            if lastTypeDict[key] >= 2:
                draw.text((500, 500+j*50), key, font=font, fill=(255, 255, 255)) 
                # cv2.putText(frame, key, (500,500+50*j), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1, 4)
                # cv2.putText(frame, map[classes[int(cls_pred)]], (500,500+50*j), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1, 4)
                j = j+1
        frame = np.array(frame_pil)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            # break
        # 写每一帧
        output.write(frame)

        # 更新：该帧对应的检测结果
        train_results["imgs"][nums] = {"objects": objects}
        
        frame_end_t = time.time()
        print("该帧总用时："+str(frame_end_t-frame_start_t))

        # NEW 周期循环
        period = (period+1)%4
        if period == 0:
            lastTypeDict = typeDict
            typeDict = dict()


    input.release()
    cv2.destroyAllWindows()
    end_t = time.time()
    print("\n\n该视频总用时："+str(time.time()-start_t)+"  \n处理帧数："+str(nums))
    # 输出：所有帧的检测结果写到 json 文件中去
    with open(output_result_path, "w") as file_object:
        json.dump(train_results, file_object, ensure_ascii=False)
