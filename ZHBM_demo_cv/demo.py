"""
    该文件用于算法模型交付，算法工程人员根据该demo，部署算法。
    Author:
    Version:
    Date:
"""
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from res50_model import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"


class Demo:
    """ 算法模型推理demo
        用于算法部署
    """
    
    def __init__(self):
        """ 类初始化
        """
        # TODO
        self.img_width = 224
        self.img_height = 224
        self.model_path = "./model/7_Epoch4_Loss_0.0000.pkl"
        self.dire='./images/'
        self.num_classes = 7
        self.class_names = ['gz', 'normal', 'file', 'gctz',  
                     'tyy', 'xsq','red_file']

    def load_m(self):
        """ 加载模型
        详细描述。

        Args:
            model_path: 模型文件路径
        Returns:
            None.
        Raises:
            None.
        """
        # TODO
        
        model = gcn_resnet50(num_classes=self.num_classes,pretrained=True)
        checkpoint  = torch.load(self.model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def inference(self):
        """ 模型推理
        支持批量推理。

        Args:
            data_paths: 图片路径数组
        Returns:
            None.
        Raises:
            None.
        """        
        
        # 返回结果为数组，对应data_paths中的每张图片
        # TODO
        plt.ion()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        path = os.listdir(self.dire)
        for file in path:
            image_cv = cv2.imread(self.dire+file)
            image_cv = cv2.resize(image_cv, (224, 224))
            miu = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = np.array(image_cv, dtype=float) / 255.
            r = (img_np[:, :, 0] - miu[0]) / std[0]
            g = (img_np[:, :, 1] - miu[1]) / std[1]
            b = (img_np[:, :, 2] - miu[2]) / std[2]
            img_np_t = np.array([r, g, b])

            # img_np_nchw = np.expand_dims(img_np_t, axis=0)
            image_tensor = torch.from_numpy(img_np_t.astype('float32'))
            # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
            image_tensor.unsqueeze_(0)
            # 没有这句话会报错
            image_tensor = image_tensor.to(device)
            model = self.load_m()
            model = model.to('cuda')
            out = model(image_tensor)
            ## 4. 标签和分数输出
            _, indices = torch.sort(out, descending=True)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            max_val =percentage.argmax()
            print('filename :',file,self.class_names[max_val], percentage[max_val].item())
            
    def release(self):
        """ 模型释放(若需要)
        支持批量推理。

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """        
        # TODO


def main():
    """ 自测试代码
    """
    
    demo = Demo()
    # TODO 实现Demo运行的最小代码
    demo.load_m()
    demo.inference()
    # demo.release()

if __name__ == '__main__':
    main()