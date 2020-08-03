"""
    该文件用于算法模型交付，算法工程人员根据该demo，部署算法。
    Author:
    Version:
    Date:
"""
from keras.preprocessing.image import ImageDataGenerator
import os
import keras.backend as K
import shutil
import cv2
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import ImageFile
from keras.models import load_model
import keras
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        self.model_path = "./model/17_weights_1.3.h5"
        self.data_paths='./aaa/images/'
        self.dic = {0:'10map',1:'11controlled knife',2:'12firearms and ammunition',3:'13warship',4:'14tank',5:'15military aircraft',6:'16guided missile',7:'17Normal',
           8:'1bedin',9:'2blood',10:'3ruins_train',11:'4Crisis event',12:'5Chinese national flag',13:'6Public inspection vehicles',14:'7fire fighting truck',
           15:'8ambulance',16:'9policeman uniform'}

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
        def getPrecision(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
            N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
            TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
            FP=N-TN
            precision = TP / (TP + FP + K.epsilon())#TT/P
            return precision

        def getRecall(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
            P=K.sum(K.round(K.clip(y_true, 0, 1)))
            FN = P-TP #FN=P-TP
            recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
            return recall
        model = load_model(self.model_path, custom_objects={'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D,'getRecall':getRecall,'getPrecision':getPrecision})
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
        if self.data_paths == None:
            self.data_paths = []
        # 返回结果为数组，对应data_paths中的每张图片
        # TODO
        
        test_path = self.data_paths
        img_paths = os.listdir(self.data_paths)
        test = np.empty((len(img_paths), self.img_width, self.img_width, 3))
        count = 0
        for img_path in img_paths:
            img = image.load_img((test_path+img_path), target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            test[count] = img
            count += 1
        result = self.load_m().predict(test,batch_size=32)
        predicted_class_indices = np.argmax(result, axis=1)
        for i,key in enumerate(predicted_class_indices):
            print('class:',self.dic[key],'---score:',result[i][key])
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