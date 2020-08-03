#from dlr_implementation import Adam_dlr
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.05
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config)) 
import tqdm
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
model = load_model('./keras_model/best_keras_1.0.h5', custom_objects={'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D,'getRecall':getRecall,'getPrecision':getPrecision},compile=False)
print(model.summary())
#print('+++++++++++++++++++++++++++++++++++++++')
dic = {0:'file', 1:'normal'}

test_path = "/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/test_data/"
# test_path = "/home/huangxin/work/testcode/MNasNet-Keras-Tensorflow-master/douyin_crop_data/"
test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224),classes=dic,batch_size=2,class_mode='categorical', shuffle=False,)
print("1")
print('test_datagen:',test_datagen)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False)

pred = model.predict_generator(test_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
print("1")

predicted_class_indices = np.argmax(pred, axis=1)

test_generators = []
for i in test_generator.filenames:
    test_generators.append(i)
t = 0
for i,key in enumerate(tqdm.tqdm(predicted_class_indices)):
    test_generator = test_generators[i].split('\\')
    test_generator = '/'.join(test_generator)
    img = cv2.imread(test_path+'/'+test_generator)
    save_path = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/save_test_data'
    if key == 0:
        if pred[i][key] >= 0.7:
            cv2.imwrite('{}/{}/{}.jpg'.format(save_path,dic[key],pred[i][key]),img)
    # if key == 1:
    #     if pred[i][key] >= 0.7:
    #         cv2.imwrite('{}/{}/{}.jpg'.format(save_path,dic[key],pred[i][key]),img)
    
       