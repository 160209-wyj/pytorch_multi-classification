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


img_size = 224
# Batch size (you should tune it based on your memory)
batch_size = 32
testing_folder = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/data/train'

val_datagen = ImageDataGenerator(
    rescale=1. / 255)
validation_generator = val_datagen.flow_from_directory(
    testing_folder,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')
from sklearn.metrics import classification_report
from sklearn import metrics
steps = 1
predictions = model.predict_generator(validation_generator, verbose=1)
# print('predictions:',predictions)
val_preds = np.argmax(predictions, axis=-1)
val_trues = validation_generator.classes
# print('val_trues:',val_trues)
# print('val_preds:',val_preds)
# cm = metrics.confusion_matrix(val_trues, val_preds)

# labels = validation_generator.class_indices.keys()
# print('val_trues:',val_trues)
# print('val_preds:',val_preds)
# print('labels:',labels)
# precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds)

from sklearn.metrics import classification_report
import numpy as np


print(classification_report(val_trues, val_preds))
