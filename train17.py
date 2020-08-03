"""
Train the MobileNet V2 model
"""
import os

import keras_applications
from keras.applications import mobilenet_v2
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#from keras.utils.training_utils import multi_gpu_model
from parallel_model import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config)) 
import sys
import argparse
import pandas as pd
# from mobilenet_v2 import MobileNetv2
from keras.optimizers import Adam
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model
from keras.callbacks.tensorboard_v2 import TensorBoard
import keras.backend as K
# from mobilenetv2_finetune import MobileNetV2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--classes",
        help="The number of classes of dataset.")
    # Optional arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=300,
        help="The number of train iterations.")
    parser.add_argument(
        "--weights",
        default=False,
        help="Fine tune with other weights.")
    parser.add_argument(
        "--tclasses",
        default=0,
        help="The number of classes of pre-trained model.")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs), int(args.classes), int(args.size), args.weights, int(args.tclasses))


def generate(batch, size):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    ptrain = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/data/train'
    pval = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/data/val'

    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        zca_whitening=True,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        #save_to_dir='/home/zhangjunjie/disk1/data/save',
        #save_prefix='data_up',
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2

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


def train(batch, epochs, num_classes, size, weights, tclasses):
    """Train the model.

    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
        tclasses, Integer, The number of classes of pre-trained model.
    """

    train_generator, validation_generator, count1, count2 = generate(batch, size)

    
    model = mobilenet_v2.MobileNetV2((size, size, 3), weights=weights, include_top = False, pooling = 'avg')
    
    predictions = Dense(num_classes, activation='softmax')(model.output)
    model = Model(inputs=model.input, outputs=predictions)
    #GPU_COUNT = 2 # 同时使用3个GPU
    #model = ParallelModel(model, GPU_COUNT)
    
        
    # model = multi_gpu_model(model, gpus=4)
    opt = Adam()
    earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', getRecall, getPrecision])
    log_filepath = "./keras_logs_crop1.1"
    board = TensorBoard(log_dir=log_filepath, histogram_freq=0, batch_size=32, write_graph=True,
                        write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    checkpoint_path = "keras_model/keras_red_1.1.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=False, mode="auto", save_best_only=True)

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=epochs,
        workers=4,
        use_multiprocessing=True,
        callbacks=[earlystop,board,checkpoint])

    if not os.path.exists('keras_model'):
        os.makedirs('keras_model')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('keras_model/hist_keras_red_1.1.csv', encoding='utf-8', index=False)
    model.save('keras_model/best_keras_red_1.1.h5')


if __name__ == '__main__':
    main(sys.argv)
