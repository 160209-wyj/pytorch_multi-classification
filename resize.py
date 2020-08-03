import os
import cv2
import tqdm
read_dire = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/douy_data_yuanchicun/4-3368/'
save_dire = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/douy_data_resize/4-3368/'
path = os.listdir(read_dire)
for file in tqdm.tqdm(path):
    img = cv2.imread(read_dire+file)
    img = cv2.resize(img,(224,224))
    cv2.imwrite('{}{}'.format(save_dire,file),img)