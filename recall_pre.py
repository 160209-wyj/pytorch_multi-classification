from evaluation import *
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


class vgg_junjie(nn.Module):
    def __init__(self,num_classes=20,init_weights=True):
        super(vgg_junjie,self).__init__()
        model_ft = models.vgg16_bn(pretrained=True)
        self.feature=model_ft.features  ## vgg features

        for p in self.parameters():
            p.requires_grad=True

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # if init_weights:
        #     self._initialize_weights()

    def forward(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        out = self.feature(input)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


model_path = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/model/20bk_best_vgg16_1.4.pkl'
data_dir = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/download_dy_data'


model = torch.load(model_path, map_location='cuda:0')

data_transforms = {
        'recall_test_data': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=(0,1),contrast=(0,1),saturation=(0,1),hue=(-0.5,0.5)),
            # transforms.RandomGrayscale(p=0.4),
            # transforms.RandomRotation(90),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}


image_datasets = {'recall_test_data': datasets.ImageFolder(os.path.join(data_dir, 'recall_test_data'),
                                            data_transforms['recall_test_data'])}

dataloader = {'recall_test_data': torch.utils.data.DataLoader(image_datasets['recall_test_data'],
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=8)}


#Your classes from 'A' to 'Z'
CLASSES = ['10map', '11controlled_knife', '12firearms_and_ammunition', '13warship',
           '14tank', '15military_aircraft', '16guided_missile', '17tattoo','18smoking','19gamble', '1bedin','20Normal', '2blood',
           '3ruins_train', '4Crisis_event', '5Chinese_national_flag',
           '6Public_inspection_vehicles', '7fire_fighting_truck', '8ambulance',
           '9policeman_uniform']

#list to put pred & groundtruth labels
pred_list = []
gt_list = []

#some loop to train your data
for  data in tqdm.tqdm(dataloader['recall_test_data']):
    inputs, labels = data

    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
  #get gt labels
    label = labels.squeeze(-1)
    gt_list.append(label.detach().cpu().numpy().tolist())
  
  #get pred labels
  
    pred_list.extend(preds.detach().cpu().numpy().tolist())

#transform list to np.ndarray
pred_np = np.array(pred_list)
gt_np = np.array(gt_list)

evals = Evaluations(pred_np,gt_np,CLASSES)
print(evals)
print('-------------------------------------------------------')
print(evals.average.precision())
print('-------------------------------------------------------')
print(evals.A.recall())
print('-------------------------------------------------------')