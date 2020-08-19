from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class GCNResnet_junjie(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file='data/voc/voc_adj.pkl'):
        super(GCNResnet_junjie, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        # self.pooling = nn.MaxPool2d(14, 14)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.relu = nn.LeakyReLU(0.2)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature):
        feature = self.features(feature)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature= self.fc(feature)
        return feature

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr},
                # {'params': self.gc2.parameters(), 'lr': lr},
                ]




def gcn_resnet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    return GCNResnet_junjie(model, num_classes)

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
        out = self.feature(input)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         # if isinstance(m, nn.Conv2d):
    #         #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         #     if m.bias is not None:
    #         #         nn.init.constant_(m.bias, 0)
    #         # elif isinstance(m, nn.BatchNorm2d):
    #         #     nn.init.constant_(m.weight, 1)
    #         #     nn.init.constant_(m.bias, 0)
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)



plt.ion()   # interactive mode

# 图片路径
# save_path = '/home/guomin/.cache/torch/checkpoints/resnet18-customs-angle.pth'
model_path = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/model/20bk_best_vgg16_1.4.pkl'
 
# ------------------------ 加载数据 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定义预训练变换
preprocess_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
 
class_names = ['10map', '11controlled_knife','12firearms_and_ammunition','13warship','14tank','15military_aircraft','16guided_missile','17tattoo',
                '18smoking','19gamble','1bedin','20Normal','2blood','3ruins_train','4Crisis_event','5Chinese_national_flag','6Public_inspection_vehicles',
                '7fire_fighting_truck','8ambulance','9policeman_uniform']
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# ------------------------ 载入模型并且训练 --------------------------- #

# model = torch.load(model_path, map_location='cuda:0')
# model.eval()
# print(model)
#加载最终模型
# model = torch.load(model_path, map_location='cuda:0')
# model.eval()

#加载中间模型resnet50
num_classes = 7
model = gcn_resnet50(num_classes=num_classes,pretrained=True)
model.cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

#加载中间模型vgg16_bn
# num_classes = 7
# model = vgg_junjie(num_classes=num_classes)
# model.cuda()
# model = nn.DataParallel(model)
# model.load_state_dict(torch.load(model_path))
# model.eval()

dire = '/home/zhangjunjie/keras-applications-master/kafka_data/all_test_data/'
# dire = '/home/zhangjunjie/MobileNetV2-master/image_140_yuanchicun/images_highrisk/'

path = os.listdir(dire)
for file in tqdm.tqdm(path):

    image_PIL = Image.open(dire+file)
    if image_PIL.mode != 'RGB':
        print('image:',image_PIL,'image_path:',dire+file)
        image_PIL = image_PIL.convert("RGB")
        os.remove(dire+file)
        image_PIL.save(dire+file)
    #
    image_tensor = preprocess_transform(image_PIL)
    # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor.unsqueeze_(0)
    # 没有这句话会报错
    image_tensor = image_tensor.to(device)
    
    out = model(image_tensor)
    # 得到预测结果，并且从大到小排序
    _, indices = torch.sort(out, descending=True)
    # 返回每个预测值的百分数
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    max_val =percentage.argmax()
    
    # print(class_names[max_val], percentage[max_val].item())

    # save_path = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/dy_pytorch_save_data'
    save_path = '/home/zhangjunjie/MobileNetV2-master/img_crop'
    def read():
        if percentage[max_val].item() >= 70:
            img = cv2.imread(dire+file)
            cv2.imwrite('{}/{}/{}.jpg'.format(save_path,class_names[max_val],percentage[max_val].item()),img)
    if max_val == 0:
        read() 
    if max_val == 1:
        read()
    if max_val == 2:
        read()
    if max_val == 3:
        read()
    if max_val == 4:
        read()
    if max_val == 5:
        read()
    if max_val == 6:
        read()
    if max_val == 7:
        read()
    if max_val == 8:
        read()
    if max_val == 9:
        read()
    if max_val == 10:
        read()
    if max_val == 11:
        read()
    if max_val == 12:
        read()
    if max_val == 13:
        read()
    if max_val == 14:
        read()
    if max_val == 15:
        read()
    if max_val == 16:
        read()
    if max_val == 17:
        read()
    if max_val == 18:
        read()
    if max_val == 19:
        read()
    if max_val == 20:
        read()
    




