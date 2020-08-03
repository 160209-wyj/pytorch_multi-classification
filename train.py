from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import tqdm
import numpy as np
from evaluation import Evaluations
from tensorboardX import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



class vgg_junjie(nn.Module):
    def __init__(self,num_classes=20):
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

CLASSES = ['10map', '11controlled_knife', '12firearms_and_ammunition', '13warship',
           '14tank', '15military_aircraft', '16guided_missile', '17tattoo','18smoking','19gamble', '1bedin','20Normal', '2blood',
           '3ruins_train', '4Crisis_event', '5Chinese_national_flag',
           '6Public_inspection_vehicles', '7fire_fighting_truck', '8ambulance',
           '9policeman_uniform']
# 定义tensotboard
writer = SummaryWriter('runs/logs_20_vgg16_1.4')

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            gt_list = []
            pred_list = []
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in tqdm.tqdm(dataloders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()


                
                # print('loss:{:.4f},acc:{:.4f}'.format(loss,torch.sum(preds == labels.data).to(torch.float32)),end='',flush=True)

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

                label = labels.squeeze(-1)
                gt_list.extend(label.detach().cpu().numpy().tolist())
                pred_list.extend(preds.detach().cpu().numpy().tolist())
            
            pred_np = np.array(pred_list)
            gt_np = np.array(gt_list)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            evals = Evaluations(pred_np, gt_np, CLASSES)
            print(evals)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            evals.writelog(writer, path=phase, global_step=epoch)
            writer.add_scalar('{}_lr:'.format(phase), lr, global_step=epoch)
            writer.add_scalar('{}_loss:'.format(phase), epoch_loss, global_step=epoch)
            writer.add_scalar('{}_acc:'.format(phase), epoch_acc, global_step=epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                writer.add_scalar('{}_loss:'.format(phase), epoch_loss, global_step=epoch)
                writer.add_scalar('{}_acc:'.format(phase), best_acc, global_step=epoch)
                torch.save(best_model_wts, './model/20_Epoch{}_Loss_{:.4f}.pkl'.format(epoch, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=(0,1),contrast=(0,1),saturation=(0,1),hue=(-0.5,0.5)),
            # transforms.RandomGrayscale(p=0.4),
            # transforms.RandomRotation(90),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file
    # data_dir = '/home/zhangjunjie/keras-applications-master/data/data'
    data_dir = '/home/zhangjunjie/juemiwenijan/ImageClassification-PyTorch/level1/huang_data/data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=8) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # use gpu or not
    use_gpu = torch.cuda.is_available()

    # get model and replace the original fc layer with your fc layer
    # model_ft = models.vgg16_bn(pretrained=False)
    model_ft=vgg_junjie(num_classes=20)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    # for m in model_ft.modules():
    #     print(m)
    if use_gpu:
        # model_ft = nn.DataParallel(model_ft)
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # ignored_params = list(map(id, model_ft.feature.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, model_ft.parameters())

    

    # 余弦退火
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10, eta_min=10e-5)
    #学习率衰减
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft,gamma=0.9)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=27)
    torch.save(model_ft,"model/20bk_best_vgg16_1.4.pkl")