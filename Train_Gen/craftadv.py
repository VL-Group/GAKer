import os
from typing import Dict
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from .Generator import Generator
from .gaussian_smoothing import get_gaussian_kernel
import numpy as np
import torch.nn as nn
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t

class CustomDataset(Dataset):
    def __init__(self, img_dir, target_label,transform=None):
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
        self.transform = transform
        self.target_label = target_label

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.target_label  
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img_name = img_path.split('/')[-1]
        return img, label ,img_name


class CustomDenseNet121(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.classifier = original_model.classifier
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


class CustomResnet50(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def craftadv(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    scale_size = 256
    img_size = 224

    val_transform = transforms.Compose([transforms.Resize(scale_size),transforms.CenterCrop(img_size),transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.Resize(scale_size),transforms.CenterCrop(img_size),transforms.ToTensor()])

    target_select = modelConfig["target_select"]
    set_targets = modelConfig["set_targets"]
    unknown = modelConfig["unknown"]

    if unknown=='False':
        src = "source samples for known classes test"
        val_set = torchvision.datasets.ImageFolder(src, transform=val_transform)
    elif unknown=='True':
        src = "source samples for unknown classes test"
        val_set = torchvision.datasets.ImageFolder(src, transform=val_transform)



    if set_targets=='targets_200_cossimilar':
        targets = [22, 30, 43, 51, 53, 67, 76, 84, 107, 111, 116, 139, 156, 163, 174, 191, 194, 199, 228, 241, 251, 288, 301, 310, 313, 323, 324, 354,393, 398, 399, 401, 405, 418, 419, 420, 422, 428, 429, 439, 441, 451, 455, 457, 465, 467, 478, 480, 481, 488, 489, 490, 493, 496, 498, 499, 500, 507, 508, 514, 515, 519, 523, 530, 532, 533, 539, 540, 550, 552, 553, 557, 565, 566, 575, 576, 579, 583, 588, 592, 593, 594, 599, 601, 604, 605, 606, 607, 608, 611, 614, 622, 627, 640, 644, 646, 647, 659, 660, 666, 668, 674, 678, 683, 684, 687, 688, 691, 694, 700, 704, 712, 714, 715, 722, 726, 729, 738, 739, 740, 741, 749, 751, 761, 766, 769, 772, 773, 783, 785, 789, 790, 793, 794, 796, 798, 800, 807, 815, 822, 825, 826, 831, 843, 844, 851, 853, 854, 855, 858, 860, 862, 863, 869, 876, 877, 879, 880, 884, 888, 891, 897, 898, 901, 903, 904, 908, 910, 912, 914, 916, 918, 919, 924, 925, 927, 931, 932, 933, 934, 937, 938, 943, 946, 950, 952, 954, 958, 959, 961, 963, 971, 974, 977, 979, 980, 984, 985, 995, 996]
    elif set_targets=='targets_1000':
        targets = list(range(1000))
    else:
        print('please choose target')
        quit()
    print(targets)

    if unknown=='True':
        targets_num = [i for i in range(1000)]
        for i in targets:
            targets_num.remove(i)
        targets = targets_num
    elif unknown=='False':
        targets = targets
    else:
        print('please choose True or False')
        quit()
    print(targets)

    if modelConfig["Source_Model"] == "ResNet50": 
        original_model = torchvision.models.resnet50(pretrained=True)#######固定了模型res50
        feature_extraction = CustomResnet50(original_model)
        feature_extraction = feature_extraction.eval().to(device)
        feature_channel = 2048
        source_name = 'resnet50'
    elif modelConfig["Source_Model"] == "DenseNet121":
        original_model = torchvision.models.densenet121(pretrained=True)
        feature_extraction = CustomDenseNet121(original_model)
        feature_extraction = feature_extraction.eval().to(device)
        feature_channel = 1024
        source_name = 'densenet121'
    elif modelConfig["Source_Model"] == "vgg19bn":
        vgg19bn = torchvision.models.vgg19_bn(pretrained=True).eval().to(device)
        feature_channel = 4096
        global hook_output
        hook_output = None
        def hook(module, input, output):
            global hook_output
            hook_output = output
        handle = vgg19bn.classifier[5].register_forward_hook(hook)
        source_name = 'vgg19_bn'

    generator = Generator( num_target=len(targets), ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],num_res_blocks=modelConfig["num_res_blocks"],feature_channel_num=feature_channel)
    ckpt = torch.load(os.path.join(modelConfig["Generator_save_dir"], modelConfig["test_load_weight"]), map_location=device)

    generator.load_state_dict(ckpt,strict=False)
    print("model load weight done.")
    ran_best = modelConfig["ran_best"]
    generator.eval().to(device)

    eps = 16.0/255
    print('eps:',eps*255)

    for target in targets:
        print('##########################')
        target_label = target
        numtotargetname = []
        with open('imagenet_numtotarget.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()  
                numtotargetname.append(parts[1])  
        target_name = numtotargetname[target_label]


        source_img_dir = src + target_name
        source_set = CustomDataset(source_img_dir, target_label, val_transform)

        print('target==>',target,'  num:',len(source_set))
        source_set = DataLoader(source_set, batch_size=10, shuffle=True, num_workers=12,pin_memory=True)

        ####################################################################################################
        if target_select=='10':
            target_img_dir = 'target images folder'+target_name
            target_set = DataLoader(CustomDataset(target_img_dir,target_label, target_transform), batch_size=10, shuffle=True, num_workers=12, pin_memory=True)
        
        elif target_select=='1':
            target_img_dir = 'target images folder'+target_name
            top1datasets = CustomDataset(target_img_dir, target_label, target_transform)
            target_samples = []
            if unknown=='False':
                target_num= 10
            elif unknown=='True':
                target_num= 10
            while len(target_samples) < target_num:
                sample = random.choice(top1datasets)
                img_name, label,_ = sample
                target_samples.append(sample)
            print(len(target_samples))
            target_set = DataLoader(target_samples, batch_size=10, shuffle=True, num_workers=12, pin_memory=True)
        target_iter = iter(target_set)
        ####################################################################################################
        with torch.no_grad():


            for imgs,labels,file_name in source_set:
                imgs = imgs.to(device)
                kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).cuda()

                try:
                    imgs_target, labels_target, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_set)
                    imgs_target, labels_target , _ = next(target_iter)

                imgs_target=imgs_target.to(device)

                if ran_best== 'random':
                    if modelConfig["Source_Model"] == "vgg19bn":
                        output = vgg19bn(normalize(imgs_target.clone().detach().to(device)))
                        target_feature = hook_output
                    else:
                        target_feature=feature_extraction(normalize(imgs_target.clone().detach().to(device)))
                    output_to_mix = target_feature.squeeze()
                elif ran_best== 'best':
                    print('not used')

                else:
                    print('please choose random or best')

                perturbated_imgs = generator(imgs, mix=output_to_mix)

                perturbated_imgs = kernel(perturbated_imgs)



                adv = torch.min(torch.max(perturbated_imgs, imgs-eps), imgs + eps)
                adv = torch.clamp(adv, 0, 1.0)

                save_adv_path = "..."
                os.makedirs(save_adv_path, exist_ok=True)
                adv_img = (adv.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
                for j in range(adv_img.shape[0]):
                    Image.fromarray(adv_img[j]).save(save_adv_path + "/" + file_name[j])


