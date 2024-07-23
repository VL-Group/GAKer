
import os
from typing import Dict
import numpy as np
import torchvision
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from .Generator import Generator
import torch.nn as nn
import torch.nn.functional as F
import time
import timm
from scipy.spatial.distance import cosine
from .gaussian_smoothing import get_gaussian_kernel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import random


def get_device_count():
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = cuda_visible_devices.split(',')
    return len(devices)

def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t

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
        # self.fc = original_model.fc
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
        # x = self.fc(x)
        return x


class CustomVgg19(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = original_model.classifier
        init_weights = original_model.init_weights
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        print(x.shape)
        return x

def seed_torch(seed=0):

    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
def train(modelConfig: Dict):
    time_start = time.time()
    device_count = get_device_count()
    print('gpu_num',device_count)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    if modelConfig["Source_Model"] == "ResNet50" or modelConfig["Source_Model"] == "DenseNet121" or modelConfig["Source_Model"] == "vgg19bn":
        scale_size = 256
        img_size = 224

    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.03, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ])
    target_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    ])

    src = "source benign images folder"
    src_target = "target images folder"

    train_set = torchvision.datasets.ImageFolder(src, train_transform)
    target_set = torchvision.datasets.ImageFolder(src_target, train_transform )
    set_targets = modelConfig["set_targets"]

    # choose target class of ImageNet
    if set_targets=='targets_200_random':
        targets = [0, 3, 6, 7, 9, 11, 15, 19, 31, 33, 41, 43, 46, 62, 64, 70, 75, 83, 85, 88, 92, 97, 104, 113, 115, 116, 118, 119, 120, 121, 128, 149, 153, 161, 179, 183, 184, 190, 199, 204, 208, 213, 216, 218, 221, 223, 225, 237, 240, 248, 252, 260, 265, 276, 280, 283, 294, 297, 298, 306, 309, 310, 318, 323, 339, 343, 350, 355, 358, 370, 375, 379, 384, 388, 391, 393, 416, 418, 440, 442, 454, 464, 465, 469, 477, 486, 489, 490, 494, 495, 501, 503, 512, 514, 516, 519, 520, 522, 529, 539, 541, 543, 548, 552, 554, 560, 561, 565, 567, 568, 571, 575, 578, 587, 590, 596, 597, 600, 601, 606, 607, 611, 612, 637, 640, 645, 655, 664, 673, 678, 681, 687, 690, 693, 696, 697, 700, 704, 705, 713, 719, 721, 726, 729, 734, 736, 738, 744, 753, 754, 758, 760, 764, 765, 768, 783, 784, 794, 796, 803, 807, 812, 816, 821, 823, 825, 827, 837, 841, 845, 855, 859, 868, 870, 875, 878, 883, 889, 897, 901, 908, 909, 910, 927, 928, 933, 934, 942, 943, 953, 958, 969, 972, 974, 977, 986, 990, 991, 993, 998]
    elif set_targets=='all_classes':
        targets = list(range(1000))

    target_samples = []
    for img_name, label in target_set.samples:
        if label in targets:
            target_samples.append((img_name, label))
    target_set.samples = target_samples


    # seed_torch(0)
    if modelConfig["Source_Model"] == "ResNet50": 
        original_model = torchvision.models.resnet50(pretrained=True)
        feature_extraction = CustomResnet50(original_model)
        feature_extraction = feature_extraction.eval().to(device)
        feature_channel = 2048
    elif modelConfig["Source_Model"] == "DenseNet121":
        original_model = torchvision.models.densenet121(pretrained=True)
        feature_extraction = CustomDenseNet121(original_model)
        feature_extraction = feature_extraction.eval().to(device)
        feature_channel = 1024
    elif modelConfig["Source_Model"] == "vgg19bn":
        vgg19bn = torchvision.models.vgg19(pretrained=True).eval().to(device)
        feature_channel = 4096
        global hook_output
        hook_output = None
        def hook(module, input, output):
            global hook_output
            hook_output = output
        handle = vgg19bn.classifier[5].register_forward_hook(hook)


    generator = Generator( num_target=len(targets), ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],num_res_blocks=modelConfig["num_res_blocks"],feature_channel_num=feature_channel)
    generator = generator.to(device)

    dataloader = DataLoader(train_set, batch_size=modelConfig["batch_size"], shuffle=True,num_workers=12, pin_memory=True)
    dataloader_target = DataLoader(target_set, batch_size=modelConfig["batch_size"], shuffle=True,num_workers=12, pin_memory=True)

    learning_rate = modelConfig["lr"]

    print('learining rate ',learning_rate)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=learning_rate, weight_decay=5e-5)#GAN的优化器

    if not os.path.exists(modelConfig["Generator_save_dir"]):
        os.makedirs(modelConfig["Generator_save_dir"])


    eps = 16/255
    print('eps:',eps*255)
    kernel = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1).to(device)

    for e in range(modelConfig["epoch"]):
        
        iteration = 0

        target_iter = iter(dataloader_target)
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()
                
                try:
                    imgs_target, labels_target = next(target_iter)
                except StopIteration:
                    target_iter = iter(dataloader_target)
                    imgs_target, labels_target = next(target_iter)
                images = images.to(device)    
                imgs_target=imgs_target.to(device)

                if imgs_target.shape[0]!=modelConfig["batch_size"] or images.shape[0]!=modelConfig["batch_size"]:
                    continue

                with torch.no_grad():
                    if modelConfig["Source_Model"] == "ResNet50" or modelConfig["Source_Model"] == "DenseNet121":
                        target_fea=feature_extraction(normalize(imgs_target.clone().detach())).squeeze()
                    if modelConfig["Source_Model"] == "vgg19bn":
                        output = vgg19bn(normalize(imgs_target.clone().detach()))
                        target_fea = hook_output
                output_to_mix = target_fea
                target_feature = []
                for i in range(labels_target.shape[0]):
                    target_feature.append(target_fea[i])

                target_feature = torch.tensor(np.array([item.cpu().detach().numpy() for item in target_feature])).to(device)
                mask = torch.ne(labels,labels_target).long().to(device)
                perturbated_imgs = kernel(generator(images,mix=output_to_mix))
                adv = torch.min(torch.max(perturbated_imgs, images-eps), images + eps)
                adv = torch.clamp(adv, 0.0, 1.0)
                
                if modelConfig["Source_Model"] == "ResNet50" or modelConfig["Source_Model"] == "DenseNet121":
                    adv_feature = feature_extraction(normalize(adv))
                elif modelConfig["Source_Model"] == "vgg19bn":
                    output = vgg19bn(normalize(adv))
                    adv_feature = hook_output

                if modelConfig["Source_Model"] == "ResNet50" or modelConfig["Source_Model"] == "DenseNet121" or modelConfig["Source_Model"] == "vgg19bn":
                    adv_feature = adv_feature.squeeze()
                    loss = 1 - torch.cosine_similarity(adv_feature, target_feature, dim=1)

                loss = mask*loss

                noise = adv - images
                if modelConfig["Source_Model"] == "ResNet50" or modelConfig["Source_Model"] == "DenseNet121" :
                    noise_feature = feature_extraction(normalize(noise)).squeeze()
                    loss_noise = 1 - torch.cosine_similarity(noise_feature, target_feature, dim=1)
                elif modelConfig["Source_Model"] == "vgg19bn":
                    output = vgg19bn(normalize(noise))
                    noise_feature = hook_output.squeeze()
                    loss_noise = 1 - torch.cosine_similarity(noise_feature, target_feature, dim=1)

                loss_noise = mask*loss_noise*0.5
                loss = loss+loss_noise


                loss = (loss.sum())/images.shape[0]

                loss.backward()
                optimizer.step()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item()
                })
                if iteration % 1000 == 0:
                    with open(os.path.join(modelConfig["Generator_save_dir"], 'loss' + ".txt"), 'a') as f:
                        f.write(f'epoch {e}:  iter {iteration}: loss {loss.item()}\n')
                iteration += 1
        
        torch.cuda.empty_cache()
        if (e+1)%1==0:
            torch.save(generator.state_dict(), os.path.join(modelConfig["Generator_save_dir"], 'ckpt_' + str(e) + "_" + modelConfig["Source_Model"] +"_.pt"))
    time_end = time.time()
    print('time cost'+':  ',time_end-time_start,'s')


