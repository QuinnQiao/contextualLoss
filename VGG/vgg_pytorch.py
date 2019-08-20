import numpy as np
import torch 
import torch.nn as nn
import scipy.io as io

class VGG(nn.Module):
    def __init__(self, pretrained_model_path):
        super(VGG, self).__init__()
        self.model = []
        # 00 conv1-1
        self.model.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        # 01 relu
        self.model.append(nn.ReLU(inplace=True))
        # 02 conv1-2
        self.model.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        # 03 relu
        self.model.append(nn.ReLU(inplace=True))
        # 04 pool
        self.model.append(nn.AvgPool2d(2, stride=2))
        # 05 conv2-1
        self.model.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        # 06 relu
        self.model.append(nn.ReLU(inplace=True))
        # 07 conv2-2
        self.model.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        # 08 relu
        self.model.append(nn.ReLU(inplace=True))
        # 09 pool
        self.model.append(nn.AvgPool2d(2, stride=2))
        # 10 conv3-1
        self.model.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        # 11 relu
        self.model.append(nn.ReLU(inplace=True))
        # 12 conv3-2
        self.model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        # 13 relu
        self.model.append(nn.ReLU(inplace=True))
        # 14 conv3-3
        self.model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        # 15 relu
        self.model.append(nn.ReLU(inplace=True))
        # 16 conv3-4
        self.model.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        # 17 relu
        self.model.append(nn.ReLU(inplace=True))
        # 18 pool
        self.model.append(nn.AvgPool2d(2, stride=2))
        # 19 conv4-1
        self.model.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        # 20 relu
        self.model.append(nn.ReLU(inplace=True))
        # 21 conv4-2
        self.model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # 22 relu
        self.model.append(nn.ReLU(inplace=True))
        # 23 conv4-3
        self.model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # 24 relu
        self.model.append(nn.ReLU(inplace=True))
        # 25 conv4-4
        self.model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # 26 relu
        self.model.append(nn.ReLU(inplace=True))
        # 27 pool
        self.model.append(nn.AvgPool2d(2, stride=2))
        # 28 conv5-1
        self.model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # 29 relu
        self.model.append(nn.ReLU(inplace=True))
        # 30 conv5-2
        self.model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # 31 relu
        self.model.append(nn.ReLU(inplace=True))
        # 32 conv5-3
        self.model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # 33 relu
        self.model.append(nn.ReLU(inplace=True))
        # 34 conv5-4
        self.model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        # 35 relu
        self.model.append(nn.ReLU(inplace=True))
        
        self.model = nn.Sequential(*self.model)
        self.assign_weights(pretrained_model_path)
        self.disable_grad()
        
        self.map = {'1': 0, '2': 1, '3': 2, '4': 3}

    def _loadmat(self, pretrained_model_path):
        vgg_layers = io.loadmat(pretrained_model_path)['layers'][0]
        return vgg_layers
    
    def _assign_weight(self, index, vgg_layers):
        for i in index:
            weight, bias = self._get_weight(vgg_layers, i)
            self.model[i].weight.data = torch.as_tensor(weight)
            self.model[i].bias.data = torch.as_tensor(bias)

    def _get_weight(self, vgg_layers, layer):
        weight = vgg_layers[layer][0][0][0][0][0] # k_h * k_w * c_in *c _out
        bias = vgg_layers[layer][0][0][0][0][1] # 1 * c_out
        w_ = weight.transpose((3,2,0,1)) # c_out * c_in * k_h * k_w
        b_ = bias.reshape(-1) # c_out
        return w_, b_

    def assign_weights(self, pretrained_model_path):
        vgg_layers = self._loadmat(pretrained_model_path)
        index = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        self._assign_weight(index, vgg_layers)

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, layers=['4']):
        conv1_2 = self.model[:3]
        conv2_2 = self.model[3:8]
        conv3_2 = self.model[8:13]
        conv4_2 = self.model[13:22]
        f1 = conv1_2(x)
        f2 = conv2_2(f1)
        f3 = conv3_2(f2)
        f4 = conv4_2(f3)
        full_feats = [f1, f2, f3, f4]
        feats = []
        for layer in layers:
            feats.append(full_feats[self.map[layer]])
        return feats

def vgg_preprocess(img):
    # [-1,1] --> [0,255]
    img = (img + 1) * 255 * 0.5
    mean = torch.tensor([123.680, 116.779, 103.939], dtype=img.dtype, device=img.device).view(1,3,1,1)
    img = img - mean
    return img
