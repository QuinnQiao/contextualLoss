import torch
import torchvision.transforms as transforms
from PIL import Image
import os, sys, argparse
from VGG.vgg_pytorch import *
from CX.CXLoss import *


def get_img(img_path, transform, device):
    imgs = []
    files = os.listdir(img_path)
    for file in files:
        img = Image.open(os.path.join(img_path, file))
        img = transform(img)
        imgs.append(img.unsqueeze(0).cuda(device))
    return imgs

def get_id(ids_str):
    ids = ids_str.split(',')
    return ids

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder_A', type=str, help="input image folder")
parser.add_argument('--input_folder_B', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output log folder")
parser.add_argument('--gpu_id', type=int, default=0, help="which gpu to use")
parser.add_argument('--centercrop', action='store_true', default=False, help="if centercrop when load datase")
parser.add_argument('--resize_size', type=int, default=256, help="resize image to this size")
parser.add_argument('--pretrained_vgg_path', type=str, help="the path of the pretrained vgg checkpoint")
parser.add_argument('--band_width_src', type=float, default=0.1, help="the band width paramater for source image")
parser.add_argument('--band_width_tgt', type=float, default=0.2, help="the band width paramater for target image")
parser.add_argument('--vgg_layer_src', type=str, default='4', help="the layer of VGG tp extracr features for source image")
parser.add_argument('--vgg_layer_tgt', type=str, default='2,3,4', help="the layer of vGG to extract features for target image")
opts = parser.parse_args()

transform_list = [transforms.Resize(opts.resize_size)]
if opts.centercrop:
    transform_list.append(transforms.CenterCrop((opts.resize_size, opts.resize_size)))
transform_list.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(transform_list)

device = 'cuda:%d'%opts.gpu_id

vgg = VGG(opts.pretrained_vgg_path).cuda(device)
print('vgg loaded')

input_A = get_img(opts.input_folder_A, transform, device)
input_B = get_img(opts.input_folder_B, transform, device)
print('img loaded')

with open(os.path.join(opts.output_folder, 'results.txt'), 'w') as f:
    # within-domain
    f.write('Domain A:\n')
    for i in range(len(input_A)-1):
        content_1 = vgg(vgg_preprocess(input_A[i]), layers=get_id(opts.vgg_layer_src))
        style_1 = vgg(vgg_preprocess(input_A[i]), layers=get_id(opts.vgg_layer_tgt))
        for j in range(i+1, len(input_A)):
            content_2 = vgg(vgg_preprocess(input_A[j]), layers=get_id(opts.vgg_layer_src))
            style_2 = vgg(vgg_preprocess(input_A[j]), layers=get_id(opts.vgg_layer_tgt))
            content_losses = []
            for _, (feat_1, feat_2) in enumerate(zip(content_1, content_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_src)
                content_losses.append(loss.cpu())
            style_losses = []
            for _, (feat_1, feat_2) in enumerate(zip(style_1, style_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_tgt)
                style_losses.append(loss.cpu())
            f.write(str(i) + '-' + str(j) + ': ' + '\n')
            f.write('Content:' + str(content_losses) + ', Style: ' + str(style_losses) + '\n')           
    print('A done')
    f.write('\nDomain B:\n')
    for i in range(len(input_B)-1):
        content_1 = vgg(vgg_preprocess(input_B[i]), layers=get_id(opts.vgg_layer_src))
        style_1 = vgg(vgg_preprocess(input_B[i]), layers=get_id(opts.vgg_layer_tgt))
        for j in range(i+1, len(input_B)):
            content_2 = vgg(vgg_preprocess(input_B[j]), layers=get_id(opts.vgg_layer_src))
            style_2 = vgg(vgg_preprocess(input_B[j]), layers=get_id(opts.vgg_layer_tgt))
            content_losses = []
            for _, (feat_1, feat_2) in enumerate(zip(content_1, content_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_src)
                content_losses.append(loss.cpu())
            style_losses = []
            for _, (feat_1, feat_2) in enumerate(zip(style_1, style_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_tgt)
                style_losses.append(loss.cpu())
            f.write(str(i) + '-' + str(j) + ': ' + '\n')
            f.write('Content:' + str(content_losses) + ', Style: ' + str(style_losses) + '\n')           
    print('B done')
    # cross-domain
    f.write('\nDomain A-B:\n')
    for i in range(len(input_A)-1):
        content_1 = vgg(vgg_preprocess(input_A[i]), layers=get_id(opts.vgg_layer_src))
        style_1 = vgg(vgg_preprocess(input_A[i]), layers=get_id(opts.vgg_layer_tgt))
        for j in range(len(input_B)):
            content_2 = vgg(vgg_preprocess(input_B[j]), layers=get_id(opts.vgg_layer_src))
            style_2 = vgg(vgg_preprocess(input_B[j]), layers=get_id(opts.vgg_layer_tgt))
            content_losses = []
            for _, (feat_1, feat_2) in enumerate(zip(content_1, content_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_src)
                content_losses.append(loss.cpu())
            style_losses = []
            for _, (feat_1, feat_2) in enumerate(zip(style_1, style_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_tgt)
                style_losses.append(loss.cpu())
            f.write('A-' + str(i) + '~B-' + str(j) + ': ' + '\n')
            f.write('Content:' + str(content_losses) + ', Style: ' + str(style_losses) + '\n')           
    print('A-B done')
