import torch
from PIL import Image
import os, sys, argparse
from vgg_pytorch import *
from CXLoss import *


def get_img(img_path, transform, device):
    imgs = []
    files = os.listdir(img_path)
    for file in files:
        img = PIL.open(os.path.join(img_path, file))
        img = transform(img)
        imgs.append(img.cuda(device))
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

input_A = get_img(opts.input_folder_A, transform, device)
input_B = get_img(opts.input_folder_B, transform, device)

with open(os.path.join(opts.output_folder, 'results.txt'), 'w') as f:
    # within-domain
    f.write('Domain A:\n')
    for i in range(len(input_A)-1):
        for j in range(i+1, len(input_A)):
            feats_1 = vgg(vgg_preprocess(input_A[i]), layers=get_id(opts.vgg_layer_src))
            feats_2 = vgg(vgg_preprocess(input_A[j]), layers=get_id(opts.vgg_layer_src))
            losses = []
            for _, (feat_1, feat_2) in enumerate(zip(feats_1, feats_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_src)
                losses.append(loss.cpu())
            f.write(str(i) + '-' + str(j) + ': ' + str(losses) + '\n')
    f.write('\nDomain B:\n')
    for i in range(len(input_B)-1):
        for j in range(i+1, len(input_B)):
            feats_1 = vgg(vgg_preprocess(input_B[i]), layers=get_id(opts.vgg_layer_src))
            feats_2 = vgg(vgg_preprocess(input_B[j]), layers=get_id(opts.vgg_layer_src))
            losses = []
            for _, (feat_1, feat_2) in enumerate(zip(feats_1, feats_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_src)
                losses.append(loss.cpu())
            f.write(str(i) + '-' + str(j) + ': ' + str(losses) + '\n')  
    # cross-domain
    f.write('\nDomain A-B:\n')
    for i in range(len(input_A)):
        for j in range(len(input_B)):
            feats_1 = vgg(vgg_preprocess(input_A[i]), layers=get_id(opts.vgg_layer_src))
            feats_2 = vgg(vgg_preprocess(input_B[j]), layers=get_id(opts.vgg_layer_src))
            losses = []
            for _, (feat_1, feat_2) in enumerate(zip(feats_1, feats_2)):
                loss = CX_loss(feat_1, feat_2, sigma=opts.band_width_src)
                losses.append(loss.cpu())
            f.write('A' + str(i) + '-' + 'B' + str(j) + ': ' + str(losses) + '\n')
