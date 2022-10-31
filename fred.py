import cv2
#@title Setup
import argparse, subprocess, sys, time
from ast import For

# def setup():
#     install_cmds = [
#         ['pip', 'install', 'ftfy', 'regex', 'tqdm', 'transformers==4.21.2', 'timm', 'fairscale', 'requests'],
#         ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'],
#         ['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@main#egg=blip'],
#         ['git', 'clone', 'https://github.com/pharmapsychotic/clip-interrogator.git']
#     ]
#     for cmd in install_cmds:
#         print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

# setup()

import sys
sys.path.append('src/blip')
sys.path.append('src/clip')

import clip
import hashlib
import io
import IPython
import ipywidgets as widgets
import math
import numpy as np
import os
import pickle
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.blip import blip_decoder
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from zipfile import ZipFile


chunk_size = 2048
flavor_intermediate_count = 2048

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print("Loading BLIP model...")
blip_image_eval_size = 384
# blip_model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'        
# blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit='large', med_config='./src/blip/configs/med_config.json')
# blip_model.eval()
# blip_model = blip_model.to(device)

print("Loading CLIP model...")
clip_model_name = 'ViT-L/14' #@param ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64'] {type:'string'}
clip_model, clip_preprocess = clip.load(clip_model_name, device="cuda")
clip_model.cuda().eval()

def rank_top(image_features, text_array):
    text_tokens = clip.tokenize([text for text in text_array]).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = torch.zeros((1, len(text_array)), device=device)
    for i in range(image_features.shape[0]):
        similarity += (image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)

    _, top_labels = similarity.cpu().topk(1, dim=-1)
    return text_array[top_labels[0][0].numpy()]

def rank_top_from_tokens(image_features, text_tokens):
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = torch.zeros((1, len(text_array)), device=device)
    for i in range(image_features.shape[0]):
        similarity += (image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)

    # _, top_labels = similarity.cpu().topk(1, dim=-1)
    fred, top_labels = similarity.cpu().topk(1, dim=-1)
    return text_array[top_labels[0][0].numpy()]


def interrogate(image, text_tokens):
    caption = "Skipping blip"
    # caption = generate_caption(image)

    images = clip_preprocess(image).unsqueeze(0).cuda()
    with torch.no_grad():
        image_features = clip_model.encode_image(images).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

    res = rank_top_from_tokens(image_features, text_tokens)
    return res

text_array = [
    'bunny, flowers, the bunny is smelling the flowers',
    'bunny, flowers, the bunny is yawning',
    'bunny, flowers, the bunny is smelling the air',
    'bunny, flowers, the bunny looking at the flowers',
    'bunny, flowers',
    'bunny, no flowers',
    'no bunny, flowers',
    'no bunny, no flowers'
]

text_tokens = clip.tokenize([text for text in text_array]).cuda()

vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
success,image = vidcap.read()
count = 0
while success:
    from PIL import Image 
    im_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im_pil)
    res = interrogate(im_pil, text_tokens)
    print ('frame '+str(count)+': ' +res)
    if res == text_array[0]:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    count += 1

