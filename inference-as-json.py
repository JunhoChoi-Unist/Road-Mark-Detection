import os
import sys
sys.path.append('../')
from Models import MaskRCNNWithInceptionV3
from RoadDataset import CeyMoDataset
import torch
from torch.utils.data import DataLoader
import numpy as np

import cv2
import json

from tqdm import tqdm

GT_DIR = 'D:/CeyMo/original/test/'
PRED_DIR = 'D:/CeyMo/original/test_pred'
os.makedirs(PRED_DIR, exist_ok=True)


'''cv2.findContours: binary mask -> polygon'''
def mask_to_polygon(mask, threshold=0.3):
    # Find contours in the mask
    contours = [cv2.findContours(
        (m>threshold)[0].detach().cpu().numpy().astype(np.uint8), 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0][0].squeeze() for m in mask]
    # Convert contours to polygon points
    polygons = [c * (1920/299, 1080/299) for c in contours if len(c)>3]
    return polygons

'''json format of CeyMo Dataset'''
def create_annotation(mask, label, filename):
    polygons = mask_to_polygon(mask)
    class_dict = {0:'Background', 1:'SA', 2:'LA', 3:'RA', 4:'SLA', 5:'SRA', 6:'DM', 7:'PC', 8:'JB', 9:'SL', 10:'BL', 11:'CL'}
    annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [{
            "label": class_dict[l.item()],
            "points": p.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
            } for l,p in zip(label, polygons)],
        "imagePath": f"{filename}.jpg",
        "imageData": None,
        "imageHeight": 1080,
        "imageWidth": 1920,
        "category": "normal",
        "vehicle": 1,
        "camera": 1
        }
    return annotation


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

label_dict = {
    0:'background',
    1:'BL',
    2:'CL',
    3:'DM',
    4:'JB',
    5:'LA',
    6:'PC',
    7:'RA',
    8:'SA',
    9:'SL',
    10:'SLA',
    11:'SRA'
}

'''Load CeyMo Test Dataset'''
dataset = CeyMoDataset(GT_DIR);dataloader = DataLoader(dataset, 4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

'''Load Mask R-CNN & pretrained weight'''
model = MaskRCNNWithInceptionV3(num_classes=20).to(device); model_state_dict = model.state_dict()
# # Mask R-CNN trained from VPGNet Dataset, VPP pretrained from VPGNet Datset
# vpp_trained_state_dict = torch.load('D:/checkpoints/mask_rcnn/best.pt');model_state_dict.update(vpp_trained_state_dict)  
# # Mask R-CNN trained from CeyMo Dataset, VPP pretrained from VPGNet Datset
vpp_trained_state_dict = torch.load('D:/checkpoints/CeyMo/run_0523_164045/best.pt');model_state_dict.update(vpp_trained_state_dict)  
# # VPP-only trained
# mask_rcnn_trained_state_dict = torch.load('D:/checkpoints/Inception_v3/run_0522_030212/best.pt')    
# mask_rcnn_trained_state_dict = {k:v for k,v in mask_rcnn_trained_state_dict.items() if k in model_state_dict.keys()}
model.load_state_dict(model_state_dict)
model.eval()

'''Inference'''
with torch.no_grad():
    for rgbs, targets, filenames in tqdm(dataloader):
        rgbs = tuple([r.to(device) for r in rgbs])
        out = model(rgbs)
        
        boxes = [o['boxes'] for o in out]
        labels = [o['labels'] for o in out]
        scores = [o['scores'] for o in out]
        masks = [o['masks'] for o in out]

        for rgb, box, label, score, mask, filename in zip(rgbs, boxes, labels, scores, masks, filenames):
            with open(os.path.join(PRED_DIR,f'{filename}.json'), 'w') as f:
                json.dump(create_annotation(mask, label, filename), f, indent=4)