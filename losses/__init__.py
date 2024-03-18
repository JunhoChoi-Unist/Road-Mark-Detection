from cProfile import label
import torch
import numpy as np

def FourTaskLoss(out, seg, vp, device='cuda'):
    CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    # Object Mask Loss
    object_mask = (seg>0).long().to(device)
    L_om = CrossEntropyLoss(out[1], object_mask)
    # Multi Label Loss
    label_mask = torch.nn.functional.max_pool2d(seg, 2)
    label_mask = label_mask.long().to(device)
    L_ml = CrossEntropyLoss(out[2], label_mask)
    # VPP Loss
    L_vp = CrossEntropyLoss(out[3], vp)

    return L_om, L_ml, L_vp

def fourPoints(tt):
    indices = tt>0
    
    # Get the minimum and maximum x and y coordinates
    min_x, _ = torch.min(indices, dim=0)
    max_x, _ = torch.max(indices, dim=0)
    
    # Return the left-top and right-bottom coordinates
    left_top = (min_x[1].item(), min_x[0].item())
    right_bottom = (max_x[1].item(), max_x[0].item())
    
    return *left_top, *right_bottom

