from scipy import optimize
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def loop(model, optimizer, scheduler, dataloader, device, args):
    running_loss = AverageMeter()
    for i, data in tqdm(
        enumerate(dataloader), total=len(dataloader), disable=args.verbose
    ):
        rgb, vp = data
        rgb = rgb.to(device)
        vp = vp.to(device)
        shared_out, vp_out = model(rgb)
        criterion = CrossEntropyLoss()
        loss = criterion(vp_out, vp)
        running_loss.update(loss.item(), n=len(data))
        if args.verbose:
            print(f"{running_loss.avg:02.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    # print(f"Train Loss: {running_loss.avg:02.4f}", end=' ')
    return running_loss.avg

def eval(model, dataloader, device, args):
    running_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), disable=args.verbose):
            rgb, vp = data
            rgb = rgb.to(device)
            vp = vp.to(device)
            shared_out, vp_out = model(rgb)
            criterion = CrossEntropyLoss()
            loss = criterion(vp_out, vp)
            running_loss.update(loss.item(), n=len(data))
            if args.verbose:
                print(f"{running_loss.avg:02.4f}")
          
        # print(f"  Val Loss: {running_loss.avg:02.4f}")  
    return running_loss.avg