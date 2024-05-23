import os
from datetime import datetime
from RoadDataset import VPGDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from Models import VPPNet
from train import loop, eval
from tqdm import tqdm


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", type=float, default=5e-1)
    parser.add_argument("--max_lr", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--phase", type=int, choices=[1, 2], help="1:VPP 2.ALL")
    parser.add_argument("--save_path", type=str, default='D:/checkpoints/Inception_v3/')
    parser.add_argument("--load_checkpoint", type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--timestamp', type=str)
    args = parser.parse_args()
    # args.verbose=True
    print(args)

    train_dataset = VPGDataset("./Data/Train", phase=args.phase)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = VPGDataset("./Data/Val", phase=args.phase)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = 'cuda'
    model = VPPNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_loss = np.inf
    checkpoint_dir = os.path.join(args.save_path, f"run_{args.timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir,'best.pt')

    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epoch):

        print(f"[{1+epoch:02} / {args.epoch}]")

        model.train()
        train_loss = []
        for rgb, vp in tqdm(train_dataloader):
            vp_out = model(rgb.to(device))
            loss = criterion(vp_out, vp.to(device))
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = np.mean(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_loss = []
            for rgb, vp in tqdm(val_dataloader):
                vp_out = model(rgb.to(device))
                loss = criterion(vp_out, vp.to(device))
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

    
        # train_loss = loop(model, optimizer, scheduler, train_dataloader, device, args)
        # val_loss = eval(model, val_dataloader, device, args)
        print(f"Train Loss: {train_loss:02.4f} Val Loss: {val_loss:02.4f}")
        scheduler.step(val_loss)
        print(f"learning rate: {scheduler.get_last_lr()[0]:.0e}")
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print('\033[1;32m' + f'model weight saved as {best_checkpoint_path}' + '\033[0m')