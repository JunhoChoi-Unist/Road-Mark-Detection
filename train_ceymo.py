from RoadDataset import CeyMoDataset
from torch.utils.data import DataLoader

import os
from Models import MaskRCNNWithInceptionV3
import torch
import numpy as np
from tqdm import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    
    model = MaskRCNNWithInceptionV3(num_classes=20).to(device)
    model_state_dict = model.state_dict()
    pretrained_weight = torch.load('D:/checkpoints/Inception_v3/run_0522_030212/best.pt')
    inception_weight = {f'model.{k}':v for k,v in pretrained_weight.items() if f'model.{k}'in model_state_dict}
    # print(model_state_dict.keys())
    # print(inception_weight.keys())
    model_state_dict.update(inception_weight)
    model.load_state_dict(model_state_dict)

    train_ds = CeyMoDataset('./Data/CeyMo/train')
    train_dl = DataLoader(train_ds, 5, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    # val_ds = CeyMoDataset('./Data/Val', phase=2)
    # val_dl = DataLoader(val_ds, 5, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    num_epochs = 50
    best_loss = np.inf
    checkpoint_dir = os.path.join(args.save_path, f"run_{args.timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir,'best.pt')

    for epoch in range(args.epoch):
        print(f"[{1+epoch:02} / {args.epoch}]")

        model.train()
        #
        loss = []
        for rgb, target, filename in tqdm(train_dl):
            '''
            Bounding Boxes (boxes): The coordinates [x_min, y_min, x_max, y_max] 
                                    specify the top-left and bottom-right corners of the bounding box.
            Labels (labels): Each integer corresponds to the class of the object within the bounding box.
            Masks (masks): Each mask is a 2D array of the same height and width as the image, 
                            where pixels belonging to the object are 1 and others are 0.
            '''
            rgb = list(image.to(device) for image in rgb)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]

            loss_dict = model(rgb, target)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss.append(losses.item())
        
        train_loss = np.mean(loss)

        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_checkpoint_path)

        print(epoch)
        print(f'train loss: {train_loss:.3f}')
        scheduler.step(train_loss)
        print(f"learning rate: {scheduler.get_last_lr()[0]:.0e}")
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print('\033[1;32m' + f'model weight saved as {best_checkpoint_path}' + '\033[0m')
