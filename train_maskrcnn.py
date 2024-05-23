from RoadDataset import VPGDataset
from torch.utils.data import DataLoader

from Models import MaskRCNNWithInceptionV3
import torch
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MaskRCNNWithInceptionV3(num_classes=20).to(device)
model_state_dict = model.state_dict()
pretrained_weight = torch.load('D:/checkpoints/Inception_v3/run_0522_030212/best.pt')
inception_weight = {f'model.{k}':v for k,v in pretrained_weight.items() if f'model.{k}'in model_state_dict}
# print(model_state_dict.keys())
# print(inception_weight.keys())
model_state_dict.update(inception_weight)
model.load_state_dict(model_state_dict)

train_ds = VPGDataset('./Data/Train', phase=2)
train_dl = DataLoader(train_ds, 5, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_ds = VPGDataset('./Data/Val', phase=2)
val_dl = DataLoader(val_ds, 5, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
best_loss = np.inf
best_checkpoint_path = 'D:/checkpoints/mask_rcnn/best.pt'

for epoch in range(num_epochs):
    model.train()
    loss = []
    for rgb, target, _ in tqdm(train_dl):
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

    # model.eval()
    with torch.no_grad():
        loss = []
        for rgb, target, _ in tqdm(val_dl):
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

            loss.append(losses.item())
    
        val_loss = np.mean(loss)
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print('\033[1;32m' + f'model weight saved as {best_checkpoint_path}' + '\033[0m')

    print(epoch)
    print(f'train loss: {train_loss:.3f}')
    print(f'val loss: {val_loss:.3f}')
