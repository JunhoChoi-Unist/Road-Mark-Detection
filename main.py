def trainLoop(model, dataloader, criterion, optimizers, weights, wandb=None):
    losses = {'l_reg':0, 'l_om':0, 'l_ml':0, 'l_vp':0, 'loss':0}
    w1, w2, w3, w4 = weights

    for i, (rgb, gridbox, seg, vpxy) in tqdm(enumerate(dataloader), total=len(dataloader)):
        rgb = rgb.to(DEVICE)
        out = model(rgb)
        l_reg, l_om, l_ml, l_vp = criterion(out, gridbox, seg, vpxy)
        losses['l_reg'] += l_reg.item() * rgb.shape[0]
        losses['l_om'] += l_om.item() * rgb.shape[0]
        losses['l_ml'] += l_ml.item() * rgb.shape[0]
        losses['l_vp'] += l_vp.item() * rgb.shape[0]

        # update loss weights every batch when >10.0
        if max(w1 * l_reg, w2 * l_om, w3 * l_ml, w4 * l_vp) / max(w1 * l_reg, w2 * l_om, w3 * l_ml, w4 * l_vp) > 10.0:
            l_sum = torch.sum(1/l_reg, 1/l_om, 1/l_ml, 1/l_vp)
            w1 = (1/l_reg)/l_sum
            w2 = (1/l_om)/l_sum
            w3 = (1/l_ml)/l_sum
            w4 = (1/l_vp)/l_sum
        if wandb:
            wandb.log({
                'w1':w1,
                'w2':w2,
                'w3':w3,
                'w4':w4,
            })

        # weighted loss sum
        loss = w1 * l_reg + w2 * l_om + w3 * l_ml + w4 * l_vp
        losses['loss'] += loss.item() * rgb.shape[0]

        # update parameters
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        if i%100==0:
            print(f"\t @iter {i:<4}: l_reg={l_reg.item():>02.4f} l_om={l_om.item():>02.4f} l_ml={l_ml.item():>02.4f} l_vp={l_vp.item():>02.4f}")

    losses['l_reg'] /= len(dataloader.dataset)
    losses['l_om'] /= len(dataloader.dataset)
    losses['l_ml'] /= len(dataloader.dataset)
    losses['l_vp'] /= len(dataloader.dataset)
    losses['loss'] /= len(dataloader.dataset)

    if wandb:
        wandb.log(
            {
                "train loss": losses['loss'],
                "train l_reg": losses['l_reg'],
                "train l_om": losses['l_om'],
                "train l_ml": losses['l_ml'],
                "train l_vp": losses['l_vp'],
            },
            step=epoch,
        )
    print(f"\t train loss: {losses['loss']:.4f} l_reg:{losses['l_reg']:.4f} l_om:{losses['l_om']:.4f} l_ml:{losses['l_ml']:.4f} l_vp:{losses['l_vp']:.4f}")
    return (w1, w2, w3, w4)

def evalLoop(model, dataloader, criterion, weights, wandb=None):
    losses = {'l_reg':0, 'l_om':0, 'l_ml':0, 'l_vp':0, 'loss':0}
    w1, w2, w3, w4 = weights

    for rgb, gridbox, seg, vpxy in dataloader:
        rgb = rgb.to(DEVICE)
        out = model(rgb)
        l_reg, l_om, l_ml, l_vp = criterion(out, gridbox, seg, vpxy)
        losses['l_reg'] += l_reg.item() * rgb.shape[0]
        losses['l_om'] += l_om.item() * rgb.shape[0]
        losses['l_ml'] += l_ml.item() * rgb.shape[0]
        losses['l_vp'] += l_vp.item() * rgb.shape[0]

        # weighted loss sum
        loss = w1 * l_reg + w2 * l_om + w3 * l_ml + w4 * l_vp
        losses['loss'] += loss.item() * rgb.shape[0]

    losses['l_reg'] /= len(dataloader.dataset)
    losses['l_om'] /= len(dataloader.dataset)
    losses['l_ml'] /= len(dataloader.dataset)
    losses['l_vp'] /= len(dataloader.dataset)
    losses['loss'] /= len(dataloader.dataset)

    if wandb:
        wandb.log(
            {
                "val loss": losses['loss'],
                "val l_reg": losses['l_reg'],
                "val l_om": losses['l_om'],
                "val l_ml": losses['l_ml'],
                "val l_vp": losses['l_vp'],
            },
            step=epoch,
        )
    print(f"\t   val loss: {losses['loss']:.4f} l_reg:{losses['l_reg']:.4f} l_om:{losses['l_om']:.4f} l_ml:{losses['l_ml']:.4f} l_vp:{losses['l_vp']:.4f}")
    return losses['l_reg'], losses['l_om'], losses['l_ml'], losses['l_vp']

if __name__ == "__main__":
    from utils import train_test_split
    train, val, test = train_test_split(root_dir="D:/VPGNet-DB-5ch/", val_size=0.15)

    from RoadDataset import RoadDataset
    from models import VPGNet
    from losses import FourTaskLoss

    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    import os
    from datetime import datetime

    train_ds = RoadDataset(train, transform=T.Compose([T.ToTensor()]))
    val_ds = RoadDataset(val, transform=T.Compose([T.ToTensor()]))
    # test_ds = RoadDataset(test, transform=T.Compose([T.ToTensor()]))

    BATCH_SIZE = 14
    # train_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    import numpy as np
    from tqdm import tqdm
    import torch

    NOTES = "training from phase 1 with dynamic weight change"
    DEVICE = "cuda"
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    SAVE_PATH = f'exps/{datetime.now().strftime("%m%d-%H%M%S")}'
    WANDB = True
    N_CLASSES = 17
    PHASE = 1

    model = VPGNet(N_CLASSES).to(DEVICE)
    if PHASE==2:
        model.load_state_dict(torch.load('exps/0320-141111/06-0.4857.pt'))
    criterion = FourTaskLoss()
    # TODO: temporary fix
    if PHASE==1:
        optimizer_0 = torch.optim.Adam(model.shared.parameters(), lr=LEARNING_RATE)
        optimizer_1 = torch.optim.Adam(model.gridBox.parameters(), lr=0)
        optimizer_2 = torch.optim.Adam(model.objectMask.parameters(), lr=0)
        optimizer_3 = torch.optim.Adam(model.multiLabel.parameters(), lr=0)
        optimizer_4 = torch.optim.Adam(model.vpp.parameters(), lr=LEARNING_RATE)
    elif PHASE==2:
        optimizer_0 = torch.optim.Adam(model.shared.parameters(), lr=LEARNING_RATE)
        optimizer_1 = torch.optim.Adam(model.gridBox.parameters(), lr=LEARNING_RATE)
        optimizer_2 = torch.optim.Adam(model.objectMask.parameters(), lr=LEARNING_RATE)
        optimizer_3 = torch.optim.Adam(model.multiLabel.parameters(), lr=LEARNING_RATE)
        optimizer_4 = torch.optim.Adam(model.vpp.parameters(), lr=LEARNING_RATE)
    
    optimizers = [optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4]
    schedulers = [
        torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)
        for optimizer in optimizers
    ]

    run=None
    if WANDB:
        import wandb
        run = wandb.init(
            project="VPGNet",
            config={
                "EPOCHS": EPOCHS,
                "LEARNING_RATE": LEARNING_RATE,
                "SAVE_PATH": SAVE_PATH,
                "BATCH_SIZE": BATCH_SIZE,
                "N_CLASSES": N_CLASSES,
            },
            notes=NOTES
        )

    best_l_vp = np.inf
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    # Start fitting
    print(f'PHASE #{PHASE}')
    for epoch in range(1, 1 + EPOCHS):
        print(f'\nEpoch:{epoch:>3}/{EPOCHS}')
        # Start of the Training Loop
        model.train()
        w1, w2, w3, w4 = trainLoop(model, train_dl, criterion, optimizers, 
              weights=(w1, w2, w3, w4), wandb=run)

        # Start of the Validation Loop
        model.eval()
        with torch.no_grad():
            l_reg, l_om, l_ml, l_vp = evalLoop(model, val_dl, criterion, 
                weights=(w1, w2, w3, w4), wandb=run)

        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(
            model.state_dict(), f"{SAVE_PATH}/{epoch:02}-{(l_reg+l_om+l_ml+l_vp):02.4f}.pt"
        )
        print(f"\t model saved as {SAVE_PATH}/{epoch:02}-{(l_reg+l_om+l_ml+l_vp):02.4f}.pt")

        if WANDB:
            run.log(
                {
                    "shared_lr": optimizer_0.param_groups[0]["lr"],
                    "gridBox_lr": optimizer_1.param_groups[0]["lr"],
                    "objectMask_lr": optimizer_2.param_groups[0]["lr"],
                    "multiLabel_lr": optimizer_3.param_groups[0]["lr"],
                    "vpp_lr": optimizer_4.param_groups[0]["lr"],
                    "w1": criterion.weights[0].item(),
                    "w2": criterion.weights[1].item(),
                    "w3": criterion.weights[2].item(),
                    "w4": criterion.weights[3].item(),
                },
                step=epoch,
                commit=True
            )

        if PHASE == 1:
            if best_l_vp > l_vp:
                best_l_vp = l_vp
                patience = 0
            else:
                patience+=1
                if patience > 2:
                    patience = 0
                    PHASE = 2
                    print('PHASE 2 ENTERED!')
                    optimizers = [torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)]
                    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer=optimizers[0], step_size=5, gamma=0.7)]

        # reduce the learning rates...?
        for scheduler in schedulers:
            scheduler.step()
