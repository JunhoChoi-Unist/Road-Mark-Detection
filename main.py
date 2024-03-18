from utils import train_test_split
train, val, test = train_test_split(root_dir='D:/VPGNet-DB-5ch/', val_size=0.15)

if __name__ == '__main__':
    from RoadDataset import RoadDataset
    from models import VPGNet
    from losses import FourTaskLoss

    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    import os
    from datetime import datetime

    # train_ds = RoadDataset(train, transform=T.Compose([T.ToTensor()]))
    val_ds = RoadDataset(val, transform=T.Compose([T.ToTensor()]))
    # test_ds = RoadDataset(test, transform=T.Compose([T.ToTensor()]))

    BATCH_SIZE = 14
    train_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    

    import numpy as np
    from tqdm import tqdm
    import torch
    DEVICE = 'cuda'
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    SAVE_PATH = f'exps/{datetime.now().strftime("%m%d-%H%M%S")}'
    WANDB = True


    model = VPGNet().to(DEVICE)
    criterion = FourTaskLoss
    optimizer_0 = torch.optim.Adam(model.shared.parameters(), lr=LEARNING_RATE)
    optimizer_1 = torch.optim.Adam(model.gridBox.parameters(), lr=0)
    optimizer_2 = torch.optim.Adam(model.objectMask.parameters(), lr=0)
    optimizer_3 = torch.optim.Adam(model.multiLabel.parameters(), lr=0)
    optimizer_4 = torch.optim.Adam(model.vpp.parameters(), lr=LEARNING_RATE)

    optimizers = [optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7) for optimizer in optimizers]

    if WANDB:
        import wandb
        run = wandb.init(project='VPGNet', config={
            'EPOCHS': EPOCHS,
            'LEARNING_RATE': LEARNING_RATE,
            "SAVE_PATH": SAVE_PATH,
            "BATCH_SIZE": BATCH_SIZE,
        })

    phase=1
    best_loss = np.inf
    patience = 0
    for epoch in range(1,1+EPOCHS):

        train_loss = 0.0
        L_regs_train, L_oms_train, L_mls_train, L_vps_train = 0, 0, 0, 0
        model.train()
        for i, (rgb, seg, vp) in tqdm(enumerate(train_dl), total=len(train_dl)):
            rgb = rgb.to(DEVICE)
            vp = vp.to(DEVICE)
            seg = seg.to(DEVICE)
            seg = seg.float()
            vp = vp.long()

            out = model(rgb)

            # L_reg, L_om, L_ml, L_vp = criterion(out, seg, vp)
            L_om_train, L_ml_train, L_vp_train = criterion(out, seg, vp)
            # if i==0 and epoch==1:
            #     # w1 = 1 / L_reg
            #     w2 = 1 / L_om_train
            #     w3 = 1 / L_ml_train
            #     w4 = 1 / L_vp_train
            # loss = w1*L_reg + w2*L_om + w3*L_ml + w4*L_vp
            loss = L_om_train + L_ml_train + L_vp_train
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            train_loss += loss.item() * vp.shape[0]
            # L_regs += L_reg.item() * vp.shape[0]
            L_oms_train += L_om_train.item() * vp.shape[0]
            L_mls_train += L_ml_train.item() * vp.shape[0]
            L_vps_train += L_vp_train.item() * vp.shape[0]
        train_loss /= len(train_dl.dataset)
        # L_regs /= len(train_dl.dataset)
        L_oms_train /= len(train_dl.dataset)
        L_mls_train /= len(train_dl.dataset)
        L_vps_train /= len(train_dl.dataset)
        for scheduler in schedulers:
            scheduler.step()



        val_loss = 0.0
        L_regs_val, L_oms_val, L_mls_val, L_vps_val = 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for rgb, seg, vp in tqdm(val_dl):
                rgb = rgb.to(DEVICE)
                vp = vp.to(DEVICE)
                seg = seg.to(DEVICE)
                seg = seg.float()
                vp = vp.long()

                out = model(rgb)

                L_om_val, L_ml_val, L_vp_val = criterion(out, seg, vp)
                loss = L_om_val + L_ml_val + L_vp_val
                # loss = w2*L_om_val + w3*L_ml_val + w4*L_vp_val
                val_loss += loss.item() * vp.shape[0]
                L_oms_val += L_om_val.item() * vp.shape[0]
                L_mls_val += L_ml_val.item() * vp.shape[0]
                L_vps_val += L_vp_val.item() * vp.shape[0]
            val_loss /= len(val_dl.dataset)
            L_oms_val /= len(val_dl.dataset)
            L_mls_val /= len(val_dl.dataset)
            L_vps_val /= len(val_dl.dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                patience=0
            else:
                patience+=1
                if patience > 3 and phase == 1:
                    print("\nTurning into phase #2!\n")
                    phase=2
                    patience=0
                    lr = optimizer_0.param_groups[0]['lr']
                    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
                    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)]


        
        print(f'Epoch {epoch:2}/{EPOCHS}... | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} |\n')
        os.makedirs(SAVE_PATH, exist_ok = True)  
        torch.save(model.state_dict(), f'{SAVE_PATH}/{epoch:02}-{val_loss:.4f}.pt')
        

        if WANDB:
            run.log({
                "shared_lr": optimizer_0.param_groups[0]['lr'],
                "gridBox_lr": optimizer_1.param_groups[0]['lr'],
                "objectMask_lr": optimizer_2.param_groups[0]['lr'],
                "multiLabel_lr": optimizer_3.param_groups[0]['lr'],
                "vpp_lr": optimizer_4.param_groups[0]['lr'],
                "train_loss": train_loss,
                "train_om_loss": L_oms_train,
                "train_ml_loss": L_mls_train,
                "train_vp_loss": L_vps_train,
                "val_loss": val_loss,
                "val_om_loss": L_oms_val,
                "val_ml_loss": L_mls_val,
                "val_vp_loss": L_vps_val,
                # 'w2':w2,
                # 'w3':w3,
                # 'w4':w4,
            }, step=epoch)
        
        # if max(w2*L_oms_train, w3*L_mls_train, w4*L_vps_train)/min(w2*L_oms_train, w3*L_mls_train, w4*L_vps_train) > 2:
        #     w2 = 1/L_oms_train
        #     w3 = 1/L_mls_train
        #     w4 = 1/L_vps_train