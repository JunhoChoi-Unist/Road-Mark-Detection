from utils import train_test_split

train, val, test = train_test_split(root_dir="D:/VPGNet-DB-5ch/", val_size=0.15)

if __name__ == "__main__":
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

    NOTES = "phase 2 training"
    DEVICE = "cuda"
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    SAVE_PATH = f'exps/{datetime.now().strftime("%m%d-%H%M%S")}'
    WANDB = True
    N_CLASSES = 17
    PHASE = 2

    model = VPGNet(N_CLASSES).to(DEVICE)
    if PHASE==2:
        model.load_state_dict(torch.load('exps/0320-141111/06-0.4857.pt'))
    criterion = FourTaskLoss(weights=[1/0.5, 1/0.2, 1/0.4, 1/0.4])
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

    phase = 1
    best_loss = np.inf
    patience = 0

    for epoch in range(1, 1 + EPOCHS):
        # Start of the Train Loop
        train_loss_epoch = 0.0
        (
            train_loss_reg_epoch,
            train_loss_om_epoch,
            train_loss_ml_epoch,
            train_loss_vp_epoch,
        ) = (0.0, 0.0, 0.0, 0.0)
        model.train()
        for i, (rgb, gridbox, seg, vpxy) in tqdm(
            enumerate(train_dl), total=len(train_dl)
        ):
            rgb = rgb.to(DEVICE)
            out = model(rgb)

            # print(f"out[0]: {out[0].shape} {out[0].dtype} {out[0].device}")
            # print(f"out[1]: {out[1].shape} {out[1].dtype} {out[1].device}")
            # print(f"out[2]: {out[2].shape} {out[2].dtype} {out[2].device}")
            # print(f"out[3]: {out[3].shape} {out[3].dtype} {out[3].device}")

            train_loss_reg, train_loss_om, train_loss_ml, train_loss_vp = criterion(
                out, gridbox, seg, vpxy
            )
            train_loss = train_loss_reg + train_loss_om + train_loss_ml + train_loss_vp
            # TODO: Apply weighted sum 1

            for optimizer in optimizers:
                optimizer.zero_grad()
            train_loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            train_loss_epoch += train_loss.item() * rgb.shape[0]
            train_loss_reg_epoch += train_loss_reg.item() * rgb.shape[0]
            train_loss_om_epoch += train_loss_om.item() * rgb.shape[0]
            train_loss_ml_epoch += train_loss_ml.item() * rgb.shape[0]
            train_loss_vp_epoch += train_loss_vp.item() * rgb.shape[0]

            if i%100==0:
                print(f"@iter {i:<4}: l_reg={train_loss_reg_epoch:.4f} l_om={train_loss_om_epoch:.4f} l_ml={train_loss_ml_epoch:.4f} l_vp={train_loss_vp_epoch:.4f}")

        train_loss_epoch /= len(train_dl.dataset)
        train_loss_reg_epoch /= len(train_dl.dataset)
        train_loss_om_epoch /= len(train_dl.dataset)
        train_loss_ml_epoch /= len(train_dl.dataset)
        train_loss_vp_epoch /= len(train_dl.dataset)

        if WANDB:
            run.log(
                {
                    "train loss": train_loss_epoch,
                    "train l_reg": train_loss_reg_epoch,
                    "train l_om": train_loss_om_epoch,
                    "train l_ml": train_loss_ml_epoch,
                    "train l_vp": train_loss_vp_epoch,
                },
                step=epoch,
            )
        # End of the Training Loop

        # Start of the Validation Loop
        val_loss_epoch = 0.0
        val_loss_reg_epoch, val_loss_om_epoch, val_loss_ml_epoch, val_loss_vp_epoch = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        model.eval()
        with torch.no_grad():
            for i, (rgb, gridbox, seg, vpxy) in tqdm(enumerate(val_dl), total=len(val_dl)):
                rgb = rgb.to(DEVICE)
                out = model(rgb)

                val_loss_reg, val_loss_om, val_loss_ml, val_loss_vp = criterion(
                    out, gridbox, seg, vpxy
                )
                val_loss = val_loss_reg + val_loss_om + val_loss_ml + val_loss_vp
                # TODO: Apply weighted sum 2

                val_loss_epoch += val_loss.item() * rgb.shape[0]
                val_loss_reg_epoch += val_loss_reg.item() * rgb.shape[0]
                val_loss_om_epoch += val_loss_om.item() * rgb.shape[0]
                val_loss_ml_epoch += val_loss_ml.item() * rgb.shape[0]
                val_loss_vp_epoch += val_loss_vp.item() * rgb.shape[0]

            val_loss_epoch /= len(val_dl.dataset)
            val_loss_reg_epoch /= len(val_dl.dataset)
            val_loss_om_epoch /= len(val_dl.dataset)
            val_loss_ml_epoch /= len(val_dl.dataset)
            val_loss_vp_epoch /= len(val_dl.dataset)

            if WANDB:
                run.log(
                    {
                        "val loss": val_loss_epoch,
                        "val l_reg": val_loss_reg_epoch,
                        "val l_om": val_loss_om_epoch,
                        "val l_ml": val_loss_ml_epoch,
                        "val l_vp": val_loss_vp_epoch,
                    },
                    step=epoch,
                )
        # End of the Validation Loop

        print(
            f"Epoch: {epoch:02}/{EPOCHS}\n\t train_loss: {train_loss_epoch:.4f}\n\t val_loss: {val_loss_epoch:.4f}"
        )
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(
            model.state_dict(), f"{SAVE_PATH}/{epoch:02}-{val_loss_epoch:.4f}.pt"
        )
        print(f"\t model saved as {SAVE_PATH}/{epoch:02}-{val_loss_epoch:.4f}.pt")

        if WANDB:
            run.log(
                {
                    "shared_lr": optimizer_0.param_groups[0]["lr"],
                    "gridBox_lr": optimizer_1.param_groups[0]["lr"],
                    "objectMask_lr": optimizer_2.param_groups[0]["lr"],
                    "multiLabel_lr": optimizer_3.param_groups[0]["lr"],
                    "vpp_lr": optimizer_4.param_groups[0]["lr"],
                    # TODO: Log weights of the Losses
                },
                step=epoch,
                commit=True
            )

        if best_loss > val_loss_epoch:
            best_loss = val_loss_epoch
            patience = 0
        else:
            patience += 1
            if phase == 1:
                if patience > 3:
                    phase = 2
                    patience = 0
                    # TODO: Test if fixed or not
                    current_lr = optimizer_0.param_groups[0]["lr"]
                    optimizer_1.param_groups[0]["lr"] = current_lr
                    optimizer_2.param_groups[0]["lr"] = current_lr
                    optimizer_3.param_groups[0]["lr"] = current_lr

            elif phase == 2:
                if patience > 5:    
                    print(f"Early stopping!\n\t @ Epoch: {epoch:2}\n\t Best Loss record: {best_loss:.4f}")
                    break # EARLY-STOPPING
                

        # reduce the learning rates...?
        for scheduler in schedulers:
            scheduler.step()
