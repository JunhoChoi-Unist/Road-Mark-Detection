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

    BATCH_SIZE = 22
    # train_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    import numpy as np
    from tqdm import tqdm
    import torch

    DEVICE = "cuda"
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    SAVE_PATH = f'exps/{datetime.now().strftime("%m%d-%H%M%S")}'
    WANDB = True
    N_CLASSES = 17
    NOTES="Training only vp"

    model = VPGNet(N_CLASSES).to(DEVICE)
    criterion = FourTaskLoss()
    # TODO: temporary fix
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)

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

    best_loss = np.inf
    patience = 0

    for epoch in range(1, 1 + EPOCHS):
        # Start of the Train Loop
        train_loss_vp_epoch = 0.0
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

            train_loss_vp = criterion(
                out, gridbox, seg, vpxy
            )

            optimizer.zero_grad()
            train_loss_vp.backward()
            optimizer.step()

            train_loss_vp_epoch += train_loss_vp.item() * rgb.shape[0]

        train_loss_vp_epoch /= len(train_dl.dataset)

        if WANDB:
            run.log(
                {
                    "train l_vp": train_loss_vp_epoch,
                },
                step=epoch,
            )
        # End of the Training Loop

        # Start of the Validation Loop
        val_loss_vp_epoch = 0.0
        model.eval()
        with torch.no_grad():
            for i, (rgb, gridbox, seg, vpxy) in tqdm(enumerate(val_dl), total=len(val_dl)):
                rgb = rgb.to(DEVICE)
                out = model(rgb)

                val_loss_vp = criterion(
                    out, gridbox, seg, vpxy
                )

                val_loss_vp_epoch += val_loss_vp.item() * rgb.shape[0]

            val_loss_vp_epoch /= len(val_dl.dataset)

            if WANDB:
                run.log(
                    {
                        "val l_vp": val_loss_vp_epoch,
                    },
                    step=epoch,
                )
        # End of the Validation Loop

        print(
            f"Epoch: {epoch:02}/{EPOCHS}\n\t train_loss: {train_loss_vp_epoch:.4f}\n\t val_loss: {val_loss_vp_epoch:.4f}"
        )
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(
            model.state_dict(), f"{SAVE_PATH}/{epoch:02}-{val_loss_vp_epoch:.4f}.pt"
        )
        print(f"\t model saved as {SAVE_PATH}/{epoch:02}-{val_loss_vp_epoch:.4f}.pt")

        if WANDB:
            run.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        if best_loss > val_loss_vp_epoch:
            best_loss = val_loss_vp_epoch
            patience = 0
        else:
            patience += 1
            if patience > 5:    
                print(f"Early stopping!\n\t @ Epoch: {epoch:2}\n\t Best Loss record: {best_loss:.4f}")
                break # EARLY-STOPPING
                

        # reduce the learning rates...?
        scheduler.step()
