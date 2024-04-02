def trainLoop(model, dataloader, criterion, optimizers, weights, epoch, wandb=None):
    losses = {"l_reg": 0, "l_om": 0, "l_ml": 0, "l_vp": 0, "loss": 0}
    w1, w2, w3, w4 = weights

    for i, (rgb, gridbox, seg, vpxy) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        rgb = rgb.to(DEVICE)
        out = model(rgb)
        l_reg, l_om, l_ml, l_vp = criterion(out, gridbox, seg, vpxy)
        losses["l_reg"] += l_reg.item() * rgb.shape[0]
        losses["l_om"] += l_om.item() * rgb.shape[0]
        losses["l_ml"] += l_ml.item() * rgb.shape[0]
        losses["l_vp"] += l_vp.item() * rgb.shape[0]

        # update loss weights every batch when > 5.0
        if (
            max(w1 * l_reg, w2 * l_om, w3 * l_ml, w4 * l_vp)
            / min(w1 * l_reg, w2 * l_om, w3 * l_ml, w4 * l_vp)
            > 5.0
        ):
            l_sum = (
                1 / l_reg.item() + 1 / l_om.item() + 1 / l_ml.item() + 1 / l_vp.item()
            )
            w1 = (1 / l_reg.item()) / l_sum
            w2 = (1 / l_om.item()) / l_sum
            w3 = (1 / l_ml.item()) / l_sum
            w4 = (1 / l_vp.item()) / l_sum
        if wandb:
            wandb.log(
                {
                    "w1": w1,
                    "w2": w2,
                    "w3": w3,
                    "w4": w4,
                    "iter_step": (epoch - 1) * len(dataloader) + i,
                }
            )

        # weighted loss sum
        loss = w1 * l_reg + w2 * l_om + w3 * l_ml + w4 * l_vp
        losses["loss"] += loss.item() * rgb.shape[0]

        # update parameters
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        if i % 100 == 0:
            print(
                f"\t @iter {i:<4}: l_reg={l_reg.item():>02.4f} l_om={l_om.item():>02.4f} l_ml={l_ml.item():>02.4f} l_vp={l_vp.item():>02.4f}"
            )

        if wandb:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/l_reg": l_reg.item(),
                    "train/l_om": l_om.item(),
                    "train/l_ml": l_ml.item(),
                    "train/l_vp": l_vp.item(),
                    "iter_step": (epoch - 1) * len(dataloader) + i,
                },
            )

    losses["l_reg"] /= len(dataloader.dataset)
    losses["l_om"] /= len(dataloader.dataset)
    losses["l_ml"] /= len(dataloader.dataset)
    losses["l_vp"] /= len(dataloader.dataset)
    losses["loss"] /= len(dataloader.dataset)
    print(
        f"\t train loss: {losses['loss']:.4f} l_reg:{losses['l_reg']:.4f} l_om:{losses['l_om']:.4f} l_ml:{losses['l_ml']:.4f} l_vp:{losses['l_vp']:.4f}"
    )
    return (w1, w2, w3, w4)


def evalLoop(model, dataloader, criterion, weights, epoch, wandb=None):
    losses = {"l_reg": 0, "l_om": 0, "l_ml": 0, "l_vp": 0, "loss": 0}
    w1, w2, w3, w4 = weights

    for rgb, gridbox, seg, vpxy in tqdm(dataloader):
        rgb = rgb.to(DEVICE)
        out = model(rgb)
        l_reg, l_om, l_ml, l_vp = criterion(out, gridbox, seg, vpxy)
        losses["l_reg"] += l_reg.item() * rgb.shape[0]
        losses["l_om"] += l_om.item() * rgb.shape[0]
        losses["l_ml"] += l_ml.item() * rgb.shape[0]
        losses["l_vp"] += l_vp.item() * rgb.shape[0]

        # weighted loss sum
        loss = w1 * l_reg + w2 * l_om + w3 * l_ml + w4 * l_vp
        losses["loss"] += loss.item() * rgb.shape[0]


    losses["l_reg"] /= len(dataloader.dataset)
    losses["l_om"] /= len(dataloader.dataset)
    losses["l_ml"] /= len(dataloader.dataset)
    losses["l_vp"] /= len(dataloader.dataset)
    losses["loss"] /= len(dataloader.dataset)

    if wandb:
        wandb.log(
            {
                "val/loss": losses["loss"],
                "val/l_reg": losses["l_reg"],
                "val/l_om": losses["l_om"],
                "val/l_ml": losses["l_ml"],
                "val/l_vp": losses["l_vp"],
                "epoch": epoch
            },
        )
    print(
        f"\t   val loss: {losses['loss']:.4f} l_reg:{losses['l_reg']:.4f} l_om:{losses['l_om']:.4f} l_ml:{losses['l_ml']:.4f} l_vp:{losses['l_vp']:.4f}"
    )
    return losses["l_reg"], losses["l_om"], losses["l_ml"], losses["l_vp"]


if __name__ == "__main__":
    from utils import train_test_split

    train, val = train_test_split(root_dir="data/VPGNet-DB-5ch/", test_size=0.1)

    from RoadDataset import RoadDataset
    from models import VPGNet
    from losses import FourTaskLoss

    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    import os
    from datetime import datetime

    train_ds = RoadDataset(train)
    # train_ds = RoadDataset(val, transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]))
    val_ds = RoadDataset(val)

    BATCH_SIZE = 20
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    import numpy as np
    from tqdm import tqdm
    import torch

    NOTES = "..."
    DEVICE = "cuda:0"
    EPOCHS = 50
    LEARNING_RATE = 1e-2
    SAVE_PATH = f'exps/{datetime.now().strftime("%m%d-%H%M%S")}'
    WANDB = True
    # N_CLASSES = 17
    N_CLASSES = 63
    PHASE = 1

    model = VPGNet(N_CLASSES).to(DEVICE)
    if PHASE == 2:
        model.load_state_dict(torch.load("exps/0331-160619/18-5.5598.pt", map_location=DEVICE))
    criterion = FourTaskLoss()
    # TODO: temporary fix
    if PHASE == 1:
        optimizer_0 = torch.optim.SGD(model.shared.parameters(), lr=LEARNING_RATE, momentum=0.9)
        optimizer_1 = torch.optim.SGD(model.gridBox.parameters(), lr=0, momentum=0.9)
        optimizer_2 = torch.optim.SGD(model.objectMask.parameters(), lr=0, momentum=0.9)
        optimizer_3 = torch.optim.SGD(model.multiLabel.parameters(), lr=0, momentum=0.9)
        optimizer_4 = torch.optim.SGD(model.vpp.parameters(), lr=LEARNING_RATE, momentum=0.9)
        w1, w2, w3, w4 = 0, 0, 0, 1.0
    elif PHASE == 2:
        optimizer_0 = torch.optim.SGD(model.shared.parameters(), lr=LEARNING_RATE, momentum=0.9)
        optimizer_1 = torch.optim.SGD(model.gridBox.parameters(), lr=LEARNING_RATE, momentum=0.9)
        optimizer_2 = torch.optim.SGD(model.objectMask.parameters(), lr=LEARNING_RATE, momentum=0.9)
        optimizer_3 = torch.optim.SGD(model.multiLabel.parameters(), lr=LEARNING_RATE, momentum=0.9)
        optimizer_4 = torch.optim.SGD(model.vpp.parameters(), lr=LEARNING_RATE, momentum=0.9)
        w1, w2, w3, w4 = 0.4683, 0.3328, 0.05551, 0.1434

    optimizers = [optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4]
    schedulers = [
        torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)
        for optimizer in optimizers
    ]

    run = None
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
            notes=NOTES,
        )
        wandb.define_metric("iter_step")
        wandb.define_metric("w[0-9]", step_metric="iter_step")
        wandb.define_metric("train/*", step_metric="iter_step")
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")

    best_l_vp = np.inf
    # Start fitting
    print(f"PHASE #{PHASE}")
    for epoch in range(1, 1 + EPOCHS):
        print(f"\nEpoch:{epoch:>3}/{EPOCHS}")

        # Start of the Training Loop
        model.train()
        w1, w2, w3, w4 = trainLoop(
            model,
            train_dl,
            criterion,
            optimizers,
            weights=(w1, w2, w3, w4),
            epoch=epoch,
            wandb=run,
        )

        # Start of the Validation Loop
        model.eval()
        with torch.no_grad():
            l_reg, l_om, l_ml, l_vp = evalLoop(
                model,
                val_dl,
                criterion,
                weights=(w1, w2, w3, w4),
                epoch=epoch,
                wandb=run,
            )

        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(
            model.state_dict(),
            f"{SAVE_PATH}/{epoch:02}-{(l_reg+l_om+l_ml+l_vp):02.4f}.pt",
        )
        print(
            f"\t model saved as {SAVE_PATH}/{epoch:02}-{(l_reg+l_om+l_ml+l_vp):02.4f}.pt"
        )

        if WANDB:
            run.log(
                {
                    "lr/shared": optimizer_0.param_groups[0]["lr"],
                    "lr/gridBox": optimizer_1.param_groups[0]["lr"],
                    "lr/objectMask": optimizer_2.param_groups[0]["lr"],
                    "lr/multiLabel": optimizer_3.param_groups[0]["lr"],
                    "lr/vpp": optimizer_4.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                commit=True,
            )

        # reduce the learning rates...?
        for scheduler in schedulers:
            scheduler.step()

        if PHASE == 1:
            if best_l_vp > l_vp:
                best_l_vp = l_vp
                patience = 0
            else:
                patience += 1
                if patience > 1:
                    patience = 0
                    PHASE = 2
                    print("\nPHASE 2 ENTERED!")
                    del optimizers
                    del schedulers
                    optimizer_0 = torch.optim.SGD(model.shared.parameters(), lr=LEARNING_RATE, momentum=0.9)
                    optimizer_1 = torch.optim.SGD(model.gridBox.parameters(), lr=LEARNING_RATE, momentum=0.9)
                    optimizer_2 = torch.optim.SGD(model.objectMask.parameters(), lr=LEARNING_RATE, momentum=0.9)
                    optimizer_3 = torch.optim.SGD(model.multiLabel.parameters(), lr=LEARNING_RATE, momentum=0.9)
                    optimizer_4 = torch.optim.SGD(model.vpp.parameters(), lr=LEARNING_RATE, momentum=0.9)
                    optimizers = [optimizer_0, optimizer_1, optimizer_2, optimizer_3, optimizer_4]
                    schedulers = [
                        torch.optim.lr_scheduler.StepLR(
                            optimizer=optimizer, step_size=5, gamma=0.7
                        ) for optimizer in optimizers
                    ]

