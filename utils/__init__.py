def train_test_split(root_dir="D:/VPGNet-DB-5ch/", val_size=0.15):
    from pathlib import Path
    from glob import glob

    all_data = glob(str(Path(root_dir) / "scene_*/*/*.mat"))

    # import random
    # random.seed(42)
    # random.shuffle(all_data)

    total_len = len(all_data)
    val_len = int(val_size * total_len)
    train_len = 5 * val_len

    train = all_data[:train_len]
    val = all_data[train_len : train_len + val_len]
    test = all_data[train_len + val_len :]

    return train, val, test


def draw_vpp(model, dataset, num=10):
    import torch
    import torch.nn.functional as F

    rgbs = [dataset[i][0] for i in range(num)]
    quads = [dataset[i][2] for i in range(num)]
    vpps = [model(rgb.unsqueeze(0))[3] for rgb in rgbs]  # (1 x 5 x 120 x 160)
    quad_hats = [torch.argmax(vpp, dim=1) for vpp in vpps]  # (1 x 120 x 160)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(num, 2, figsize=(6 * 2, 4 * num))
    for n in range(num):
        axs[n, 0].imshow(rgbs[n].permute(1, 2, 0))
        axs[n, 0].imshow(
            quads[n].repeat_interleave(4, 1).repeat_interleave(4, 0), alpha=0.5
        )
        axs[n, 0].axis("off")
        axs[n, 1].imshow(rgbs[n].permute(1, 2, 0))
        axs[n, 1].imshow(
            quad_hats[n].repeat_interleave(4, 1).repeat_interleave(4, 2).squeeze(0),
            alpha=0.5,
        )
        axs[n, 1].axis("off")
    plt.show()


def draw_segs(model, dataset, num=10):
    rgbs = [dataset[i][0] for i in range(num)]
    segs = [dataset[i][1] for i in range(num)]
    segps = [model(rgb.unsqueeze(0))[2] for rgb in rgbs]  # (1 x 5 x 120 x 160)
    seg_hats = [torch.argmax(segp, dim=1) for segp in segps]  # (1 x 120 x 160)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(num, 2, figsize=(6 * 2, 4 * num))
    for n in range(num):
        axs[n, 0].imshow(rgbs[n].permute(1, 2, 0))
        axs[n, 0].imshow(segs[n], alpha=0.7)
        axs[n, 0].axis("off")
        axs[n, 1].imshow(rgbs[n].permute(1, 2, 0))
        axs[n, 1].imshow(segs[n], alpha=0.6)
        axs[n, 1].axis("off")
    plt.show()


if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    import sys

    sys.path.insert(0, "..")
    from models import VPGNet
    from RoadDataset import RoadDataset

    _, test, _ = train_test_split(root_dir="D:/VPGNet-DB-5ch/", val_size=0.15)
    test_ds = RoadDataset(test, transform=T.Compose([T.ToTensor()]))

    # draw_segs(test_ds, num=10)

    # model = VPGNet()
    # model.load_state_dict(torch.load('../exps/1.pt'))
    # model.load_state_dict(torch.load('../exps/11.pt'))
    # model.eval()
    # with torch.no_grad():
    #     draw_vpp(model, test_ds, num=15)
