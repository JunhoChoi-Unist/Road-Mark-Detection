import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np


class RoadDataset(Dataset):
    def __init__(self, dir_list, transform=None, small_seg=True):
        self.data = dir_list
        self.transform = transform
        self.small_seg = small_seg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import scipy.io

        data = scipy.io.loadmat(self.data[idx])
        data = data["rgb_seg_vp"]

        rgb = data[:, :, 0:3]
        seg = torch.Tensor(data[:, :, 3])
        vp = torch.Tensor(data[:, :, 4])

        rgb = T.ToTensor()(rgb)

        import random  
        x = random.random()
        if x > 0.5:
            flipper = T.RandomHorizontalFlip(p=1)
            rgb = flipper(rgb)
            seg = flipper(seg)
            vp = flipper(vp)


        # grid bbox-4x120x160
        grid_seg = (
            seg.view(120, 4, 640).transpose(1, 2).view(120, 160, 4, 4).transpose(2, 3)
        )
        small_seg = []
        for row in grid_seg:
            small_seg_row = []
            for cell in row:
                if cell.sum() != 0:
                    values = cell.flatten()
                    values = values[values != 0]
                    mode_value = torch.mode(values).values.item()
                    small_seg_row.append(mode_value)
                else:
                    small_seg_row.append(0)
            small_seg.append(small_seg_row)

        if self.small_seg:
            seg = torch.Tensor(small_seg)
        seg = seg.long()

        if vp.max() == 1:
            vp_x = vp.sum(axis=0).argmax(axis=0)
            vp_y = vp.sum(axis=1).argmax(axis=0)
            vpxy = torch.Tensor(
                (
                    vp_x,
                    vp_y,
                )
            )
        else:
            vpxy = torch.Tensor(
                (
                    0,
                    0,
                )
            )
        vpxy = vpxy.long()

        return rgb, seg, vpxy


if __name__ == "__main__":
    from utils import train_test_split
    import torchvision.transforms as T

    _, _val, _ = train_test_split(root_dir="D:/VPGNet-DB-5ch/", val_size=0.15)
    _dsds = RoadDataset(_val, T.Compose([T.ToTensor()]))
    _rgb, _gridbox, _seg, _vp = _dsds[0]
    print(f"_rgb: {_rgb.shape} {_rgb.dtype}")
    print(f"_gridbox: {_gridbox.shape} {_gridbox.dtype}")
    print(f"_seg: {_seg.shape} {_seg.dtype}")
    print(f"_vp: {_vp} {_vp.dtype}")
