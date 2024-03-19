import torch
import torch.nn.functional as F
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
        vp = data[:, :, 4]

        if self.transform is not None:
            rgb = self.transform(rgb)

        # grid bbox-4x120x160
        grid_seg = (
            seg.view(120, 4, 640).transpose(1, 2).view(120, 160, 4, 4).transpose(2, 3)
        )
        xs, ys, ws, hs = [], [], [], []
        small_seg = []
        for row in grid_seg:
            x_row, y_row, w_row, h_row = [], [], [], []
            small_seg_row = []
            for cell in row:
                if cell.sum() != 0:
                    values = cell.flatten()
                    values = values[values != 0]
                    mode_value = torch.mode(values).values.item()
                    cell = torch.zeros_like(cell)
                    cell[cell == mode_value] = 1
                    x = cell.argmax(axis=1).min()
                    y = cell.flatten().argmax() // 4
                    w = 3 - torch.flip(cell, dims=[1]).argmax(axis=1).min() - x
                    h = 3 - torch.flip(cell, dims=[0]).flatten().argmax() // 4 - y
                    x_row.append(x / 4.0)
                    y_row.append(y / 4.0)
                    w_row.append(w / 4.0)
                    h_row.append(h / 4.0)
                    small_seg_row.append(mode_value)
                else:
                    x_row.append(0)
                    y_row.append(0)
                    w_row.append(1)
                    h_row.append(1)
                    small_seg_row.append(0)
            xs.append(x_row)
            ys.append(y_row)
            ws.append(w_row)
            hs.append(h_row)
            small_seg.append(small_seg_row)
        gridbox = torch.Tensor([xs, ys, ws, hs])

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

        # TODO: return grid bbox as well
        return rgb, gridbox, seg, vpxy


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
