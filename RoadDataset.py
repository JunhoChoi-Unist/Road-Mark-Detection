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
        vp = data[:, :, 4]

        if self.transform is not None:
            rgb = self.transform(rgb)


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

        gridbox = torch.tensor(0)
        seg = torch.tensor(0)
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
