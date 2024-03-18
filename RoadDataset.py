import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        data = data['rgb_seg_vp']

        rgb = data[:, :, 0:3]
        seg = torch.Tensor(data[:, :, 3])
        vp = data[:, :, 4]

        quad = torch.zeros((120, 160), dtype=torch.uint8)

        if vp.max() == 1:
            vp_x = vp.sum(axis=0).argmax(axis=0)
            vp_y = vp.sum(axis=1).argmax(axis=0)

            quad_x = round(vp_x/4)
            quad_y = round(vp_y/4)
            
            quad[:quad_y,quad_x:] = 1
            quad[:quad_y,:quad_x] = 2
            quad[quad_y:,:quad_x] = 3
            quad[quad_y:,quad_x:] = 4

        if self.transform is not None:
            rgb = self.transform(rgb)

        if self.small_seg:
            seg = F.max_pool2d(seg.unsqueeze(0), 4).squeeze(0)

        # return rgb, seg, quad, quad_x, quad_y
        # return rgb, seg, quad, vp_x, vp_y
        return rgb, seg, quad

if __name__=='__main__':
    from utils import train_test_split
    import torchvision.transforms as T
    _, _, _test = train_test_split(root_dir='D:/VPGNet-DB-5ch/', val_size=0.15)    
    _dsds = RoadDataset(_test, T.Compose([T.ToTensor()]))
    _rgb, _seg, _quad = _dsds[0]
    
    # # Test Plot 3rd quadrant
    # import matplotlib.pyplot as plt
    # _rgb, _seg, _quad, vp_x, vp_y = _dsds[0]
    # plt.figure()
    # plt.imshow(_rgb.permute(1,2,0))
    # plt.figure()
    # plt.imshow(_rgb[:,4*vp_y:,:4*vp_x].permute(1,2,0)) # 3rd
    # plt.show()

    # # Test Plot quadrant w/ color splash
    # import matplotlib.pyplot as plt
    # _rgb, _seg, _quad, _vp_x, _vp_y = _dsds[42]
    # plt.figure()
    # plt.imshow(_rgb.permute(1,2,0))
    # plt.imshow(_quad.repeat_interleave(4,1).repeat_interleave(4,0), alpha=0.5)
    # plt.scatter(_vp_x, _vp_y, marker='x', c='r')
    # plt.xticks(range(0,640,8))
    # plt.yticks(range(0,480,8))
    # plt.grid(True, alpha=0.5, linestyle='--')
    # plt.show()

    print()