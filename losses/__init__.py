import torch
import torch.nn as nn
import torch.nn.functional as F


class FourTaskLoss(nn.Module):
    def __init__(self):
        super(FourTaskLoss, self).__init__()

    def forward(self, out, seg, vpxy):
        om = (seg > 0).long()
        l_om = F.cross_entropy(out[1], om.to('cuda:0'))
        ml_seg = torch.nn.functional.max_pool2d(seg, 2).long()
        l_ml = F.cross_entropy(out[2], ml_seg.to('cuda:0'))
        vp = torch.zeros((out[3].shape[0], 120, 160)).long()
        for i, (vp_x, vp_y) in enumerate(vpxy):
            if vp_x != 0 or vp_y != 0:
                vp[i][: vp_y // 4, vp_x // 4 :] = 1
                vp[i][: vp_y // 4, : vp_x // 4] = 2
                vp[i][vp_y // 4 :, : vp_x // 4] = 3
                vp[i][vp_y // 4 :, vp_x // 4 :] = 4
        l_vp = F.cross_entropy(out[3], vp.to('cuda:0'))
        
        return l_om, l_ml, l_vp

class VppTaskLoss(nn.Module):
    def __init__(self):
        super(VppTaskLoss, self).__init__()

    def forward(self, out, vpxy):
        vp = torch.zeros((out[3].shape[0], 120, 160)).long()
        for i, (vp_x, vp_y) in enumerate(vpxy):
            if vp_x != 0 or vp_y != 0:
                vp[i][: vp_y // 4, vp_x // 4 :] = 1
                vp[i][: vp_y // 4, : vp_x // 4] = 2
                vp[i][vp_y // 4 :, : vp_x // 4] = 3
                vp[i][vp_y // 4 :, vp_x // 4 :] = 4
        l_vp = F.cross_entropy(out[3], vp)
        
        return l_vp
