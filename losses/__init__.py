import torch
import torch.nn as nn
import torch.nn.functional as F


class FourTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super(FourTaskLoss, self).__init__()
        if weights is None:
            self.weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Default weights
        else:
            self.weights = torch.tensor(weights)

    def forward(self, out, gridbox, seg, vpxy):
        # print(f"gridbox: {gridbox.shape} {gridbox.dtype} {gridbox.device}")
        l_reg = F.l1_loss(out[0], gridbox.to('cuda'))
        om = (seg > 0).long()
        # print(f"om: {om.shape} {om.dtype} {om.device}")
        l_om = F.cross_entropy(out[1], om.to('cuda'))
        ml_seg = torch.nn.functional.max_pool2d(seg, 2).long()
        # print(f"ml_seg: {ml_seg.shape} {ml_seg.dtype} {ml_seg.device}")
        l_ml = F.cross_entropy(out[2], ml_seg.to('cuda'))
        vp = torch.zeros((out[3].shape[0], 120, 160)).long()
        for i, (vp_x, vp_y) in enumerate(vpxy):
            if vp_x != 0 or vp_y != 0:
                vp[i][: vp_y // 4, vp_x // 4 :] = 1
                vp[i][: vp_y // 4, : vp_x // 4] = 2
                vp[i][vp_y // 4 :, : vp_x // 4] = 3
                vp[i][vp_y // 4 :, vp_x // 4 :] = 4
        # print(f"vp: {vp.shape} {vp.dtype} {vp.device}")
        l_vp = F.cross_entropy(out[3], vp.to('cuda'))
        # total_loss = torch.zeros(1, requires_grad=True, device='cuda')
        # for weight, loss in zip(self.weights, [l_reg, l_om, l_ml, l_vp]):
        # total_loss += weight * loss
        # return total_loss
        # print(l_reg, l_om, l_ml, l_vp)
        return l_reg, l_om, l_ml, l_vp
