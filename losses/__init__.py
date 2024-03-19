import torch
import torch.nn as nn

class FourTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super(FourTaskLoss, self).__init__()
        if weights is None:
            self.weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Default weights
        else:
            self.weights = torch.tensor(weights)
    
    def forward(self, out, gridbox, seg, vpxy):
        l_reg = nn.L1Loss()(out[0].cpu().detach().numpy, gridbox)
        om = (seg > 0).long()
        l_om = nn.CrossEntropyLoss()(out[1].cpu().detach().numpy, om)
        ml_seg = torch.nn.functional.max_pool2d(seg, 2).long()
        l_ml = nn.CrossEntropyLoss()(out[2].cpu().detach().numpy, ml_seg)
        vp = torch.zeros(120,160)
        vp_x, vp_y = vpxy
        if vp_x!=0 or vp_y!=0:
            vp[:vp_y//4,vp_x//4:] = 1
            vp[:vp_y//4,:vp_x//4] = 2
            vp[vp_y//4:,:vp_x//4] = 3
            vp[vp_y//4:,vp_x//4:] = 4
        l_vp = nn.CrossEntropyLoss()(out[3].cpu().detach().numpy, vp)
        total_loss = torch.zeros(1, requires_grad=True, device='cuda')
        for weight, loss in zip(self.weights, [l_reg, l_om, l_ml, l_vp]):
            total_loss += weight * loss
        return total_loss
