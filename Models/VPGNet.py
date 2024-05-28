import torch
import torch.nn as nn


class VPGNet(nn.Module):
    def __init__(self, n_classes=17, phase=2):
        super(VPPNet, self).__init__()
        self.n_classes = n_classes
        self.phase = phase
        self.shared = nn.Sequential(
            # (3 x 480 x 640)
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(11),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # (96 x 59 x 79)
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (256 x 29 x 39)
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            # (384 x 29 x 39)
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            # (384 x 29 x 39)
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (384 x 14 x 19)
        )
        self.vpp = nn.Sequential(
            # (384 x 14 x 19)
            nn.Conv2d(
                in_channels=384, out_channels=4096, kernel_size=6, stride=1, padding=3
            ),
            nn.ReLU(),
            nn.Dropout2d(),
            # (4096 x 15 x 20)
            nn.Conv2d(
                in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Dropout2d(),
            # (4096 x 15 x 20)
            nn.Conv2d(
                in_channels=4096, out_channels=320, kernel_size=1, stride=1, padding=0
            ),
            # (320 x 15 x 20)
        )        
        # TODO: Temporary fix
        if phase==1:
            return
        self.gridBox = nn.Sequential(
            # (384 x 14 x 19)
            nn.Conv2d(
                in_channels=384, out_channels=4096, kernel_size=6, stride=1, padding=3
            ),
            nn.ReLU(),
            nn.Dropout2d(),
            # (4096 x 15 x 20)
            nn.Conv2d(
                in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Dropout2d(),
            # (4096 x 15 x 20)
            nn.Conv2d(
                in_channels=4096, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
            # (256 x 15 x 20)
        )
        self.objectMask = nn.Sequential(
            # (384 x 14 x 19)
            nn.Conv2d(
                in_channels=384, out_channels=4096, kernel_size=6, stride=1, padding=3
            ),
            nn.ReLU(),
            nn.Dropout2d(),
            # (4096 x 15 x 20)
            nn.Conv2d(
                in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Dropout2d(),
            # (4096 x 15 x 20)
            nn.Conv2d(
                in_channels=4096, out_channels=128, kernel_size=1, stride=1, padding=0
            ),
            # (128 x 15 x 20)
        )
        self.multiLabel = nn.Sequential(
        # (384 x 14 x 19)
        nn.Conv2d(
            in_channels=384, out_channels=4096, kernel_size=6, stride=1, padding=3
        ),
        nn.ReLU(),
        nn.Dropout2d(),
        # (4096 x 15 x 20)
        nn.Conv2d(
            in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0
        ),
        nn.ReLU(),
        nn.Dropout2d(),
        # (4096 x 15 x 20)
        nn.Conv2d(
            in_channels=4096,
            out_channels=16 * (n_classes + 1),
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        # (288 x 15 x 20) 1+17-classes
        # nn.Conv2d(in_channels=4096, out_channels=1024, kernel_size=1, stride=1, padding=0),
        # (1024 x 15 x 20) 64-classes
    )


    def tiling(self, x, tile_size:int =8):
        batch_size, n_channels, w, h = x.size() # -1, 320, 15, 20
        out_channels = int(n_channels / tile_size / tile_size) # 320/8/8=5
        x = x.view(batch_size, out_channels, tile_size * tile_size, w, h) # -1, 5, 64, 15, 20
        out = torch.zeros(batch_size, out_channels, w*tile_size, h*tile_size) # -1, 5, 120, 160
        for i in range(tile_size):
            for j in range(tile_size):
                out[:, :, i::tile_size, j::tile_size] = x[:, :, tile_size*i+j]
        del x
        device = next(self.parameters()).device
        return out.to(device)

    def forward(self, x):
        shared = self.shared(x)
        vpp = self.tiling(self.vpp(shared))  # (320 x 15 x 20)
        if self.phase==1:
            return shared, vpp
        gridBox = self.tiling(self.gridBox(shared)) # (4 x 120 x 160)
        objectMask = self.tiling(self.objectMask(shared)) # (2 x 120 x 160)

        multiLabel = self.tiling(self.multiLabel(shared), tile_size=4) # (64 x 60 x 80) 1+63-classes

        return gridBox, objectMask, multiLabel, vpp
