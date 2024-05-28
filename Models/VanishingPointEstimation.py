import torch
import torch.nn as nn
from torchvision.models import inception_v3

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

inception = inception_v3()


class InceptionV3Backbone(nn.Module):
    def __init__(self):
        super(InceptionV3Backbone, self).__init__()
        self.features = nn.Sequential(*list(inception.children())[:15])  # out: Nx768x17x17
        self.out_channels = 768
    
    def forward(self, x):
        x = self.features(x)
        return x

class FullInceptionV3Backbone(nn.Module):
    def __init__(self):
        super(FullInceptionV3Backbone, self).__init__()
        self.features = nn.Sequential(*list(inception.children())[:15],*list(inception.children())[16:18])  # out: Nx768x17x17
        self.out_channels = 2048
    
    def forward(self, x):
        x = self.features(x)
        return x

class VPPNet(nn.Module):
    ''' 
    input: 3x299x299 image
        1. feature extractor: inception_v3 >> 768x17x17 feature space
        2. vpp layer: 768x17x17 > 
    output: 5x299x299 classification logit for each quadrant. (0:None, 1:+/+, 2:-/+, 3:-/-, 4:+/-)
    '''
    def __init__(self):
        super(VPPNet, self).__init__()
        self.backbone = InceptionV3Backbone()
        self.vpp = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 5, kernel_size=3, padding=1),
            nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.vpp(x)
        return x


class MaskRCNNWithInceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNNWithInceptionV3, self).__init__()
        self.backbone = InceptionV3Backbone()

        # Define the RPN anchor generator
        rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                               aspect_ratios=((0.5, 1.0, 2.0),) * 5)

        # Define the RoI align layer
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        # Create the Mask R-CNN model
        self.model = MaskRCNN(backbone=self.backbone,
                              num_classes=num_classes,
                              rpn_anchor_generator=rpn_anchor_generator,
                              box_roi_pool=roi_pooler)

    def forward(self, images, targets=None):
        return self.model(images, targets)

class MaskRCNNVanilla(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNNVanilla, self).__init__()
        self.backbone = FullInceptionV3Backbone()

        # Define the RPN anchor generator
        rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                               aspect_ratios=((0.5, 1.0, 2.0),) * 5)

        # Define the RoI align layer
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        # Create the Mask R-CNN model
        self.model = MaskRCNN(backbone=self.backbone,
                              num_classes=num_classes,
                              rpn_anchor_generator=rpn_anchor_generator,
                              box_roi_pool=roi_pooler)

    def forward(self, images, targets=None):
        return self.model(images, targets)

''' backbone edited 05.22 16:32:00 '''
'''
class MaskRCNNWithInceptionV3(nn.Module):
    def __init__(self, num_classes, backbone=None):
        super(MaskRCNNWithInceptionV3, self).__init__()
        if backbone:
            self.backbone = backbone
        else: 
            self.backbone = torchvision.models.inception__v3()

        # Define the RPN anchor generator
        rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                               aspect_ratios=((0.5, 1.0, 2.0),) * 5)

        # Define the RoI align layer
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        # Create the Mask R-CNN model
        self.model = MaskRCNN(self.backbone,
                              num_classes=num_classes,
                              rpn_anchor_generator=rpn_anchor_generator,
                              box_roi_pool=roi_pooler)

    def forward(self, images, targets=None):
        return self.model(images, targets)
'''




if __name__=="__main__":
    # backbone = InceptionV3Backbone()
    # dummy_input = torch.randn(2, 3, 299, 299) 
    # output = backbone(dummy_input)
    # print(output.shape)

    # Number of classes (including background)
    num_classes = 64

    model = MaskRCNNWithInceptionV3(num_classes)
    print(model)