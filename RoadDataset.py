import cv2
import os
from glob import glob
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import random
import numpy as np
from skimage.measure import regionprops, label

"""
scene_1
312 05/12
231 05/18
76  08/05
605 08/08
8   08/10

scene_2
53  05/03
232 05/10
32  05/24
48  08/09

scene_3
14  05/03
60  08/09

scene_4
199 05/18
18  08/04
25  08/05
"""
class RandomResizeCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, target):
        # Generate parameters for crop
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        
        # Apply the same crop to both image and target
        img = T.functional.resized_crop(img.unsqueeze(0), i, j, h, w, self.size)
        target = T.functional.resized_crop(target.unsqueeze(0), i, j, h, w, self.size, interpolation=T.InterpolationMode.NEAREST)

        return img.squeeze(0), target.squeeze(0)


class VPGDataset(Dataset):
    def __init__(self, data_root="./data/VPGNet-DB-5ch/Train", phase=1):
        self.data_root = os.path.abspath(data_root)
        self.scene_paths = sorted(glob(os.path.join(self.data_root, 'scene_*')))
        self.scenes = [os.path.basename(_path) for _path in self.scene_paths]
        self.mat_files = [_mat_file for _path in self.scene_paths for _mat_file in sorted(glob(os.path.join(_path, '*.mat')))]
        self.phase = phase
        self.transform = RandomResizeCrop(size=(299,299))
        # self.transform = T.Resize(size=(299,299))

    def train_test_split(self):
        os.makedirs(os.path.join(self.data_root, 'Train'),exist_ok=True)
        os.makedirs(os.path.join(self.data_root, 'Test'),exist_ok=True)
        os.makedirs(os.path.join(self.data_root, 'Val'),exist_ok=True)
        for _path in self.scene_paths:
            scene_name = os.path.basename(_path)
            os.makedirs(os.path.join(self.data_root, 'Train', scene_name),exist_ok=True)
            os.makedirs(os.path.join(self.data_root, 'Test', scene_name),exist_ok=True)
            os.makedirs(os.path.join(self.data_root, 'Val', scene_name),exist_ok=True)
            scene_timestamps = sorted(glob(os.path.join(_path, '*')))
            num_items = len(scene_timestamps)
            for i, __path in enumerate(scene_timestamps):
                _dir = "Val"
                if i < 0.625 * num_items:
                    _dir = "Train"
                elif i > 0.875 * num_items:
                    _dir = "Test"
                _timestamp = os.path.basename(__path)
                for mat_file in sorted(glob(os.path.join(__path, '*.mat'))):
                    _num_frame = os.path.basename(mat_file)
                    _save_name = os.path.join(self.data_root, _dir, scene_name, f'{_timestamp}_{_num_frame}')
                    if os.path.islink(_save_name):
                        print('\033[1;31m' + f'{_save_name} already exists!' + '\033[0m')
                    else:
                        os.symlink(mat_file, _save_name)
                        print('\033[1;32m' + f"file linked at {os.path.relpath(_save_name)}" + '\033[0m')
                        print()

    def __len__(self):
        return len(self.mat_files)


    def create_targets(self, annotated_tensor):
        # Get unique object classes excluding the background (0)
        object_classes = np.unique(annotated_tensor)
        object_classes = object_classes[object_classes != 0]  # Exclude background

        boxes = []
        labels = []
        masks = []

        for cls in object_classes:
            # Create a binary mask for the current class
            mask = (annotated_tensor == cls).int()

            # Find bounding box for the current mask
            label_img = label(mask)
            regions = regionprops(label_img)

            for region in regions:
                # Bounding box in format [x_min, y_min, x_max, y_max]
                minr, minc, maxr, maxc = region.bbox
                boxes.append([minc, minr, maxc, maxr])
                labels.append(int(cls))
                masks.append(mask)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        if masks:
            masks = torch.stack(masks)
        else:
            boxes = torch.empty((0,4),dtype=torch.float32)
            labels = torch.empty((0), dtype=torch.int64)
            masks = torch.empty(0,299,299)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }
        assert len(masks)==len(boxes) and len(boxes)==len(labels), "Numbers of boxes don't match"
        return target
    
    def __getitem__(self, index):
        data = loadmat(os.readlink(self.mat_files[index]))['rgb_seg_vp']
        rgb = T.ToTensor()(data[:,:,:3])
        seg = torch.tensor(data[:,:,3], dtype=torch.float)
        vp = torch.tensor(data[:,:,4], dtype=torch.int)

        """ downsample vp by 4 """
        # vp = vp.unsqueeze(0).unsqueeze(0)
        # vp = F.max_pool2d(vp, kernel_size=4)
        # vp = vp.squeeze(0).squeeze(0).to(torch.long)
        vp = vp.to(torch.long)

        """ vp -> quadrant """
        non_zero_indices = np.nonzero(vp)
        vp = torch.zeros_like(vp)
        if len(non_zero_indices) > 0:
            row_idx, col_idx = non_zero_indices[0][0], non_zero_indices[0][1]
            vp[:row_idx,:col_idx] = 2
            vp[:row_idx,col_idx:] = 1
            vp[row_idx:,:col_idx] = 3
            vp[row_idx:,col_idx:] = 4

        if self.phase == 1:
            rgb = self.transform(rgb)
            vp = self.transform(vp.unsqueeze(0)).squeeze(0)
            return rgb, vp
        
        # TODO: seg, bbox, vp 전처리
        vp = T.functional.resize(vp.unsqueeze(0), size=(299,299), interpolation=T.InterpolationMode.NEAREST).squeeze(0)
        rgb = self.transform(rgb)
        seg = self.transform(seg)

        target = self.create_targets(seg)
        return rgb, target, vp



class CeyMoDataset(Dataset):
    def __init__(self, path='./Data/CeyMo/train'):
        self.data_root = path
        self.bbox_path = sorted(glob(os.path.join(self.data_root, 'bbox_annotations/*.xml')))
        self.image_path = sorted(glob(os.path.join(self.data_root, 'images/*.jpg')))
        self.mask_path = sorted(glob(os.path.join(self.data_root, 'mask_annotations/*.png')))
        self.polygon_path = sorted(glob(os.path.join(self.data_root, 'polygon_annotations/*.json')))
        assert len(self.bbox_path) == len(self.image_path) == len(self.mask_path) == len(self.polygon_path), '\033[1;31m' + 'Number of Annotations don\t match!' + '\033[0m'    
        self.filenames = [os.path.basename(_path).split('.jpg')[0] for _path in self.image_path]


    def __len__(self):
        return len(self.image_path)
    
    def get_label_and_bbox(self, bbox_path='Data/CeyMo/train/bbox_annotations/1.xml'):
        import xml.etree.ElementTree as ET
        tree = ET.parse(bbox_path)
        root = tree.getroot()
        objects = root.findall('object')
        labels = [obj.find('name').text for obj in objects]
        _boxes = [obj.find('bndbox') for obj in objects]; boxes = [[int(x.text) for x in list(_box.iter())[1:]] for _box in _boxes]
        return labels, boxes
    
    def get_mask(self, mask_path, boxes, labels):
        masks = []
        mask = cv2.imread(mask_path); mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        '''
        zip(boxes, labels) iter 돌면서 
        1920x1080 중 box 범위 내에서 label에 해당하는 RGB값 비교해서 mask 만들어서 
        masks로 리스트나 텐서 형태로 반환해야 함
        
        Road Marking Class	        Color Code
        Bus Lane (BL)	            (0,255,255)
        Cycle Lane (CL)	            (0,128,255)
        Diamond (DM)	            (178,102,255)
        Junction Box (JB)	        (255,255,51)
        Left Arrow (LA)         	(255,102,178)
        Pedestrian Crossing (PC)	(255,255,0)
        Right Arrow (RA)	        (255,0,127)
        Straight Arrow (SA)	        (255,0,255)
        Slow (SL)	                (0,255,0)
        Straight-Left Arrow (SLA)	(255,128,0)
        Straight-Right Arrow (SRA)	(255,0,0)
        '''
        color_dict = {0:(0,0,0), 1:(255,0,255), 2:(255,102,178), 3:(255,0,127), 
                      4:(255,128,0), 5:(255,0,0), 6:(178,102,255),7:(255,255,0),
                      8:(255,255,51), 9:(0,255,0), 10:(0,255,255),11:(0,128,255)}      
        for _box, _label in zip(boxes, labels):
            _xmin, _ymin, _xmax, _ymax = list(map(int, _box))
            _color = color_dict[_label.item()]
            _mask = torch.zeros((1080,1920))
            _label_mask = torch.tensor(np.prod(mask==_color, axis=2))
            _mask[_ymin:_ymax+1, _xmin:_xmax+1] = _label_mask[_ymin:_ymax+1, _xmin:_xmax+1]
            masks.append(_mask)
        if len(masks)==0:
            return torch.empty((0,1920,1080))
            
        return torch.stack(masks)
    
    def __getitem__(self, idx):
        ''' return rgb, target(boxes, labels, masks), fileinfo'''
        class_dict = {'Background':0, 'SA':1, 'LA':2, 'RA':3, 'SLA':4, 'SRA':5, 'DM':6, 'PC':7, 'JB':8, 'SL':9, 'BL':10, 'CL':11}
        bbox_path = self.bbox_path[idx]; image_path = self.image_path[idx];
        mask_path = self.mask_path[idx]; polygon_path = self.polygon_path[idx];
        
        raw_rgb = cv2.imread(image_path); rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
        rgb = T.ToTensor()(rgb)

        labels, boxes = self.get_label_and_bbox(bbox_path)
        labels = [class_dict[l] for l in labels]; labels = torch.tensor(labels).to(torch.int64)
        boxes = torch.tensor(boxes).float()

        masks = self.get_mask(mask_path, boxes, labels)

        # transforms 
        rgb = T.functional.resize(rgb, size=(299,299)).float()
        masks = T.functional.resize(masks, size=(299,299), interpolation=T.InterpolationMode.NEAREST).to(torch.uint8)
        boxes = torch.tensor([[box[0]*299/1920, box[1]*299/1080, box[2]*299/1920, box[3]*299/1080] for box in boxes]).float()
        target = {'boxes':boxes, 'labels':labels, 'masks':masks}

        filename = self.filenames[idx]
        return rgb, target, filename



if __name__=='__main__':
    """ train:val:test = 5:2:1 symlink 생성"""
    # if(os.path.isdir("Data/Train")):
    #     import shutil
    #     shutil.rmtree("Data/Train"); print('\033[1;31m' + 'Deleting directory Data/Train!' + '\033[0m')
    #     shutil.rmtree("Data/Test"); print('\033[1;31m' + 'Deleting directory Data/Test!' + '\033[0m')
    #     shutil.rmtree("Data/Val"); print('\033[1;31m' + 'Deleting directory Data/Val!' + '\033[0m')
    # dataset = VPGDataset(data_root="./Data")
    # dataset.train_test_split()

    """ dataset sanity test """
    # dataset = VPGDataset(data_root="data/VPGNet-DB-5ch/Train", phase=1)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    # for i, data in enumerate(dataloader):
    #     print(rgb.shape, vp.shape)
    #     rgb, vp = data
    #     print(vp[0])
    #     # import torchvision
    #     # import matplotlib.pyplot as plt
    #     # plt.imshow(rgb[0].permute(1,2,0))
    #     break

    dataset = CeyMoDataset()
    dataset[0]
    pass