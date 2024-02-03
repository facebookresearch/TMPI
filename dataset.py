# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import glob
import config
from dpt_wrapper import DPTWrapper
from utils import imutils
from utils import utils
import config

class TMPIDataset(Dataset):
    def __init__(self,
                 data_root,
                 image_ext=["png", "jpeg", "jpg"],
                 ):

        super(TMPIDataset, self).__init__()
        self.data_root = data_root
        self.image_path_list = []
        for ext in image_ext:
            self.image_path_list.extend( glob.glob(self.data_root + "/**/*.%s" % ext, recursive=True) )
        self.image_path_list.sort()
        
        self.monocular_depth = DPTWrapper(model_path='./DPT/weights/dpt_hybrid-midas-501f0c75.pt')
        
    # Subdivide the image into over-lapping tiles
    def tiles(self, src_disp, src_rgb, K, tile_sz, pad_sz):

        bs, _, h, w = src_disp.shape
        K_, sx_, sy_, dmap_, rgb_ = [], [], [], [], []

        sy = torch.arange(0, h, tile_sz - pad_sz)
        sx = torch.arange(0, w, tile_sz - pad_sz)

        src_disp = F.pad(src_disp, (0, tile_sz, 0, tile_sz), 'replicate') 
        src_rgb = F.pad(src_rgb, (0, tile_sz, 0, tile_sz), 'replicate') 

        K_, src_disp_, src_rgb_,  sx_, sy_ = [], [], [], [], []
        for y in sy:
            for x in sx:
                l, r, t, b = x, x + tile_sz, y, y + tile_sz
                Ki = K.clone()
                Ki[:, 0, 2] = Ki[:, 0, 2] - x
                Ki[:, 1, 2] = Ki[:, 1, 2] - y

                K_.append(Ki)
                src_disp_.append( src_disp[:, :, t:b, l:r] )
                src_rgb_.append( src_rgb[:, :, t:b, l:r] )
                sx_.append(x)
                sy_.append(y)

        src_rgb_ = torch.stack(src_rgb_, 1)
        src_disp_ = torch.stack(src_disp_, 1)
        K_ = torch.stack(K_, 1)
        sx_, sy_ = torch.tensor(sx_).unsqueeze(0).expand(bs, -1), torch.tensor(sy_).unsqueeze(0).expand(bs, -1)
        return src_disp_, src_rgb_, K_, sx_, sy_

    def __len__(self):
        return len(self.image_path_list)

    
    def __getitem__(self, idx):

        src_rgb = imutils.png2np( self.image_path_list[idx] ).astype(np.float32)
        h, w = src_rgb.shape[:2]

        # Scale the image if too large
        if h >= w and h >= config.imgsz_max:
            h_scaled, w_scaled = config.imgsz_max, int(config.imgsz_max / h * w)
        elif w > h and w >= config.imgsz_max:
            h_scaled, w_scaled = int(config.imgsz_max / w * h), config.imgsz_max
        else:
            h_scaled, w_scaled = h, w
            
        src_rgb = F.interpolate(torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0), (h_scaled, w_scaled), mode="bilinear")

        # Estimate monocular depth
        src_disp = torch.from_numpy(self.monocular_depth( src_rgb.squeeze().permute(1, 2, 0).numpy())).unsqueeze(0).unsqueeze(0)
        src_disp = (src_disp - torch.min(src_disp)) / (torch.max(src_disp) - torch.min(src_disp)) # Normalize to [0, 1]
        
        K = torch.tensor([
            [0.58, 0, 0.5],
            [0, 0.58, 0.5],
            [0, 0, 1]
        ]).unsqueeze(0)
        K[:, 0, :] *= w_scaled
        K[:, 1, :] *= h_scaled

        tile_sz = int(np.clip(utils.next_power_of_two(config.tilesz2w_ratio * w_scaled - 1), a_min=config.tilesz_min, a_max=config.tilesz_max))
        pad_sz = int(tile_sz * config.padsz2tile_ratio)
        src_disp_tiles, src_rgb_tiles, K_tiles, sx, sy = self.tiles(src_disp, src_rgb, K, tile_sz, pad_sz)
        
        return {
            "src_rgb_tiles": src_rgb_tiles.squeeze(0),
            "src_disp_tiles": src_disp_tiles.squeeze(0),
            "K": K_tiles.squeeze(0),
            "sx": sx.squeeze(0),
            "sy": sy.squeeze(0),
            "src_rgb": src_rgb.squeeze(0),
            "src_disp": src_disp.squeeze(0),
            "src_width": w,
            "src_height": h,
            "tile_sz": tile_sz,
            "pad_sz": pad_sz,
            "cam_int": K.squeeze(0), 
            "cam_ext": torch.eye(4),
            "filename": os.path.basename(os.path.splitext(self.image_path_list[idx])[0])
        }


