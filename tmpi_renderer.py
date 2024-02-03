# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import config
from torchvision import models
from homography_sampler import HomographySample
import mpi_rendering
import time
from tqdm import tqdm

class TMPIRenderer(torch.nn.Module):
    def __init__(self, ho, wo, device='cuda:0'):
        super(TMPIRenderer, self).__init__()
        self.ho = ho
        self.wo = wo
        self.device = device

    def render(self, mpis, mpi_disp, cam_ext, K, sx, sy): 
        b, nt, n, _, h, w = mpis.shape

        cam_ext = cam_ext.expand( nt, -1, -1 )
        K = K.view( b * nt, 3, 3)

        tile_sz = h
        pad_sz = int(tile_sz * config.padsz2tile_ratio)

        # Pad MPIs to account for over-flow.
        # Alpha and color are padded in different ways to prevent tile edges showing when grid sampling
        tile_pad_sz = config.render_padsz
        mpis = mpis.view( b * nt * n, 4, h, w )
        mpis_color = F.pad( mpis[:, :3, ...], (tile_pad_sz, tile_pad_sz, tile_pad_sz, tile_pad_sz), 'replicate')
        mpis_alpha = F.pad( mpis[:, 3:, ...], (tile_pad_sz, tile_pad_sz, tile_pad_sz, tile_pad_sz), 'constant', 0)
        mpis = torch.cat( (mpis_color, mpis_alpha), 1).view(b * nt, n, 4, h + tile_pad_sz * 2, w + tile_pad_sz * 2)
        mpi_disp = mpi_disp.view(b * nt, n)

        hp, wp = mpis.shape[-2:]

        # Update K and sx/sy
        sx, sy = sx - tile_pad_sz, sy - tile_pad_sz
        K[:, 0, 2] = K[:, 0, 2] + tile_pad_sz
        K[:, 1, 2] = K[:, 1, 2] + tile_pad_sz 

        sx = sx.unsqueeze(-1).expand(-1, -1, n).flatten()
        sy = sy.unsqueeze(-1).expand(-1, -1, n).flatten()

        mpi_depth_src = torch.reciprocal(mpi_disp).flatten()  # BxS

        K_inv = torch.linalg.inv(K)
        G = cam_ext
        
        homography_sampler = HomographySample(hp, wp, self.device)
        
        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity( homography_sampler.meshgrid, mpi_disp, K_inv )
        xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity( xyz_src_BS3HW, G )

        mpi_xyz_src = torch.cat((mpis[:, :, :3, ...], mpis[:, :, 3:4, ...], xyz_tgt_BS3HW), dim=2)  

        G_tgt_src_Bs44 = G.unsqueeze(1).repeat(1, n, 1, 1).contiguous().reshape(b * nt * n, 4, 4) 
        K_src_inv_Bs33 = K_inv.unsqueeze(1).repeat(1, n, 1, 1).contiguous().reshape(b * nt * n, 3, 3)  
        K_tgt_Bs33 = K.unsqueeze(1).repeat(1, n, 1, 1).contiguous().reshape(b * nt * n, 3, 3) 
        
        tgt_mpi_xyz_BsCHW, tgt_mask_BsHW = homography_sampler.sample(mpi_xyz_src.view(b * nt * n, 7, hp, wp),
                                                                     mpi_depth_src.view(b * nt * n),
                                                                     G_tgt_src_Bs44,
                                                                     K_src_inv_Bs33,
                                                                     K_tgt_Bs33)

        tgt_rgb_BS3HW = tgt_mpi_xyz_BsCHW[:, 0:3, :, :]
        tgt_sigma_BS1HW = tgt_mpi_xyz_BsCHW[:, 3:4, :, :]
        tgt_xyz_BS3HW = tgt_mpi_xyz_BsCHW[:, 4:, :, :]

        # Sort tile planes by depth across the entire image
        mpi_depth_src = mpi_depth_src.view(b, -1)
        _, o = torch.sort(mpi_depth_src, dim=-1)
        o = (o + (torch.arange(0, b).view(b, -1) * nt * n).to(self.device)).flatten()

        tgt_rgb_BS3HW = tgt_rgb_BS3HW[o, ...].view(b, nt * n, 3, -1)
        tgt_sigma_BS1HW = tgt_sigma_BS1HW[o, ...].view(b, nt * n, 1, -1)
        mpi_depth_src = mpi_depth_src.flatten()[o].view(b, -1)
        sy = sy[o]
        sx = sx[o]
        
        transmittance = torch.ones((b, 1, (self.ho + tile_sz + 2 * tile_pad_sz) * (self.wo + tile_sz + 2 * tile_pad_sz))).to(self.device)
        out_rgb = torch.zeros((b, 3, (self.ho + tile_sz + 2 * tile_pad_sz) * (self.wo + tile_sz + 2 * tile_pad_sz))).to(self.device)
        out_depth = torch.zeros((b, 1, (self.ho + tile_sz + 2 * tile_pad_sz) * (self.wo + tile_sz + 2 * tile_pad_sz))).to(self.device)

        x, y = torch.meshgrid( torch.arange(0, wp), torch.arange(0, hp), indexing='xy' )
        x, y = x.unsqueeze(0).to(self.device), y.unsqueeze(0).to(self.device)

        t, l = sy.view(b, -1) + tile_pad_sz, sx.view(b, -1) + tile_pad_sz

        for i in range(nt * n):
            index = ((y + t[:, i, None, None]) * (self.wo + tile_sz + 2 * tile_pad_sz) + (x + l[:, i, None, None])).view(b, 1, -1)
            transmittance_ = torch.gather(transmittance, -1, index)
            src_rgb = tgt_rgb_BS3HW[:, i, ...] * tgt_sigma_BS1HW[:, i, ...] * transmittance_
            src_depth = mpi_depth_src[:, i, None, None] * tgt_sigma_BS1HW[:, i, ...] * transmittance_
            out_rgb.scatter_add_(-1, index.expand(-1, 3, -1), src_rgb)
            out_depth.scatter_add_(-1, index, src_depth)
            transmittance = transmittance.scatter(-1, index, transmittance_ * (1 - tgt_sigma_BS1HW[:, i, ...]))
        
        tgt_rgb_syn = out_rgb.view(b, 3, (self.ho + tile_sz + 2 * tile_pad_sz), (self.wo + tile_sz + 2 * tile_pad_sz))
        transmittance = transmittance.view(b, 1, (self.ho + tile_sz + 2 * tile_pad_sz), (self.wo + tile_sz + 2 * tile_pad_sz))
        
        ho, wo = tgt_rgb_syn.shape[-2:]
        tgt_rgb_syn = tgt_rgb_syn[:, :, tile_pad_sz:ho-tile_sz-tile_pad_sz, tile_pad_sz:wo-tile_sz-tile_pad_sz]
        transmittance = transmittance[:, :, tile_pad_sz:ho-tile_sz-tile_pad_sz, tile_pad_sz:wo-tile_sz-tile_pad_sz]

        return tgt_rgb_syn

    def forward(self, mpis, mpi_disp, cam_ext, K, sx, sy):
        pred_frames = []
        for i in tqdm(range(cam_ext.shape[0])):
            pred_frames.append( self.render(mpis, mpi_disp, cam_ext[i, ...], K.clone(), sx, sy)[0, ...].permute(1, 2, 0).cpu().numpy() )
            
        return pred_frames
    
