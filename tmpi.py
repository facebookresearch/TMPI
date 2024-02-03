# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np
from utils import utils
from kmeans import KMeans
import networks

class TMPI(torch.nn.Module):

    def __init__(self, num_planes = 4):
        super().__init__()
        self.num_planes = num_planes

        self.conf_net = networks.DepthConfidenceNet()
        self.mpti_net = networks.TileMPINet( cin=num_planes * 4 + 1, 
                                             c0=32,
                                             cout=num_planes * 4,
                                             depth=3)

    def inpaint(self, rgb, depth, feature_masks):
        b, _, h, w = rgb.shape
        
        feature_masks = torch.cat( (torch.ones_like(feature_masks[:, 0:1, ...]).to(feature_masks.get_device()),
                                    feature_masks[:, :-1, ...]), 1) # What is visible in one layer is masked out in the next
        feature_masks = 1 - feature_masks # Convert from alpha to validity mask

        # Mask out edge regions         
        feature_masks = feature_masks * (utils.gradient(depth) < 0.01).float().expand(-1, feature_masks.shape[1], -1, -1)
        
        feature_masks[:, 0, ...] = 1
        feature_masks = feature_masks.view(b, config.num_planes - 1, 1, h, w).expand(-1, -1, 3, -1, -1)
        rgb_m = rgb.view(b, 1, 3, h, w).expand(-1, config.num_planes - 1, -1, -1, -1) * feature_masks
        sumpool2d = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, divisor_override=1)

        # Construct an image pyramid and inpaint masked regions by interpolating valid values from lower levels        
        def inpaint_level(rgb_, fmask, level):
            if level == 0 or rgb_.shape[-1] == 1 or rgb.shape[-2] == 1:
                return rgb_
            return rgb_ * fmask + (1 - fmask) * F.interpolate(inpaint_level(sumpool2d(rgb_) / sumpool2d(fmask).clamp(min=1),
                                                                            sumpool2d(fmask).clamp(max=1),
                                                                            level - 1),
                                                              size = rgb_.shape[-2:],
                                                              mode='bilinear')
        return inpaint_level(rgb_m.view(b, -1, h, w), feature_masks.reshape(b, -1, h, w), np.log2(h))

    
    def forward(self, rgb_tiles, inv_depth_tiles, rgb, inv_depth):
        device = inv_depth.get_device()

        h, w = rgb.shape[-2:]
        b, n, _, ht, wt = inv_depth_tiles.shape
        tile_sz = ht
        pad_sz = int(tile_sz * config.padsz2tile_ratio)

        # Pad images by tile_sz so that we have a full tile for each image pixel        
        rgb_padded = F.pad(rgb, (0, tile_sz, 0, tile_sz), 'replicate')
        inv_depth_padded = F.pad(inv_depth, (0, tile_sz, 0, tile_sz), 'replicate')

        # Pad to multiple of 16 (for network computation)        
        padder = utils.InputPadder( rgb_padded.shape[-2:], divis_by=16)
        rgb_padded, inv_depth_padded = padder.pad( rgb_padded, inv_depth_padded )
        
        pred_depth, pred_conf = self.conf_net( torch.cat( (rgb_padded, inv_depth_padded), 1) )
        pred_depth = padder.unpad(pred_depth)
        pred_conf = padder.unpad(pred_conf)

        pred_depth_conf = torch.cat( (pred_depth , pred_conf), 1)
        
        # Generate tiles from predicted depth and confidence
        sy = torch.arange(0, h, tile_sz - pad_sz)
        sx = torch.arange(0, w, tile_sz - pad_sz)
            
        depth_conf_tiles = F.unfold( pred_depth_conf[:, :, :(sy[-1] + tile_sz), :(sx[-1] + tile_sz)],
                                     kernel_size = tile_sz,
                                     stride = (tile_sz - pad_sz) )
        depth_conf_tiles = depth_conf_tiles.view(b, 2, tile_sz, tile_sz, -1).permute(0, 4, 1, 2, 3)

        depth_tiles = depth_conf_tiles[:, :, 0, None, ...]
        conf_tiles = depth_conf_tiles[:, :, 1, None, ...]

        # Normalize each depth tile indpendently to [0, 1]
        mn_t = torch.min(depth_tiles.view(b, n, -1), -1)[0].view(b, n, 1, 1, 1)
        mx_t = torch.max(depth_tiles.view(b, n, -1), -1)[0].view(b, n, 1, 1, 1)
        depth_tiles_norm = (depth_tiles - mn_t) / (mx_t - mn_t + 1e-10)

        #
        # K-means clustering for plane depth placement within each tile
        classifier = KMeans(k=config.num_planes - 1, featsz=1, batchsz=b * n, device=device)
        classifier.train( depth_tiles_norm.permute(0, 1, 3, 4, 2).reshape(b * n, ht * wt, -1),
                          conf_tiles.permute(0, 1, 3, 4, 2).reshape(b * n, ht * wt, -1) )
        
        labels, cluster_idx = classifier.test( depth_tiles_norm.permute(0, 1, 3, 4, 2).reshape(b * n, ht * wt, -1) ) # b * n, ht * wt, c

        # Use each tile pixel's label as the assigned plane depth, and unnormalize to original tile depth range
        depth_assign_tiles = labels.permute(0, 2, 1).unsqueeze(-1)
        depth_assign_tiles = depth_assign_tiles.view( b, n, 1, ht, wt)
        depth_assign_tiles = depth_assign_tiles * (mx_t - mn_t) + mn_t
        
        cluster_idx = cluster_idx.view(b, n, 1, ht, wt).expand(-1, -1, config.num_planes - 1, -1, -1)

        dplanes =  classifier.cluster_m.view(b, n, config.num_planes - 1).flip(dims=[-1]).sort(-1, descending=True)[0]
        dplanes = dplanes * (mx_t.view(b, n, 1) - mn_t.view(b, n, 1)) + mn_t.view(b, n, 1)

        plane_masks = cluster_idx - torch.arange(0, config.num_planes - 1).flip(-1).to(device).view(1, 1, config.num_planes - 1, 1, 1).expand(b, n, -1, -1, -1) # b, n, num_planes, ht, wt
        plane_masks = (plane_masks == 0).float()

        occlusion_masks = torch.cumsum( plane_masks, dim=2).clamp(max = 1)

        # Get a prior for what the MPI tiles should look like
        mpi0_rgb_tiles = self.inpaint(rgb_tiles.view(b * n, 3, ht, wt),
                                      inv_depth_tiles.view(b * n, 1, ht, wt),
                                      occlusion_masks.clone().detach().view(b * n, config.num_planes - 1, ht, wt)).clamp(0, 1)
        mpi0_rgba_tiles = torch.cat( (mpi0_rgb_tiles.view(b, n, config.num_planes - 1, -1, ht, wt),
                                      occlusion_masks.view(b, n, -1, 1, ht, wt)), 3)


        ##
        mpi0_rgba_tiles = torch.cat( (mpi0_rgba_tiles, mpi0_rgba_tiles[:, :, -1, None, ...]), 2)
        mpi0_rgba_tiles[:, :, -1, -1, ...] = 1.0
        dplanes = torch.cat( (dplanes, torch.ones_like(dplanes[:, :, -1, None]).to(device) * 1e-7), -1)
        ##
        
        padder = utils.InputPadder( (ht, wt), divis_by=16)
        inv_depth_tiles, mpi0_rgba_tiles = padder.pad( inv_depth_tiles.view(b * n, -1, ht, wt),
                                                       mpi0_rgba_tiles.view(b * n, config.num_planes * 4, ht, wt) )

        mpi_tiles = F.sigmoid( self.mpti_net( torch.cat( (mpi0_rgba_tiles, inv_depth_tiles), 1) ) )
        mpi_tiles = padder.unpad(mpi_tiles).view(b, n, self.num_planes, 4, ht, wt).contiguous()

        # Blend prediction with prior based on alpha
        blendw_tiles = torch.cumprod(1 - mpi_tiles[:, :, :, -1, None, ...], 2)
        blendw_tiles = torch.cat( [torch.ones_like( blendw_tiles[:, :, 0, None, ...]).cuda(), blendw_tiles[:, :, :-1, ...]], 2 )

        mpi0_rgba_tiles = mpi0_rgba_tiles.view(b, n, self.num_planes, -1, ht, wt)
        
        mpis = (blendw_tiles) * mpi_tiles[:, :, :, :3, ...] + (1 - blendw_tiles) * mpi0_rgba_tiles[:, :, :, :3, ...]
        mpi_tiles = torch.cat( (mpis, mpi_tiles[:, :, :, -1, None, ...]), 3)
        
        return mpi_tiles, dplanes




        
