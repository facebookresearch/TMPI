# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
from dataset import TMPIDataset
from torch.utils.data import DataLoader 
import numpy as np
import argparse
from tmpi import TMPI
import config
import math
from moviepy.editor import ImageSequenceClip
from tmpi_renderer_gl import TMPIRendererGL
from tmpi_renderer import TMPIRenderer

def render_3d_photo(model, sample, poses, use_gl=True):
    src_rgb_tiles = sample['src_rgb_tiles'].cuda() # (b, n, 3, h, w)
    src_disp_tiles = sample['src_disp_tiles'].cuda() # (b, n, 1, h, w)
    src_rgb = sample['src_rgb'].cuda() # (b, 3, h, w)
    src_disp = sample['src_disp'].cuda() # (b, 1, h, w)
    cam_int = sample['cam_int']
    sx = sample['sx'] 
    sy = sample['sy']

    # Predict Tiled Multiplane Images
    mpis, mpi_disp = model( src_rgb_tiles, src_disp_tiles, src_rgb, src_disp) 
    torch.cuda.empty_cache()

    h, w = src_rgb.shape[-2:]
    if use_gl:
        renderer = TMPIRendererGL(h, w)
        tgt_rgb_syn = renderer( mpis.cpu(), mpi_disp.cpu(), poses, cam_int, sx, sy)
    else:
        print("\033[91mNot Using OpenGL. Rendering will be slow.\033[0m")
        device = 'cpu' 
        renderer = TMPIRenderer(h, w, device)
        K = sample['K']

        tgt_rgb_syn = renderer( mpis.to(device),
                                mpi_disp.to(device),
                                poses.to(device).flip(dims=[0]),
                                K.to(device),
                                sx.to(device),
                                sy.to(device))
        
    return tgt_rgb_syn

def render_path(num_frames=90, r_x=0.03, r_y=0.03, r_z=0.03):
    t = torch.arange(num_frames) / (num_frames - 1)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
    poses[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
    poses[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)
    return poses

def test(indir, outdir, use_gl_renderer, crop_edges=True):
    data = TMPIDataset(data_root=indir)
    dataset_size = len(data)
    dataset_loader = DataLoader(
        dataset=data, 
        batch_size=1,
        shuffle=False)
    model = TMPI(num_planes=config.num_planes)
    model = model.cuda().eval()

    model = torch.nn.parallel.DataParallel(model, device_ids=range(torch.cuda.device_count() ))
    model_file = 'weights/mpti_04.pth'
                                        
    model.load_state_dict( torch.load( os.path.join(os.getcwd(), model_file) ) )
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True    
    
    for batch, sample in enumerate(dataset_loader):
        if batch < 1:
            continue
        
        with torch.no_grad():
            poses_tgt = render_path()
            pred_frames = render_3d_photo(model, sample, poses_tgt, use_gl_renderer)
            pred_frames = [ np.clip(np.round(frame * 255), a_min=0, a_max=255).astype(np.uint8) for frame in pred_frames ]

            if crop_edges:
                pred_frames = [ frame[20:-20, 20:-20, :] for frame in pred_frames ]
                
            rgb_clip = ImageSequenceClip(pred_frames, fps=30)
            rgb_clip.write_videofile( os.path.join(outdir, '%s.mp4' % sample['filename'][0]), verbose=False, codec='mpeg4', logger=None, bitrate='2000k')
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--indir", default="test_data", help="Directory with image files to process.")
    parser.add_argument("-o", "--outdir", default="output", help="Output directory.")
    parser.add_argument("--pytorch_renderer", action="store_true", default=False, help="Use PyTorch renderer to render the tiled MPI.")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    test(args.indir, args.outdir, not args.pytorch_renderer)

