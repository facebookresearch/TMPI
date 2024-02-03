# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

tilesz2w_ratio = 0.125 # The size of a single MPI tile relative to the width of the input image
tilesz_min = 64   # The minimum size of a square MPI tile
tilesz_max = 256  # The maximum size of a square MPI tile
padsz2tile_ratio = 0.125
imgsz_max = 1024   # The maximum height or width of the input image
num_planes = 4
render_padsz = 128 # This is used by the PyTorch Renderer 
