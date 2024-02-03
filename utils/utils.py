# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import torch.nn.functional as F

class InputPadder:
    """ Pads images such that dimensions are divisible by a given factor.
    Code from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/utils/utils.py
    """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def gradient(x):
    gy = F.conv2d(x, torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda().view((1, 1, 3, 3)), padding=1)
    gx = F.conv2d(x, torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).cuda().view((1, 1, 3, 3)), padding=1)
    return gx ** 2 + gy ** 2

def next_power_of_two(n):
    return 2 ** np.ceil(np.log2(n))
