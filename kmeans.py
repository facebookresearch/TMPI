# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

class KMeans:
    """ Weighted K-Means clustering """
    
    def __init__(self, k=4, featsz=1, batchsz=1, device='cpu'):
        super().__init__()
        self.k = k
        self.device = device
        # Start off with a linear distribution for the cluster centers
        self.cluster_m = torch.linspace(0, 1, k).view(1, k, 1).expand(batchsz, -1, featsz).to(device)
        
    def train( self, samples, conf ):
        b, n, m = samples.shape
        o = torch.randperm( n )
        samples =  samples[:, o, :]
        conf = conf[:, o, :]

        for j in range(50):
            # Calculate distance from each sample to each cluster center
            samples_ = samples.unsqueeze(-2).expand(-1, -1, self.k, -1)
            cluster_m_ = self.cluster_m.unsqueeze(-3).expand(-1, n, -1, -1)
            d = torch.square(samples_ - cluster_m_).sum(-1)
            
            w_idx = torch.min(d, dim=-1)[1].view(b, n, 1).expand(-1, -1, m)

            # Calculate new cluster centers as the weighted mean of the closest samples
            for i in range(self.k):
                w_i = w_idx == i
                self.cluster_m[:, i, :] = torch.sum(samples * conf * w_i, 1) / torch.sum(w_i * conf, 1).clamp(min=1)

    def test( self, samples ):
        b, n, m = samples.shape

        samples = samples.unsqueeze(-2).expand(-1, -1, self.k, -1)
        cluster_m = self.cluster_m.unsqueeze(-3).expand(-1, n, -1, -1)
        
        d = torch.square(samples - cluster_m).sum(-1)
        w_idx = torch.min(d, dim=-1)[1].view(b, n, 1).expand(-1, -1, m)
        w = torch.gather( self.cluster_m, 1, w_idx)

        # Return the cluster label, and cluster index for each sample
        return w.view(b, n, m), w_idx[:, :, 0]
