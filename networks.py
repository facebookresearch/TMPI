# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

"""
The recursive UNet generation code is Based on the implementation of Weder et al.'s 
Routed Fusion: https://github.com/weders/RoutedFusion.git
"""

class TileMPINet(torch.nn.Module):

    def __init__(self, cin, c0, cout, depth, level=0):

        super().__init__()
        self.c0 = c0
        self.depth = depth

        if level > 0:
            post_activation = torch.nn.ReLU()
        else:
            post_activation = torch.nn.Identity()
            
        self.pre = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(cin, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
        )

        self.post = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),            
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, cout, kernel_size=3, stride=1, padding=0),
            post_activation,
        )

        if depth > 1:
            self.process = TileMPINet(c0, 2 * c0, 2 * c0, depth - 1, level + 1)
        else:
            self.process = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(c0, 2 * c0, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(2 * c0, 2 * c0, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),                
            )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):

        features = self.pre(data)
        self.features = features
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        output = self.post(torch.cat((features, upsampled), dim=1))

        return output

class UNet(torch.nn.Module):
    def __init__(self, cin, c0, cout, nlayers, post_activation=torch.nn.ReLU()):

        super().__init__()
        self.c0 = c0
        self.nlayers = nlayers

        self.pre = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(cin, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU()
        )
        
        self.post = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, c0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, cout, kernel_size=3, stride=1, padding=0),
            torch.nn.GroupNorm(1, cout),
            post_activation,
        )

        if nlayers > 1:
            self.process = UNet(c0, 2 * c0, 2 * c0, nlayers - 1)
        else:
            self.process = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(c0, 2 * c0, kernel_size=3, stride=1, padding=0),
                torch.nn.GroupNorm(1, 2 * c0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(2 * c0, 2 * c0, kernel_size=3, stride=1, padding=0),
                torch.nn.GroupNorm(1, 2 * c0),
                torch.nn.ReLU()
            )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):
        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]

        output = self.post(torch.cat((features, upsampled), dim=1))

        return output


class DepthConfidenceNet(torch.nn.Module):
    
    def __init__(self):

        super().__init__()

        cin = 4
        c0 = 24 
        cout = 1
        nlayers = 4 

        self.pre = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(cin, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
        )

        self.process = UNet(c0, 2 * c0, 2 * c0, nlayers - 1, post_activation=nn.Identity())

        self.post = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * c0, 2 * c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(2 * c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, cout, kernel_size=3, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        
        self.confidence = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * c0, 2 * c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(2 * c0, c0, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(c0, cout, kernel_size=3, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):

        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        
        confidence = self.confidence(torch.cat((features, upsampled), dim=1))
        depth = self.post(torch.cat((features, upsampled), dim=1))

        return depth, confidence


