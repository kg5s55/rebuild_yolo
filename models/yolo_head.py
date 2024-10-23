#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rebuild_yolo 
@File    ：yolo_head.py
@Author  ：kg5s55
@Description: 
"""
import torch
import torch.nn as nn


class YOLOv5Head(nn.Module):
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, num_classes=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_output = num_classes + 5
        self.num_detect = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.num_detect)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.num_detect)]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.num_detect, -1, 2))
        self.output_conv = nn.ModuleList(nn.Conv2d(x,
                                                   self.num_output * self.num_anchors,
                                                   1) for x in ch)  # output conv
        self.inplace = inplace

    def forward(self, x):
        output = []
        for i in range(self.num_detect):
            x[i] = self.output_conv[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors,
                             self.num_output, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                # if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                #     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_classes + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                output.append(
                    y.view(bs, self.num_anchors * nx * ny, self.num_output)
                )

