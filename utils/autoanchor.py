#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rebuild_yolo 
@File    ：autoanchor.py
@Author  ：kg5s55
@Description: 
"""
from utils.general import colorstr
PREFIX = colorstr("AutoAnchor: ")

def check_anchor_order(m):
    """Checks and corrects anchor order against stride in YOLOv5 Detect() module if necessary."""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        print(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)


def takaa():
    print(f"{PREFIX}Reversing anchor order")
if __name__ == "__main__":
    takaa()