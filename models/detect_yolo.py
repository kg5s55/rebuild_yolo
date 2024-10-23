#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rebuild_yolo 
@File    ：detect_yolo.py
@Author  ：kg5s55
@Description: 
"""
import torch.nn as nn
from pathlib import Path
from utils.check import LOGGER, make_divisible
import contextlib

from models.common_model import *


class YOLOModel(nn.Module):
    def __init__(self, cfg, ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        self.model = build_model()


def build_model(model_cfg, ch):
    # 打印yaml信息
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    for i, (f, n, m, args) in enumerate(model_cfg['backbone'] + model_cfg['head']):
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        # n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if model_cfg['anchors'] == "free":
            anchors = 0
        else:
            anchors = model_cfg['anchors']
        num_classes = model_cfg['nc']
        num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        num_output = num_anchors * (num_classes + 5)
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        if m in {
            Conv, GhostConv, Bottleneck, GhostBottleneck, SPPF, DWConv, CrossConv, BottleneckCSP,
            C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            # 输入通道、输出通道
            c1, c2 = ch[f], args[0]
            if c2 != num_output:  # if not output
                c2 = make_divisible(c2, 8)
            # 输入通道、输出通道、bottleneck的个数、
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m is Contract:
                c2 = ch[f] * args[0] ** 2

            else:
                c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        num_para = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.num_para = i, f, t, num_para  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n:>3}{num_para:10.0f}  {t:<40}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
