#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：rebuild_yolo 
@File    ：__init__.py
@Author  ：kg5s55
@Description: 
"""
import contextlib
import platform
import threading


def emojis(str=""):
    """Returns an emoji-safe version of a string, stripped of emojis on Windows platforms."""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str