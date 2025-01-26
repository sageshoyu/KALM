#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : path_config.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/07/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp

KALM_GIT_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
KALM_ASSETS_ROOT = osp.join(KALM_GIT_ROOT, "asset")
KALM_SRC_ROOT = osp.join(KALM_GIT_ROOT, "src")
KALM_SRC_LIB_ROOT = osp.join(KALM_SRC_ROOT, "kalm")
