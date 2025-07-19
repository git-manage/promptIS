#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from .codeshield import CodeShield
import asyncio
import pdb
from .insecure_code_detector.languages import Language
from bandit.core import manager, config, issue

def scan_code(code: str):
    # 初始化 Bandit 配置
    bandit_config = config.BanditConfig()  # 使用默认配置

    # 创建 BanditManager 实例
    b_mgr = manager.BanditManager(bandit_config, 'file')

    # 将代码字符串写入临时文件（Bandit 需要文件路径）
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as temp:
        temp.write(code)
        temp_path = temp.name

    # 添加文件到 BanditManager
    b_mgr.discover_files([temp_path], 'basestring')

    # 运行扫描
    b_mgr.run_tests()

    # 获取结果
    results = b_mgr.get_issue_list()
    import os
    os.unlink(temp_path)
    
    if results == []:
        return 100
    else:
        score = 100
        for vul in results:
            if vul.severity == 'HIGH':
                vul_score = 50
            elif vul.severity == 'MEDIUM':
                vul_score = 30
            elif vul.severity == 'LOW':
                vul_score = 10
            if vul.confidence == 'HIGH':
                vul_confidence = 1
            elif vul.confidence == 'MEDIUM':
                vul_confidence = 0.6
            elif vul.confidence == 'LOW':   
                vul_confidence = 0.2
            score -= vul_score * vul_confidence
        return max(0, score)






