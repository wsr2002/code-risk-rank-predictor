#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本使用示例
"""

from prepare_dataset import prepare_dataset

# 示例1: 处理单个项目
if __name__ == "__main__":
    # 方法1: 直接指定项目路径列表
    repo_paths = [
        r"D:\testsprite\requests",
        # r"D:\testsprite\another_project",
        # r"D:\testsprite\yet_another_project",
    ]
    
    # 生成数据集
    df = prepare_dataset(repo_paths, output_file="dataset.csv")
    
    print("\n数据集预览:")
    print(df.head(10))

