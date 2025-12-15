#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查数据集"""

import pandas as pd

df = pd.read_csv('python_projects_dataset_v2.csv')

print("=" * 60)
print("数据集检查")
print("=" * 60)
print(f"\n总文件数: {len(df)}")
print(f"列名: {list(df.columns)}")

print(f"\n前4个变量的统计:")
print(df[['commit_count', 'churn', 'author_count', 'cc']].describe())

print(f"\n有commit的文件数: {(df['commit_count'] > 0).sum()}")
print(f"有churn的文件数: {(df['churn'] > 0).sum()}")
print(f"有author的文件数: {(df['author_count'] > 0).sum()}")
print(f"有cc的文件数: {(df['cc'] > 0).sum()}")
print(f"有label=1的文件数: {(df['label'] == 1).sum() if 'label' in df.columns else 0}")

print(f"\n样本数据（前5行）:")
print(df.head(5))

