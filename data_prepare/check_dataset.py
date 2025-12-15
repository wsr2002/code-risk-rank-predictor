#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查生成的数据集"""

import pandas as pd

df = pd.read_csv('python_projects_dataset.csv')

print("=" * 60)
print("数据集统计")
print("=" * 60)
print(f"\n总文件数: {len(df)}")
print(f"项目数: {df['project'].nunique()}")

print(f"\n各项目文件数:")
project_counts = df['project'].value_counts()
for project, count in project_counts.items():
    print(f"  {project}: {count} 个文件")

print(f"\n特征统计:")
print(df[['commit_count', 'churn', 'author_count', 'cc', 'mi', 'loc']].describe())

print(f"\n数据预览（前10行）:")
print(df.head(10).to_string())

print(f"\n有commit的文件数: {(df['commit_count'] > 0).sum()}")
print(f"有churn的文件数: {(df['churn'] > 0).sum()}")
print(f"有作者的文件数: {(df['author_count'] > 0).sum()}")

