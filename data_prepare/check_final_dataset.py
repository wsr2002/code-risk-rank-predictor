#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查最终生成的数据集"""

import pandas as pd

df = pd.read_csv('python_projects_dataset_final.csv')

print("=" * 60)
print("数据集统计")
print("=" * 60)
print(f"\n总文件数: {len(df)}")
print(f"项目数: {df['project'].nunique()}")

print(f"\n各项目文件数:")
project_counts = df['project'].value_counts()
for project, count in project_counts.items():
    print(f"  {project}: {count} 个文件")

print(f"\n变量统计:")
print(f"  有commit的文件: {(df['commit_count'] > 0).sum()} ({(df['commit_count'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  有churn的文件: {(df['churn'] > 0).sum()} ({(df['churn'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  有author的文件: {(df['author_count'] > 0).sum()} ({(df['author_count'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  有cc的文件: {(df['cc'] > 0).sum()} ({(df['cc'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  有loc的文件: {(df['loc'] > 0).sum()} ({(df['loc'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  label=1的文件: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")

print(f"\n各项目详细统计:")
stats = df.groupby('project').agg({
    'commit_count': ['count', lambda x: (x > 0).sum(), 'mean', 'sum'],
    'churn': ['sum', 'mean'],
    'author_count': ['mean'],
    'label': ['sum', lambda x: (x == 1).sum() / len(x) * 100]
}).round(2)
stats.columns = ['总文件数', '有commit数', '平均commit', '总commit', '总churn', '平均churn', '平均作者数', 'bug_fix数', 'bug_fix比例%']
print(stats)

print(f"\n特征统计:")
print(df[['commit_count', 'churn', 'author_count', 'cc', 'mi', 'loc', 'label']].describe())

print(f"\n数据预览（前10行）:")
print(df.head(10)[['project', 'file', 'commit_count', 'churn', 'author_count', 'cc', 'mi', 'loc', 'label']].to_string())

