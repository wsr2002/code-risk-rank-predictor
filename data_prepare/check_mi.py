#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查MI分布"""

import pandas as pd

df = pd.read_csv('python_projects_dataset_final.csv')

print("=" * 60)
print("MI (Maintainability Index) 分布分析")
print("=" * 60)

print(f"\n总文件数: {len(df)}")
print(f"MI=50的文件数: {(df['mi'] == 50.0).sum()}")
print(f"MI=50的比例: {(df['mi'] == 50.0).sum() / len(df) * 100:.1f}%")

print(f"\nMI统计:")
print(df['mi'].describe())

print(f"\nMI分布（前10个最常见的值）:")
print(df['mi'].value_counts().head(10))

print(f"\nMI=50的文件特征:")
mi_50 = df[df['mi'] == 50.0]
print(f"  有LOC的文件: {(mi_50['loc'] > 0).sum()}")
print(f"  有CC的文件: {(mi_50['cc'] > 0).sum()}")
print(f"  平均LOC: {mi_50['loc'].mean():.1f}")
print(f"  平均CC: {mi_50['cc'].mean():.1f}")

print(f"\nMI!=50的文件特征:")
mi_not_50 = df[df['mi'] != 50.0]
print(f"  平均LOC: {mi_not_50['loc'].mean():.1f}")
print(f"  平均CC: {mi_not_50['cc'].mean():.1f}")

