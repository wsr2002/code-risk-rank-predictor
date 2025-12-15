#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试复杂度提取"""

import sys
import subprocess
import re
from pathlib import Path

def test_radon():
    """测试radon是否能正常工作"""
    test_file = Path("D:/testsprite/python_projects/airflow/airflow-core/src/airflow/api_fastapi/core_api/datamodels/connections.py")
    
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return
    
    print(f"测试文件: {test_file}")
    
    # 方法1: 尝试使用 python -m radon
    try:
        print("\n尝试使用 python -m radon...")
        cc_output = subprocess.check_output(
            [sys.executable, "-m", "radon", "cc", "-a", "-s", str(test_file)],
            stderr=subprocess.PIPE,
            timeout=10
        ).decode("utf-8", errors="ignore")
        
        print("Radon输出:")
        print(cc_output[:500])
        
        # 解析
        cc_values = []
        for line in cc_output.splitlines():
            if "Average complexity:" in line:
                match = re.search(r'Average complexity:\s*([\d.]+)', line)
                if match:
                    cc_values.append(float(match.group(1)))
                    print(f"找到平均复杂度: {match.group(1)}")
        
        if cc_values:
            print(f"\n成功提取复杂度: {cc_values}")
        else:
            print("\n未找到平均复杂度，尝试解析单个函数...")
            for line in cc_output.splitlines():
                match = re.search(r'\([A-F]\)\s+(\d+)', line)
                if match:
                    cc_values.append(int(match.group(1)))
            if cc_values:
                print(f"找到函数复杂度: {cc_values}, 平均值: {sum(cc_values)/len(cc_values):.2f}")
            else:
                print("未找到任何复杂度数据")
                
    except subprocess.CalledProcessError as e:
        print(f"Radon调用失败: {e}")
        print(f"错误输出: {e.stderr.decode('utf-8', errors='ignore')[:200]}")
    except FileNotFoundError:
        print("Radon未安装，需要使用简单估算方法")
    except Exception as e:
        print(f"其他错误: {e}")

if __name__ == "__main__":
    test_radon()

