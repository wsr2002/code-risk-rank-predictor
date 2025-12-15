# 数据准备脚本使用说明

## 功能

从多个Git项目中提取特征，生成训练/测试数据集。

提取的6个特征变量：
1. **commit_count**: 文件的commit次数
2. **churn**: 文件的代码变更行数（添加+删除）
3. **author_count**: 修改过该文件的作者数量
4. **cc**: 平均圈复杂度（Cyclomatic Complexity）
5. **mi**: 可维护性指数（Maintainability Index）
6. **loc**: 代码行数（Lines of Code）

## 使用方法

### 方法1: 命令行直接指定项目路径

```bash
python prepare_dataset.py D:\testsprite\requests D:\testsprite\another_project
```

### 方法2: 从文件读取项目路径列表

首先编辑 `repos_list.txt`，每行一个项目路径：

```
D:\testsprite\requests
D:\testsprite\another_project
D:\testsprite\yet_another_project
```

然后运行：

```bash
python prepare_dataset.py -f repos_list.txt
```

### 方法3: 指定输出文件名

```bash
python prepare_dataset.py -f repos_list.txt -o my_dataset.csv
```

## 输出格式

生成的CSV文件包含以下列：

- `project`: 项目名称
- `file`: 文件路径（相对于项目根目录）
- `commit_count`: commit次数
- `churn`: 代码变更行数
- `author_count`: 作者数量
- `cc`: 圈复杂度
- `mi`: 可维护性指数
- `loc`: 代码行数

## 注意事项

1. 所有项目必须是Git仓库（包含`.git`目录）
2. 只处理Python文件（`.py`扩展名）
3. 只包含至少有一次commit的文件
4. 需要安装radon工具：`pip install radon`

## 示例输出

```
============================================================
数据准备: 从多个项目提取特征
============================================================

项目数量: 2

处理项目: requests
路径: D:\testsprite\requests
  正在提取Git指标...
  正在分析代码质量...
  正在构建数据集...
  完成: 提取了 287 个文件
  ✓ 成功提取 287 个文件

处理项目: another_project
路径: D:\testsprite\another_project
  正在提取Git指标...
  正在分析代码质量...
  正在构建数据集...
  完成: 提取了 150 个文件
  ✓ 成功提取 150 个文件

============================================================
合并数据集...

数据集统计:
  总文件数: 437
  项目数: 2
  特征列: commit_count, churn, author_count, cc, mi, loc

各项目文件数:
  requests: 287 个文件
  another_project: 150 个文件

✓ 数据集已保存到: D:\testsprite\hotspot_detection\data_prepare\dataset.csv
```

