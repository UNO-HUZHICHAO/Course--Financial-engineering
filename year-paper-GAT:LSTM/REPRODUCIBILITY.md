# 复现说明

本仓库是正式展示版，保留核心代码和关键结果，但不包含完整大规模数据。因此，本仓库不保证开箱即用的一键复现，需要先准备本地数据环境。

## 推荐步骤

1. 创建 Python 环境；
2. 安装依赖：

```bash
cd src/gl_v4
pip install -r requirements.txt
```

3. 根据 `DATA.md` 准备数据；
4. 检查代码中的数据路径；
5. 运行环境测试：

```bash
python smoke_test.py
```

6. 根据需要运行完整流程：

```bash
python run_on_autodl.py
# 或
bash run_full_autodl.sh
```

## 结果对照

可将复现结果与以下目录中的文件进行对照：

```text
results/final_backtest/
```

## 注意

由于不同数据版本、时间区间、缺失值处理和随机种子设置可能导致结果差异，复现时应以论文中说明的实验设定为准。
