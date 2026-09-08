# BioAlign-EMG Mechanism Validation V2.1-clean

这是用于真实 SeNic 数据机制验证的干净代码版。它保留当前实验所需代码，删除旧版 `legacy_v1`、`__pycache__`、`.pytest_cache`、smoke 测试输出、checkpoint、预测文件和其他生成结果。

## 现在要验证什么

这版代码不再只证明“分类变好”，而是专门验证：

1. 真实电极角度校正是否有上限收益；
2. 模型是否能恢复人工施加的旋转；
3. 模型预测的旋转是否和 SeNic 实测角度有关；
4. 对齐后同一手势跨位置是否更接近；
5. soft probability weighting 是否破坏手势空间模式；
6. BioAlign 是否优于 No Alignment、Uniform、Random、Hard-STE、传统模型和 Oracle / Wrong-direction 对照。

## 目录关系

真实数据仍放在旧项目：

```text
D:\BioSelect_EMG\data\raw\SeNic\subjects
```

本代码建议放在：

```text
D:\BioAlign_EMG_MechanismValidation_v2.1
```

## 安装依赖

```bat
python -m pip install -r requirements_v2.txt
```

## 数据审计

```bat
python -m mechanism_validation.audit_senic_dataset --project-root D:\BioSelect_EMG --subjects h0-h29
```

正常时每个受试者应为 231 个 session-0 trial、11 个位置、3 次重复、7 个手势、1 个角度文件，状态为 `OK`。

## 先跑 h0 快速实验

```bat
run_h0_quick_v21.bat
```

主要输出：

```text
D:\BioSelect_EMG\data\processed_v21\h0.npz
D:\BioSelect_EMG\data\processed_v21\angle_audit.csv
D:\BioSelect_EMG\results_v21\h0_quick_5epoch\metrics_all.csv
D:\BioSelect_EMG\results_v21\h0_quick_5epoch\mechanism_continuous\mechanism_report.json
```

先看 `metrics_all.csv` 中的 `accuracy`、`balanced_accuracy`、`macro_f1`；再看 `mechanism_report.json` 中的 synthetic shift recovery、real angle agreement、latent disentanglement、pre/post pattern preservation。

## 跑 30 人正式实验

h0 快速实验确认逻辑正常后再运行：

```bat
run_all_v21.bat
```

输出目录：

```text
D:\BioSelect_EMG\results_v21\full_30subjects
```

关键汇总文件：

```text
all_subject_metrics_long.csv
summary_shift_macro_f1.csv
paired_shift_macro_f1.csv
summary_ideal_macro_f1.csv
paired_ideal_macro_f1.csv
```

## 结果解释底线

只有当下面几件事同时成立，论文才可以写“模型在补偿/修复电极旋转”：

1. Continuous/Soft 在 shift 条件下明显优于 TCN、RingAug、No Alignment、Uniform、Random；
2. Oracle continuous 或 Oracle integer 明显高于普通模型；
3. Wrong-direction correction 明显变差；
4. synthetic shift recovery 能恢复人工施加的旋转；
5. predicted angle 和 real measured angle 有可解释关系；
6. 对齐后同一手势跨位置更接近，同时不同手势没有被揉成一团。

如果只有分类提高，但角度恢复和模式分析不成立，论文表述必须降级为“样本自适应循环特征混合”，不能声称强旋转修复或强仿生重映射。
