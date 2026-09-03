# BioAlign-EMG 完整复现代码包 v1.0

本代码包覆盖论文所需的完整计算流程：SeNic 数据下载与审计、训练集专属归一化、RingAug、TCN/SE/CBAM 基线、BioAlign-EMG、30 名受试者 × 3 个随机种子的主实验、整套重训练消融、物理位移角度分析、统计检验、绘图、TorchScript 导出和 CPU 延迟测试。

## 先说明清楚

这不是用户电脑 `D:\\BioSelect_EMG` 整个目录的字节级备份，而是根据已经确认的项目脚本和论文冻结方案整理出的**干净、可运行、可公开发布的复现包**。数据集和训练权重没有打进压缩包：SeNic 是公开数据集，体积较大；权重需要在本地按论文方案重新训练。

## 论文冻结实验方案

- 数据：SeNic，h0-h29，共 30 名受试者，统一使用 session 0。
- 通道：8；采样率：200 Hz；手势：7 类。
- 训练：仅 p0 的 r0+r1，即每名受试者 14 个参考位置 trial。
- 理想位置测试：p0-r2。
- 电极移位测试：p1-p10 全部 trial。
- 窗口：250 ms（50 点），步长 50 ms（10 点）。
- 随机种子：42、2026、3407。
- 主指标：所有移位位置合并后的 trial-level Macro-F1。
- 最终模型损失：仅手势交叉熵；不使用物理角度监督、辅助 shift loss 或 consistency loss。

## Windows 快速运行

### 1. 解压后安装环境

双击：

```text
run_setup_windows.bat
```

推荐 Python 3.10 或 3.11。

### 2. 下载并整理 SeNic

在 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_senic.ps1
```

完成后的目标结构为：

```text
data/raw/SeNic/subjects/h0/...
data/raw/SeNic/subjects/h1/...
...
data/raw/SeNic/subjects/h29/...
```

### 3. 审计数据

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_dataset.py
```

每名受试者的 session 0 应有 231 个 CSV：11 个位置 × 7 类手势 × 3 次重复。

### 4. 先做快速自检

双击：

```text
run_smoke_test.bat
```

该测试不需要数据，会检查模型前向传播、RingAug、指标函数和参数量。预期参数量：

- TCN：16,663
- SE-TCN：17,293
- CBAM-TCN：17,308
- 论文/原始 checkpoint 兼容版 BioAlign：33,549
- 精简前向路径版 BioAlignCompact：31,299

### 5. 运行论文主实验

双击：

```text
run_main_experiment.bat
```

等价命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_bioalign_final_30subjects_3seeds.py `
  --subjects h0-h29 `
  --seeds 42,2026,3407 `
  --epochs 20 `
  --batch-size 256
```

程序会保存 checkpoint，并在中断后根据已完成结果继续运行。

### 6. 运行正式消融与角度证据包

双击：

```text
run_evidence_pack.bat
```

这会从头训练：

- BioAlign full
- BioAlign -Circular Alignment

并对主实验结果执行 p0-p8 的物理位移 nAUPC 分析；随机移位 p9-p10 不进入单调旋转曲线。

### 7. CPU 延迟测试与 TorchScript 导出

主实验至少完成 h0、seed=2026 后，双击：

```text
run_cpu_benchmark.bat
```

## 目录

```text
bioalign_emg/       核心可复用 Python 包
scripts/            主实验、消融、角度、benchmark 与下载入口
reference/          论文中已报告的汇总数值，仅用于核对
results/            运行后自动生成
figures/            运行后自动生成
checkpoints/        运行后自动生成
logs/               进度日志
```

## 重要的参数量兼容说明

原始最终模型类继承自探索期 `BioSelectEMG`，覆盖了 `forward()`，但两个不参与最终前向计算的 gate 子模块仍然被 PyTorch 注册，因此 `sum(p.numel())` 为 33,549。这个包保留 `BioAlignEMG` 作为**原 checkpoint 兼容实现**，确保状态字典和论文已报告参数量一致；同时提供删除无效 gate 的 `BioAlignEMGCompact`，其实际前向路径参数量为 31,299。

在没有重新跑完全部 30×3 实验前，不应把 Compact 版本的结果当成论文结果。详见 `docs/MODEL_PARAMETER_NOTE.md`。

## 已做的本地质量检查

压缩包生成时已执行：

- Python 全文件编译检查；
- 模型参数量与输出维度检查；
- RingAug 形状检查；
- 一次合成数据反向传播；
- ZIP 完整性检查；
- SHA-256 清单生成。

完整 30 名受试者训练无法在本次打包环境中重跑，因为公开数据集和原 checkpoint 不在当前运行容器中。
