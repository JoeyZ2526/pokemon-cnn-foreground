# 基于 CNN 的多标签图像分类与前景信息利用

从一张宝可梦图像预测其一种或两种类型，输出 18 个标签的概率。本项目围绕一个具体问题开展对照实验：**空间信息应该如何汇总，图像自带的前景信息能否帮助分类？**

项目涵盖数据处理、模型实现、对照实验与结果分析。基于 809 张图像和 18 类属性，从零训练小型 CNN，设计并实现基于 PNG 透明通道的前景加权池化，并与平均池化、最大池化、残差结构及全局与局部信息融合进行比较。

在 122 张固定历史留出图像上，训练与推理均使用前景加权池化的 G3 模型，mAP、Macro-F1 和 Micro-F1 均高于平均池化、最大池化和残差 GAP 三个基线。相对普通 GAP 和 GMP，G3 的三项五折均值均提高，其中五折平均 Micro-F1 为七种实验条件中最高。

## 1. 从这里开始

唯一代码入口是 [Pokemon_CNN_Comparison.ipynb](Pokemon_CNN_Comparison.ipynb)。在仓库根目录打开 Notebook，选择 Python 环境，修改配置 Cell 中的 `MODE` 后从头执行。

| 模式 | 用途 | 是否需要原始数据 | 是否更新模型参数 |
| --- | --- | --- | --- |
| `MODE = "results"` | 默认模式，读取仓库中的历史结果表 | 否 | 否 |
| `MODE = "check"` | 检查真实数据、同步变换及小样本计算链路 | 是 | 否 |
| `MODE = "train"` | 从头运行或恢复完整实验，保存本次运行结果 | 是 | 是 |

公开版保留原实验的算法与选择规则，并完成代码等价性和小样本计算链路检查。下方成绩表对应已完成的统一协议实验；默认使用 `results` 模式，可直接查看随仓库提供的结果表。

```text
pokemon-cnn-foreground/
├── Pokemon_CNN_Comparison.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── experiment.json       # 算法、训练和评价约定
│   ├── splits.json           # 固定开发集、测试集及五折划分
│   ├── data_manifest.json    # 数据版本与校验信息
│   └── reference.json        # 历史结果的来源与解释
├── results/
│   └── reference_results/   # 已完成实验的轻量结果表
├── data/                    # 自行下载的数据，不提交
└── outputs/                 # 本次运行的日志与检查点，不提交
```

## 2. 环境与安装

实验运行环境为 Python 3.11.15、PyTorch 2.5.0、torchvision 0.20.0、CUDA 12.4，显卡为 NVIDIA GeForce RTX 4060 Laptop GPU。主要 Python 依赖已在 [requirements.txt](requirements.txt) 固定版本，便于配置对应实验环境。

`pandas==3.0.2` 要求 Python ≥ 3.11，与本项目 Python 3.11 环境兼容，保留实际使用版本。[pandas 3.0.2 包元数据](https://pypi.org/project/pandas/3.0.2/)

Windows PowerShell 示例，在仓库根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

上面的 CUDA 12.4 版本组合来自 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v250)。仅使用 CPU 时，将该条 PyTorch 安装命令中的 `cu124` 改为 `cpu`；查看结果和数据检查可以使用 CPU。本版完整训练入口要求使用 CUDA，以保留原实验的运行条件。

已有 Jupyter 或 VS Code Notebook 环境时，选择上述 `.venv` 的 Python 解释器即可。需要单独安装 Notebook 界面时：

```powershell
.\.venv\Scripts\python.exe -m pip install notebook ipykernel
.\.venv\Scripts\python.exe -m notebook
```

Jupyter 是交互运行工具，不属于模型的实验参数。

## 3. 数据准备

原项目记录的数据来源是 [Kaggle：Pokemon Image Dataset，vishalsubbiah](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)。请自行下载，并整理为：

```text
data/
├── pokemon.csv
└── images/
    ├── bulbasaur.png
    ├── ivysaur.png
    └── ...
```

CSV 使用 `Name`、`Type1`、`Type2` 构建目标；`Type2` 可以为空。本地原始表另有 `Evolution` 列，但它不参与模型输入或标签构建。图像名与 `Name` 对应。

实际实验使用 809 张图像，其中 404 张为单标签、405 张为双标签，共 18 种类型。预测目标为图像中的类型属性。前景方法使用 PNG 原始 alpha 透明度通道，数据准备时保留 RGBA 格式。

`DATA_DIR` 默认是仓库内 `data/`，`OUTPUT_DIR` 默认是 `outputs/`。也可以在启动 Notebook 前设置环境变量，例如：

```powershell
$env:POKEMON_DATA_DIR = "D:\datasets\pokemon"
$env:POKEMON_OUTPUT_DIR = "D:\experiments\pokemon-cnn"
```

数据版本与固定划分需匹配 `config/` 中的记录。校验使用 `Name/Type1/Type2` 三列规范化后的内容与行序，以及每张图片的文件校验值；CSV 的编码、换行或无用 `Evolution` 列不同可以接受，名称、标签、行序或图像内容不同则停止执行。原始 CSV 的字节校验值仅用于记录历史来源，新运行另行记录实际文件校验值。

运行前通过数据校验确认下载版本、样本顺序及固定划分与实验记录一致；发现差异时，可依据 `data_manifest.json` 定位。图像与标签从上述数据源获取，使用与再分发遵循数据来源的相关规定。

## 4. 比较了哪些方法

| 名称 | 改动 | 使用 alpha | 是否训练 |
| --- | --- | --- | --- |
| `Plain_GAP`，亦称 G1/C | 普通 CNN，全局平均池化 | 否 | 是 |
| `Residual_GAP` | 残差 CNN，全局平均池化 | 否 | 是 |
| `Plain_GMP` | 普通 CNN，全局最大池化 | 否 | 是 |
| `A_Shared` | 全局平均与 Top-16 局部平均，由一个共享系数融合 | 否 | 是 |
| `B_Classwise` | 同上，18 个类别各有一个融合系数 | 否 | 是 |
| `G2_Foreground` | 冻结该次 GAP 权重，只把平均池化替换为前景加权平均 | 是 | 否 |
| `G3_Foreground` | 使用前景加权平均，从相同初始参数重新训练 | 是 | 是 |

实验包含六种训练模型和一种冻结推理条件，共七种条件；G1/C 均指 `Plain_GAP`。完整运行包含六种训练模型各五折验证及一次 687 张全开发集重训，即 **30 条交叉验证轨迹和 6 条全开发集轨迹**；G2 复用 GAP 权重完成验证与推理。

前景加权平均使用图像已有的透明度信息：RGB 与 alpha 同步缩放、翻转、旋转，再将 alpha 平均下采样到特征图大小。对每个特征通道，按以下方式汇总：

$$
h_j = \frac{\sum_p m_p F_{j,p}}{\sum_p m_p}.
$$

其中，`F` 是 CNN 特征，`j` 为通道，`p` 为空间位置，`m` 是对应的前景权重，`h` 为汇总后的特征。它减少透明背景位置对平均值的贡献，不新增可训练参数；mask 不参与梯度更新，CNN 与分类头仍通过原损失训练。前提是前景权重之和大于零，并且输入确实提供有效的 alpha。

G2/G3 的前景权重直接来自 PNG 自带的 alpha 通道，保持固定，通过特征汇总环节参与分类。A/B 则从类别得分图中选取得分最高的 16 个空间位置，研究局部高响应信息与全局信息的融合。

方法与代码的对应关系如下；Cell 从 Notebook 第一个单元开始计数。

| 步骤或公式 | 代码中的量 | Notebook 位置与函数 |
| --- | --- | --- |
| RGB 与 alpha 共用几何变换 | RGB 图像、alpha 掩码 | Cell 6，`paired_transform`、`PairedForegroundDataset` |
| 全局均值与 Top-16 均值融合 | 类别得分图、`fusion_logits`，其 sigmoid 为局部项权重 | Cell 8，`SharedFusion.forward`、`ClasswiseFusion.forward` |
| 前景加权平均：Σ(mF)/Σm | 特征 `features`、掩码 `mask` | Cell 8，`ForegroundPool.forward`、`ForegroundModel.forward` |
| 按当前训练成员统计正类权重 | `train_ids`、`pos_weight` | Cell 6，`compute_pos_weight`；Cell 14，`run_stage` |
| 验证选择与最终评价 | `best_record`、`threshold`、`selection_plan_id` | Cell 10，`find_best_threshold`、`choose_epoch`；Cell 14，`freeze_selections`；Cell 16，`finalize_results` |

## 5. 实验协议

| 项目 | 固定规则 |
| --- | --- |
| 外层数据 | 687 张开发集、122 张历史留出测试集，保留固定索引 |
| 开发集验证 | 五折；前两折 549/138，后三折 550/137；各折训练不使用该折验证样本 |
| 预处理 | RGB 转换 → 128×128 双线性缩放；训练另加随机水平翻转、±15°最近邻旋转，`expand=False`；再标准化 |
| alpha 变换 | 与 RGB 使用同一次翻转和角度；不裁框、不重新合成背景、不二值化 |
| 训练预算 | 每折 100 轮，Adam，学习率 0.001，batch size 32；无早停、无学习率调度 |
| 随机控制 | 五折 seed 为 42—46，全开发集 seed 为 142；分开管理训练、数据加载和几何增强随机流 |
| 初始化 | 普通 CNN 各条件共享可对应的初始参数；残差结构按相同 seed 完成对应网络的初始化 |
| 全开发集重训 | 五折最佳轮数、分类阈值分别取中位数；在 687 张上从零训练到所定轮数 |
| G2 继承 | 各折继承该次 GAP 的最佳模型，全开发集继承该次 GAP 重训终点；只在验证集重新选阈值 |

本次结果中，G2 继承的 GAP 模型终点为第 87 轮。重新运行时，G2 自动继承该次 GAP 选出的轮数，并将自身五折验证阈值的中位数用于最终预测。

四类量各司其职：

1. **训练损失**：加权二元交叉熵。正类权重仅用当前训练成员统计，全开发集重训时重新计算；损失通过梯度更新模型参数。
2. **模型选择**：每折按验证集 Macro-F1 选轮数；在 `1e-12` 容差内并列时，选验证损失较低者，再并列选较早轮数。mAP 不用于本协议选轮数。
3. **分类阈值**：在验证集考察 0.30—0.70、间隔 0.05 的九个阈值；按 Macro-F1、Micro-F1 依次比较，并列保留先出现的较低阈值。阈值用于将预测概率转换为类别判断。
4. **最终评价**：冻结各模型与阈值后，报告历史测试集 mAP、Macro-F1、Micro-F1 和逐类结果。mAP 使用连续分数，与分类阈值无关；不使用测试集挑参数。

## 6. 已完成的历史结果

以下来自 `results/reference_results/`，均为本机统一协议下的实际记录。数字范围为 0—1；表内是测试结果，五折均值与逐折记录另见 CSV。

| 方法 | mAP | Macro-F1 | Micro-F1 |
| --- | ---: | ---: | ---: |
| Plain GAP（G1/C） | 0.279535 | 0.163489 | 0.176143 |
| Residual GAP | 0.283887 | 0.259124 | 0.273063 |
| Plain GMP | 0.282774 | 0.263449 | 0.258786 |
| A：共享融合 | 0.287019 | 0.273737 | 0.272152 |
| B：逐类融合 | 0.283280 | 0.282106 | 0.265271 |
| G2：冻结模型前景汇总 | 0.297962 | 0.299510 | 0.274151 |
| G3：前景汇总重新训练 | 0.288235 | 0.276636 | 0.273859 |

主要结果：

- **前景加权模型超过三个基线。** 在上述 122 张固定历史留出图像上，G3 的 mAP、Macro-F1、Micro-F1 均高于 Plain GAP、Residual GAP 和 Plain GMP；相对 GMP 分别提高 **0.55、1.32、1.51 个百分点**。
- **五折验证支持相对普通池化的改善。** G3 相对 Plain GAP 和 Plain GMP 的三项五折均值均提高，Micro-F1 达到 **0.336731**，为七种实验条件中最高。完整五折均值与标准差保存在 `cv_aggregate.csv`。
- **实现了两种前景信息利用方式。** G2 在冻结已有 GAP 权重的条件下替换汇总方式；G3 将前景加权池化纳入训练与推理。两种实现分别考察推理阶段的直接应用和训练过程中的适配。
- **完成全局与局部融合对照。** A 使用共享融合系数，B 使用逐类融合系数；B 在六种训练模型中的留出 Macro-F1 最高，为 **0.282106**。

本节结果由统一协议下的训练产物汇总，并保留逐折指标、预测统计和配置记录。公开版提供统一入口，可独立运行全部实验条件。

## 7. 结果文件与复核

| 文件 | 内容 |
| --- | --- |
| `all_schemes_summary.csv` | 七组主要结果、参数量、轮数和阈值 |
| `cv_fold_results.csv` | 七组共 35 条逐折结果，G2 为冻结模型评价 |
| `cv_aggregate.csv` | 五折均值与标准差 |
| `all_class_test_AP.csv` | 18 类逐类 AP 与测试正例数 |
| `test_confusion_counts.csv` | 各组 TP、FP、FN 与平均预测标签数 |
| `planned_contrasts.csv` | 预定组间比较 |
| `all_pairwise_deltas.csv` | 完整组间指标差值 |
| `fusion_weights.csv` | A/B 的融合系数 |
| `G2_fixed_source_threshold_diagnostics.csv` | G2 沿用源阈值的诊断结果，不替换主结果 |

TP、FP、FN 按“图片 × 标签”统计。本次留出评价包含 122 × 18 = 2,196 个判断，其中真实正标签为 181 个。主结果使用各组由验证集确定的阈值；诊断表用于补充分析阈值对预测结果的影响。

运行 `check` 确认数据与链路后，再按需要选择 `train`。新的日志、检查点和汇总写入 `OUTPUT_DIR`，与仓库内的历史结果分开；恢复实验时保持数据、协议和输出目录一致。
