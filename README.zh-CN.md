# Wall-X：LIBERO、RoboTwin 2.0 与 ReBot

[English](README.md) · [🎬 视频展厅](https://juxingyiwan.github.io/wall-x/showcase.html) · [上游 Wall-X](https://github.com/X-Square-Robot/wall-x) · [贡献指南](CONTRIBUTING.zh-CN.md) · [全部 CI 检查](https://github.com/juxingyiwan/wall-x/actions)

[![Core CPU regressions](https://github.com/juxingyiwan/wall-x/actions/workflows/core-tests.yml/badge.svg?branch=main)](https://github.com/juxingyiwan/wall-x/actions/workflows/core-tests.yml) [![B601 deployment adapter](https://github.com/juxingyiwan/wall-x/actions/workflows/deployment.yml/badge.svg?branch=main)](https://github.com/juxingyiwan/wall-x/actions/workflows/deployment.yml)

**训练 Wall-X 策略，在仿真中评估，并部署到 ReBot / B601 真机。**

基于 [X-Square Robot / Wall-X](https://github.com/X-Square-Robot/wall-x)，将 LIBERO、RoboTwin 2.0 和 ReBot 整合到同一代码库，每条使用路线配有独立环境和完整指南。

- **训练与评估**：演示数据准备、checkpoint 恢复和逐任务评估报告。
- **真机部署**：本地或网络推理、RTC（实时动作分块）和 Rerun 可视化。
- **动作对齐**：共享的训练/推理 padding 对齐，以及真机对比和离线消融。

[开始使用](#workflows) · [真机演示](#rebot-demos) · [实验结果](#experiments) · [技术开发说明](#development)

<a id="rebot-results"></a>

## ReBot：Wall-OSS-0.5 与 RTC

Wall-OSS-0.5 真机对比：**左侧开启 RTC（实时动作分块），右侧未开启 RTC**。下方无声动图按原速展示完整对比，点击即可打开原分辨率视频播放器。

[![Wall-OSS-0.5 真机对比：左侧开启 RTC，右侧未开启 RTC](assets/rebot_walloss05_rtc_comparison.gif)](https://juxingyiwan.github.io/wall-x/#rtc)

[🎬 播放 RTC 对比 · 19.9 秒](https://juxingyiwan.github.io/wall-x/#rtc) · [运行步骤](docs/rebot.zh-CN.md#operate) · [Action padding 技术说明与消融](#action-padding)

<a id="rebot-demos"></a>

## ReBot：任务演示

下方无声动图按原速展示每段完整演示。点击预览图可播放原分辨率视频，支持声音和全屏控制。各标题均为传给策略的完整任务指令。

[🎬 视频展厅](https://juxingyiwan.github.io/wall-x/showcase.html)

| `Grasp the cube to the bowl` | `Grasp the pen to the box` | `Grasp the cube into the closed top drawer` |
| --- | --- | --- |
| [![Grasp the cube to the bowl](assets/rebot_grasp_the_cube_to_the_bowl.gif)](https://juxingyiwan.github.io/wall-x/showcase.html#cube) | [![Grasp the pen to the box](assets/rebot_grasp_the_pen_to_the_box.gif)](https://juxingyiwan.github.io/wall-x/showcase.html#pen) | [![Grasp the cube into the closed top drawer](assets/rebot_grasp_the_cube_into_the_closed_top_drawer.gif)](https://juxingyiwan.github.io/wall-x/showcase.html#drawer) |
| [🎬 播放视频 · 12.0 秒](https://juxingyiwan.github.io/wall-x/showcase.html#cube) | [🎬 播放视频 · 11.4 秒](https://juxingyiwan.github.io/wall-x/showcase.html#pen) | [🎬 播放视频 · 13.4 秒](https://juxingyiwan.github.io/wall-x/showcase.html#drawer) |

<a id="workflows"></a>

## 选择一条使用路线

克隆仓库，并记录输出的提交号作为本次运行的代码版本。先按所选指南配置环境与资源，再运行首个任务。

```bash
git clone https://github.com/juxingyiwan/wall-x.git
cd wall-x
git rev-parse HEAD
```

| 使用目标 | 首个任务 | 完整指南 |
| --- | --- | --- |
| LIBERO 仿真评估或训练 | 已有训练 checkpoint 时先跑[单回合评估](docs/libero.zh-CN.md#evaluate)；训练新模型则从[演示数据与训练](docs/libero.zh-CN.md#data)开始 | [English](docs/libero.md) · [中文](docs/libero.zh-CN.md) |
| RoboTwin 2.0 仿真评估或训练 | 从[双任务数据示例](docs/robotwin.zh-CN.md#data)开始；已有训练 checkpoint 时先跑[评估预检](docs/robotwin.zh-CN.md#evaluate) | [English](docs/robotwin.md) · [中文](docs/robotwin.zh-CN.md) |
| ReBot / B601 真机运行 | 准备[机器人与 checkpoint 配置](docs/rebot.zh-CN.md#configuration)，先用[录制观测回放](docs/rebot.zh-CN.md#replay)验证，再进入[真机操作](docs/rebot.zh-CN.md#operate) | [English](docs/rebot.md) · [中文](docs/rebot.zh-CN.md) |

开始前按下表准备资源；下载与训练命令均在对应指南中。

| 资源 | 获取方式 |
| --- | --- |
| Wall-OSS-0.5 基础权重与 Qwen processor | 按 [LIBERO](docs/libero.zh-CN.md#assets)、[RoboTwin](docs/robotwin.zh-CN.md#assets) 或 [ReBot](docs/rebot.zh-CN.md) 指南下载固定版本 |
| LIBERO / RoboTwin 演示数据 | 下载指南中的公开数据；RoboTwin 全量路线另含独立开发集采集步骤 |
| ReBot 演示数据与回放观测 | 自行采集，按 [ReBot 指南](docs/rebot.zh-CN.md#data) 转换 |
| 三条路线的任务 checkpoint | 按指南训练生成，保留配置与归一化参数；本仓库尚未发布实验所用的任务权重 |

模型训练与推理使用 Linux、NVIDIA GPU、CUDA 和 Conda。各场景使用独立环境；RoboTwin 另设模拟器环境，需要 NVIDIA/Vulkan 图形支持和模拟器资源。

ReBot 需要 B601 机械臂、两路相机和兼容的训练 checkpoint。GPU 可位于机器人本机或独立模型服务器。指南命令均从仓库根目录执行，并将占位路径替换为实际路径。

<a id="experiments"></a>

## 实验结果

指南提供固定版本的基础模型与数据下载、训练配置和评估命令。评估使用训练产出的任务 checkpoint，并读取随附配置与归一化统计量。

复现入口：[LIBERO 资源](docs/libero.zh-CN.md#assets) · [LIBERO 训练与数据划分](docs/libero.zh-CN.md#train) · [LIBERO 评估](docs/libero.zh-CN.md#evaluate) · [RoboTwin 资源](docs/robotwin.zh-CN.md#assets) · [RoboTwin 五任务协议与种子](docs/robotwin.zh-CN.md#reported-settings)。

<a id="libero-results"></a>

### LIBERO

两种初始化均使用 Wall-X 动作模型结构并进行全参数训练：有效 batch 128、seed 10222、30,000 步余弦调度，在 25,000 次更新时评估。每个套件含 10 个任务、每任务 50 次，每种初始化共 2,000 次。Qwen 初始化加载预训练视觉语言骨干，动作专家重新初始化。具体见[训练与评估设置](docs/libero.zh-CN.md#train)。

| 套件 | Wall-OSS-0.5 初始化 | Qwen2.5-VL-3B 初始化 |
| --- | --- | --- |
| Spatial | 98.8% (494/500) | 98.2% (491/500) |
| Object | 99.6% (498/500) | 100.0% (500/500) |
| Goal | 98.4% (492/500) | 98.8% (494/500) |
| Long | 96.0% (480/500) | 94.6% (473/500) |
| **总体** | **98.2% (1964/2000)** | **97.9% (1958/2000)** |

每种初始化各进行一次独立训练。[训练模板](workspace/example/libero.yml)与[报告实验设置](docs/libero.zh-CN.md#train)说明了两种初始化方式及数据划分。

![Wall-OSS-0.5 与 Qwen2.5-VL-3B 初始化的 LIBERO 训练 flow loss](assets/libero_training_loss.png)

前 25,000 步的训练 flow loss，纵轴采用对数刻度。浅色线为日志区间均值，实线为最近 25 个记录点的滑动均值（约 250 步）。

从结果上来看，经过机器人数据预训练的基模在LIBERO上收敛更快。但是由于二者都进行了充分的训练，最终成功率差不多。

<a id="robotwin-results"></a>

### RoboTwin 2.0

五任务训练每任务使用 50 条 clean + 500 条 randomized 演示，共 2,750 条，有效 batch 256，评估 checkpoint 完成了 30,000 次更新。评估采用 unseen 指令、预测 32 步/执行 16 步，**每任务、每种设置各 100 个专家有效回合**：Clean 500 次，Randomized 500 次。具体场景与种子见[评估设置](docs/robotwin.zh-CN.md#reported-settings)。成功率按五个任务的有效评估回合计算，策略失败计入分母，专家无效场景在评估前排除。

| 任务 | Clean 成功数/总次数 | Randomized 成功数/总次数 |
| --- | --- | --- |
| `open_microwave` | 67/100 | 85/100 |
| `click_bell` | 71/100 | 92/100 |
| `stack_blocks_three` | 79/100 | 78/100 |
| `place_shoe` | 97/100 | 96/100 |
| `put_object_cabinet` | 76/100 | 82/100 |
| **五任务平均** | **78.0% (390/500)** | **86.6% (433/500)** |

结果发现，Randomized 的平均成功率（**86.6%**）高于 Clean（**78.0%**），原因可能是：

1. **训练分布偏向 Randomized。** Randomized 占训练帧的 **90.8%**，采样未对两种条件做均衡，模型可能更适应随机化场景。
2. **两组评测场景没有逐局配对。** 两组分别生成并筛选专家有效场景；随机化在物体生成前消耗随机数，因此相同种子下的物体型号、布局和操作难度也可能不同，进而影响两组成功率。

差距主要集中在 `open_microwave`（+18 个百分点）和 `click_bell`（+21 个百分点），二者合计贡献 **7.8 个百分点，约占总差距 8.6 个百分点的 91%**。

<a id="upstream"></a>

## Wall-X 相关文档

以下为官方 `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f` 版本文档，官方首页保存在 `README.upstream.*`。本仓库的三条使用路线见上方导航。

| 原生内容 | 位置 |
| --- | --- |
| 官方介绍、模型、社区与引用 | [English](README.upstream.md) · [中文](README.upstream.zh-CN.md) |
| 原生使用与脚本说明 | [Workspace](workspace/README.md) · [Scripts](scripts/README.md) |
| 原生硬件示例 | [插花](workspace/TRAIN_ARRANGE_3_FLOWERS.md) · [RTX 5090 安装](workspace/rtx5090/README.md) · [部署](workspace/rtx5090/DEPLOY.md) · [原有记录](workspace/rtx5090/local/NOTES.md) |

<a id="contribute"></a>

## 贡献与开发

测试与 PR 准备见[贡献指南](CONTRIBUTING.zh-CN.md)（[English](CONTRIBUTING.md)），实现要点见[技术开发说明](#development)（[English](README.md#development)）。项目采用 [Apache-2.0](LICENSE)；第三方归属见[来源声明](NOTICE.txt)和[客户端声明](deployment/client/NOTICE.txt)。

模型、数据和模拟器遵循各自许可。模型与数据资源（固定版本）：[Wall-OSS-0.5](https://huggingface.co/X-Square-Robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028) · [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3) · [LIBERO](https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4) · [RoboTwin 2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186) · [RoboTwin LeRobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot/tree/595d4acc7e026aa57afba218f4170c6d79e39103)。

<a id="development"></a>

## 技术开发说明

本节说明模块分工、优化分析与归位边界。操作流程见[三条使用路线](#workflows)，检查要求见[贡献指南](CONTRIBUTING.zh-CN.md)。

<a id="ownership"></a>

### 模块分工

| 模块 | 职责 |
| --- | --- |
| `wall_x/model`、`data`、`config`、`trainer`、`_vendor/harrix` | 原生核心；共享 padding、loader 与恢复补丁保留在此 |
| `wall_x/robotwin`、`scripts/*robotwin*` | RoboTwin 转换/训练/评估及隔离的模拟器 worker |
| `wall_x/data/b601*`、`wall_x/trainer/adapters/b601*` | B601 数据准备、表示和恢复 |
| `deployment/`、`deployment/client/` | 模型后端/启动器及可独立打包的硬件控制器 |
| `environments/`、`workspace/example/` | 依赖组合和公开配置模板 |

<a id="implementation-analysis"></a>

### Wall-X 优化与分析

<a id="action-padding"></a>

#### Action padding：训练与推理对齐

这个实验源于 ReBot 部署中观察到的动作抖动，针对的是 [Wall-X 原版 ce2a1ae](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl_act.py#L2080-L2092) 中已存在的训练与推理不一致。B601 的 7 个真实动作通道（6 个机械臂关节和 1 个夹爪）需要补齐到模型的 26 维。训练时先归一化真实通道，再补零；原版推理在[checkpoint 统计量已覆盖全部 26 维](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/wall_x/model/core/action/normalizer.py#L98-L109)、虚拟通道为 `min=0, delta=1` 时，会把原始零值归一化为 -1。虚拟通道虽然不参与损失计算，仍进入共享的动作编码器，可能影响真实关节输出。消融比较将虚拟动作终点恢复为训练时的 0 前后，预测动作平滑性的变化。

<a id="padding-video"></a>

##### 真机效果对比

Wall-OSS-0.5 使用 100 条演示、训练 12 个 epoch，执行“开抽屉 → 方块放入抽屉”。两组均开启 RTC，仅切换 action padding 对齐：左侧已对齐，右侧未对齐。

[![Action padding 真机对比：左侧已对齐，右侧未对齐，两组均开启 RTC](assets/rebot_walloss05_padding_comparison.gif)](https://juxingyiwan.github.io/wall-x/#padding)

[🎬 播放 Padding 对比 · 16.0 秒](https://juxingyiwan.github.io/wall-x/#padding)

<a id="padding-offline"></a>

##### 离线预测消融

使用 6 份 checkpoint 回放真机历史观测，每组配对保持权重、观测和噪声相同，仅将虚拟动作终点从 -1 改为 0；state/mask 固定，RTC、平滑和裁剪关闭。共 900 组配对预测，零终点在全部 900 组中降低了该指标。

对每组预测，使用前 $K = 16$ 步、$D = 6$ 个机械臂关节的平均绝对二阶差分衡量平滑性：

$$
J = \frac{1}{D(K-2)}
    \sum_{j=1}^{D}\sum_{k=0}^{K-3}
    \left|q_{k+2,j} - 2q_{k+1,j} + q_{k,j}\right|.
$$

其中，$q_{k,j}$ 为第 $k$ 步、第 $j$ 个关节的预测角度，单位为度。每个关节有 14 个二阶差分，共对 84 项取平均。指标单位为度/采样步²，越小越平滑。下表按配对数汇总相关 checkpoint。

| 任务 | 配对预测数 | 原生padding | 对齐后的padding | 降幅 |
| --- | --- | --- | --- | --- |
| Bowl | 360 | 2.958 | 0.262 | 91.1% |
| Drawer | 270 | 2.334 | 0.206 | 91.2% |
| Pen | 270 | 2.557 | 0.263 | 89.7% |

这些 checkpoint 使用绝对关节动作，当前 B601 模板使用增量动作。相关设置见 [ReBot 配置](docs/rebot.zh-CN.md#data)。

<a id="homing"></a>

### 归位边界

以下常量控制名义归位扫掠与夹爪零位检查。操作前检查完整机械臂路径中的障碍物。数值与源码保持一致，使用假设备测试反馈丢失和超时路径：

| 源码常量 | 值 | 含义 |
| --- | --- | --- |
| `front_home.FORWARD_LIMIT_M` | `0.5` | 前向平面（米） |
| `front_home.CLEARANCE_M` | `0.03` | 名义裕量（米） |
| `zero_servo.ZERO_PRECISION_DEG` | `0.5` | 夹爪零位误差上限（度） |
| `zero_servo.MAX_REFINEMENT_S` | `10.0` | 细化时限（秒） |

<a id="source"></a>

### 实现依据

变更行为时对照 [padding](wall_x/inference_padding.py)、[权重选择](wall_x/checkpoint_weights.py)、[客户端配置](deployment/client/src/wallx_client/config.py)、[线协议](deployment/client/src/wallx_client/protocol.py)和[示例 manifest](deployment/client/docs/examples/server_manifest.json)。通过 `--help` 查询实际参数解析器，保留 [NOTICE](NOTICE.txt)及[客户端声明](deployment/client/NOTICE.txt)中的第三方归属。
