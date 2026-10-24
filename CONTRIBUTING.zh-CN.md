# 参与 Wall-X 贡献

[English](CONTRIBUTING.md) · [总导航](README.zh-CN.md)

遵循[官方贡献原则](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/CONTRIBUTING.md)：改动聚焦、提供复现步骤、同步更新文档，不提交私有路径/凭证或大型 checkpoint。使用独立开发环境。

<a id="checks"></a>

## 1. 运行相关检查

从仓库根目录使用 Python 3.10 或 3.12，建立与 CI 对应的核心 CPU 回归环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'pip>=25,<27'
python -m pip install --no-deps torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tests/requirements-cpu.txt
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD:$PWD/deployment/client/src" python -m pytest tests -q
```

部署/客户端变更在独立环境安装其开发依赖后运行：

```bash
python -m pip install -e 'deployment/client[dev,vision]'
python -m pytest -c deployment/pyproject.toml deployment/tests -q
python -m pytest -c deployment/client/pyproject.toml deployment/client/tests -q
python scripts/check_docs.py
python scripts/check_robotwin_docs.py
python -m pip install pre-commit
pre-commit run --all-files
```

可执行检查定义以 `.github/workflows/core-tests.yml`、`deployment.yml`、`lint.yml` 为准，包含 Ruff 和独立客户端 wheel 检查。修改打包时需在源码目录外验证 wheel。可选的已分配八卡回归：

```bash
PYTHONPATH="$PWD:$PWD/tests" torchrun --standalone --nproc_per_node=8 tests/b601_training_distributed.py /path/to/new-test-output
```

<a id="validation"></a>

## 2. 按变更选择验证

| 变更 | 额外证据 |
| --- | --- |
| 数据、padding、归一化或模型输入 | 用固定保存观测和真实权重与基线比较，说明哈希/容差 |
| 训练、划分或恢复 | 相同输入/配置下短训练并从 checkpoint 继续 |
| 模拟器 | 固定模拟器/数据/资源及种子的真实 checkpoint rollout |
| 机器人生命周期或硬件 | CPU 故障注入测试、回放/预检，再单独安排现场值守实验 |
| 文档 | 内部链接、双语命令/结构、CLI 解析、shell 语法，不访问设备 |

导入、安装及 CI 不得打开相机/CAN 或驱动电机。保留单控制器所有权、共享锁、单位、过期响应拒绝及停止清理。

<a id="documentation"></a>

## 3. 维护精简文档集合

维护五对中英文文档：根 README、LIBERO/RoboTwin/ReBot 使用指南和本贡献指南。两种语言的章节、命令与参数保持一致。完整使用流程放在场景指南，实现要点放在 [README 末尾的技术开发章节](README.zh-CN.md#development)，验证摘要放在 PR。官方上游文档保持原文。

<a id="pull-requests"></a>

## 4. 准备聚焦的 PR

说明问题、最终行为与兼容性影响，标明候选版本和官方基线，解释共享核心改动，并附验证命令与结果。独立场景的改动应便于分别评审。

模型、模拟器或真机实验应记录代码与资源版本、配置、命令、种子和结果。保留版权头与 `NOTICE.txt`。私有机器配置、凭证、checkpoint、数据集及实验日志放在公共源码树之外；文档选用的图片与演示视频放入 `assets/`，并纳入打包清单。
