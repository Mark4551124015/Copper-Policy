<div align="center">

<h1 align="center"><img src="assets/copper.png" width="32" height="32" align="absmiddle" alt="Copper-Policy 标志"> Copper-Policy</h1>

### Focus on the Representation for Robust Robot Manipulation

**Zexin Feng · Yixu Feng · Lingyu Xiao · Shang Su · Kexin Zheng**<br>
**Chang Xu · Mengkai Shi · Shuo Feng · Xintao Yan**

The University of Hong Kong · The University of Sydney · Tsinghua University · DenseAI

[English](README.md) | [中文](README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2609.32779-b31b1b?logo=arxiv)](https://arxiv.org/abs/2609.32779)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb?logo=googlechrome&logoColor=white)](https://zexinfeng-cn.github.io/works/copper-policy/)

**高效训练，强大控制。**

</div>

![Copper-Policy 概览：用于鲁棒操作的紧凑世界表征](assets/teaser.png)

## ✨ 项目简介

Copper-Policy 联合学习紧凑的世界表征与机器人策略。通过任务条件下的未来嵌入预测塑造表征，并保留当前帧的空间细节用于动作执行。

- **紧凑世界建模：** 预测未来嵌入，无需重建像素。
- **高效学习：** 2B 参数模型，在 8 张 RTX 5090 上训练耗时 9.67 小时。
- **鲁棒操作：** 在 LIBERO、LIBERO-Plus、RoboTwin 和三个真实机器人任务上评测。

| LIBERO | LIBERO-Plus | RoboTwin clean | RoboTwin randomized | 真实机器人 |
| :---: | :---: | :---: | :---: | :---: |
| **97.25%** | **80.85%** | **70.84%** | **12.98%** | **96.3%** |

*以上为[论文与项目页面](https://zexinfeng-cn.github.io/works/copper-policy/#main-result)报告的结果。*

![Copper-Policy 模型架构](assets/method.png)

## 📦 开源进度

- [x] 策略推理与基准评测
- [x] 编码器下载与模拟器安装
- [x] 真实机器人策略服务

### TODO

- [ ] **训练代码——即将发布（Training code — coming soon）**
- [ ] 模型 checkpoint
- [ ] 数据集

[安装](#️-安装) · [权重](#-权重) · [评测](#-评测) · [真实机器人](#-真实机器人) · [引用](#-引用)

## 🛠️ 安装

**环境要求：** Linux x86_64、[uv](https://docs.astral.sh/uv/getting-started/installation/)、NVIDIA GPU 与 ffmpeg。`pyproject.toml` 和 `uv.lock` 固定了 Python 3.10.16、PyTorch 2.9.0 / CUDA 13.0 及依赖版本。

```bash
git clone https://github.com/mark4551124015/copper-policy.git
cd copper-policy
uv sync --frozen --all-extras --inexact
uv run --frozen --all-extras --no-sync python -m third_party.setup_sources all
```

评测前请安装下面的模拟器资产。启动器会分别选择 LIBERO 和 LIBERO-Plus 源码，两个目录可以同时放在 `third_party/` 下。

<details>
<summary><b>LIBERO 与 LIBERO-Plus 安装</b></summary>

锁定的 `libero` extra 已包含 Python 依赖；上面的命令会下载源码。系统环境要求请参考 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 与 [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) 安装指南。

LIBERO-Plus 还需要 ImageMagick 共享库（Ubuntu：`sudo apt-get install libmagickwand-dev`）及额外资产：

```bash
uv run --frozen --all-extras --no-sync hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset --local-dir third_party/LIBERO-plus
unzip third_party/LIBERO-plus/assets.zip -d third_party/LIBERO-plus/libero/libero
```

</details>

<details>
<summary><b>RoboTwin 安装</b></summary>

编译 cuRobo 需要 CUDA 13.0 toolkit 和 C++ 编译器。如果 toolkit 不在 `/usr/local/cuda`，请设置 `CUDA_HOME`；`TORCH_CUDA_ARCH_LIST` 应与你的 GPU 匹配。

```bash
# RTX 5090；其他 GPU 请设置对应架构。
TORCH_CUDA_ARCH_LIST='12.0+PTX' uv run --frozen --all-extras --no-sync bash tools/setup_robotwin.sh
uv run --frozen --all-extras --no-sync bash -c \
  'cd third_party/RoboTwin && bash script/_download_assets.sh'
```

安装脚本使用 SAPIEN 3.0.0b1、cuRobo v0.7.8 和 RoboTwin 版本 `c3ddfa8b97d5519efa828b075999bd0006778e5e`，并安装支持 Blackwell 渲染的 OIDN 2.3.3。详见 [RoboTwin 安装指南](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)。

**同步环境时保留 `--inexact`：** cuRobo 是单独编译安装的，精确模式的 `uv sync` 会移除它。更换 Python 或 PyTorch 后请重新运行安装脚本。

</details>

## 🤗 权重

### 预训练编码器

一条命令下载三个编码器预设：

```bash
uv run --frozen --all-extras --no-sync python -m tools.download_weights all --yes
```

已有缓存会直接复用。下载前会显示预计空间需求和磁盘剩余空间。

| 编码器 | 默认预设 | 来源 |
| --- | --- | --- |
| 世界编码器 | `vjepa2_1_vit_large_384` | [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) |
| 视觉编码器 | `dinov2_with_registers_large` | [DINOv2](https://huggingface.co/facebook/dinov2-with-registers-large) |
| 文本编码器与 tokenizer | `wan22_ti2v_5b` | [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |

<details>
<summary><b>手动下载编码器</b></summary>

```bash
WORLD_DIR=pretrained_weights/world_encoder/vjepa2_1_vit_large_384
mkdir -p "$WORLD_DIR"
git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "$WORLD_DIR/source"
curl -fL --retry 3 -o "$WORLD_DIR/vjepa2_1_vitl_dist_vitG_384.pt" \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt

uv run --frozen --all-extras --no-sync hf download facebook/dinov2-with-registers-large \
  config.json model.safetensors \
  --local-dir pretrained_weights/vision_encoder/dinov2_with_registers_large

uv run --frozen --all-extras --no-sync hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --include 'tokenizer/*' 'text_encoder/*' \
  --local-dir pretrained_weights/text_encoder/wan22_ti2v_5b
```

编码器预设与缓存路径定义在 `wan_va/modules/backbone_presets.py`。替换编码器时，需要与策略权重的结构匹配。

</details>

### Copper-Policy 权重

**公开下载链接即将发布。** 推理权重应放在：

| 基准 | 权重路径 |
| --- | --- |
| LIBERO / LIBERO-Plus | `pretrained_weights/copper_policy/libero/policy.pt` |
| RoboTwin clean / randomized | `pretrained_weights/copper_policy/robotwin/policy.pt` |

LIBERO 与 LIBERO-Plus 共用一份权重。导出的权重包含策略参数、必要的模型配置及归一化统计量。推理直接加载训练后的策略，无需额外的 Wan transformer 初始化权重。

## 🚀 评测

### 完整运行四项基准

```bash
uv run --frozen --all-extras --no-sync bash run.sh \
  --out-dir outputs/evaluation/full_v1 \
  --gpu-ids 0,1,2,3,4,5,6,7
```

按 **RoboTwin clean → LIBERO → RoboTwin randomized → LIBERO-Plus** 的顺序运行，默认开启编译和视频保存。

| 基准 | 完整任务范围 | 每任务 / 扰动实例的 episode 数 |
| --- | --- | :---: |
| RoboTwin clean | 50 个任务 | 50 |
| LIBERO | 四个 suite，共 40 个任务 | 50 |
| RoboTwin randomized | 50 个任务 | 50 |
| LIBERO-Plus | 10,030 个扰动实例 | 1 |

中断后续跑同一目录：

```bash
uv run --frozen --all-extras --no-sync bash run.sh \
  --out-dir outputs/evaluation/full_v1 \
  --gpu-ids 0,1,2,3,4,5,6,7 --resume
```

### 单独运行一项基准

```bash
uv run --frozen --all-extras --no-sync bash test_robotwin_clean.sh --all-tasks
uv run --frozen --all-extras --no-sync bash test_libero.sh --all-tasks
uv run --frozen --all-extras --no-sync bash test_robotwin_rand.sh --all-tasks
uv run --frozen --all-extras --no-sync bash test_libero_plus.sh --all-tasks
```

设置 `EVAL_GPU_IDS=0` 可使用单卡；设置 `OUT_DIR` 可复用同一输出目录并跳过已完成的 episode。不加 `--all-tasks` 时，独立脚本使用较小的任务范围：10 个 clean 任务、6 个 randomized 任务、10 个 LIBERO 任务，或 500 个 LIBERO-Plus 实例。

<details>
<summary><b>评测参数与输出</b></summary>

| 参数 | RoboTwin | LIBERO / LIBERO-Plus |
| --- | :---: | :---: |
| 去噪步数 | 10 | 10 |
| Replan 步数 | 24 | 10 |
| 推理 seed | 0 | 7 |
| 编译 | 开启 | 开启 |
| 视频 | 保存 | 保存 |

每张 GPU 常驻一套策略、T5 与 V-JEPA，跨 task 复用。LIBERO 可通过 `MAX_TASKS_PER_GPU` 配置每卡环境客户端数，策略推理 batch size 仍为 1。RoboTwin 保留 expert check，评测接受的场景。

LIBERO 使用明文 prompt 查表映射回训练指令；LIBERO-Plus 的语言扰动不进行映射，未匹配的 prompt 直接使用 benchmark 原句。

输出包含 `summary.json`、逐任务结果、日志及标注 `succ` / `fail` 的视频。LIBERO-Plus 的总成功率按 episode 数加权；`all_tasks_completed` 表示评测是否完整完成。

`--smoke` 减少 episode 数；`--dry-run` 预览命令；独立启动器可用 `--no-compile` 关闭编译。权重、episode 数和推理参数的覆盖方式见 `run.sh --help`。

</details>

## 🤖 真实机器人

使用真实机器人权重启动 TCP 策略服务：

```bash
uv run --frozen --all-extras --no-sync python wan_va/wan_va_server.py \
  --checkpoint /path/to/realrobot.pt \
  --t5-dir pretrained_weights/text_encoder/wan22_ti2v_5b \
  --host 127.0.0.1 --port 8767 --compile-infer-action 0
```

服务使用带长度前缀的 JSON 请求协议。模型预设位于 `inference/`，共享配置位于 `inference/config_base.json`。

## 📚 引用

```bibtex
@misc{feng2026copperpolicyfocusrepresentationrobust,
  title={Copper-Policy: Focus on the Representation for Robust Robot Manipulation},
  author={Zexin Feng and Yixu Feng and Lingyu Xiao and Shang Su and Kexin Zheng and Chang Xu and Mengkai Shi and Shuo Feng and Xintao Yan},
  year={2026},
  eprint={2609.32779},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.32779}
}
```

## 🙏 致谢

本项目基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 与 [LingBot-VA](https://github.com/robbyant/lingbot-va) 的 codebase。感谢这些项目的作者，以及 V-JEPA、DINOv2、Wan、LIBERO、LIBERO-Plus 和 RoboTwin 团队的开源工作。

数据集转换为 LMDB 使用了我们自己的 [lerobot-tools](https://github.com/Mark4551124015/lerobot-tools)。

## 📄 许可证

本项目使用 [Apache 2.0](LICENSE) 许可证。外部模型、模拟器及资产保留各自的许可证。
