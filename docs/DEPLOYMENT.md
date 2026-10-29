# Cosmos3-Edge Jetson Orin 部署细节

`nvidia/Cosmos3-Edge`（4B 全模态世界动作模型 = 2B VL reasoner + rectified-flow 扩散动作专家，MoT 单塔）
部署到 Jetson AGX Orin（sm_87 / JetPack 6.2 / CUDA 12.6 / TRT 10.7）的完整工程记录。它在 TRTEdge 0.10.1
官方支持矩阵里只属 **Compatible 档**（Official 是 JetPack 7.2 / CUDA 13 / TRT 11），部署代价 = **六道具体的
兼容墙**，逐道打通。

> 本页从 alpamayo-edge 的 Phase W 报告拆分而来。**跨代精度对比**（Cosmos3-Edge 2B vs 上一代 CR2-8B 量化版，
> n=184 驾驶域，「缩小不免费、量化几乎免费」）属 alpamayo-edge「一套管线部署多代模型」主线，见
> [alpamayo-edge](https://ranpin.github.io/alpamayo-edge/cosmos3.html)；**policy 动作链评测**（W3 帧敏感性 /
> W4 conditioning 敏感性）见本仓 [`eval/README.md`](../eval/README.md)；逐问题工程记录（现象 → 根因 → 解决 → 教训）
> 见 [PROBLEMS.md](PROBLEMS.md)。

## 一、六道 JetPack 6.2 兼容墙与打通

| 墙 | 现象 / 根因 | 解法 |
|---|---|---|
| **K1** | CuTe DSL 预编译件缺 sm_87+cuda12 ⇒ configure FATAL（上游 cuda13 targets 含 aarch64:sm_87，cuda12 只有 sm_110/121，**刻意排除**） | 初解 `-DENABLE_CUTE_DSL=OFF`（见 K5 反噬）；根治见 K5 自生成 |
| **K2** | nlohmann 子模块缺失（depth-1 clone）⇒ C++ 编译失败 | `git submodule update --init --recursive` |
| **K3** | XQA JIT NVRTC 缺头 ⇒ llm_build 插件创建失败（cuda12.6 的 `cuda_fp16.h` 依赖链被 CUDA 13 重组掉） | `pluginJitCompiler.cpp` 注入 `-I$CUDA_HOME/include`（NVRTC 从磁盘解析系统头，一处补丁覆盖全部 JIT kernel） |
| **K4** | ViTAttentionPlugin 无 legacy 后端 ⇒ visual_build 失败（枚举只剩 CuTe DSL；`USE_TRT_NATIVE_ATTN=1` 导出的 `TRT_Attention` 要 TRT≥11） | **分解注意力导出补丁**（基础算子块对角 ragged self-attn） |
| **K5** | prefill FMHA 无 legacy 后端 ⇒ 引擎能建、推理必挂（`ContextFMHABackend` 只剩 CuTe DSL） | **自生成** `cutedsl_aarch64_sm_87_cuda12.tar.gz`（`CUDA_CTK_VERSION=12.6 CUTE_DSL_TARGETS=aarch64:sm_87`）→ `-DENABLE_CUTE_DSL=fmha` 重建 → 冒烟全绿 |
| **K6** | policy und_prefill/gen 导出 **ONNX 标准 `Attention`/`RotaryEmbedding`** ⇒ TRT 10.7 parser 不认（und：28 Attention + 84 RotaryEmbedding） | **分解注意力 + RoPE 补丁**（`USE_DECOMPOSED_POLICY=1` 门控，见 §二） |

**三条教训**：① **Compatible 档的代价是具体、有限、可逐个击破的**——六道墙各有工程解，不是「不支持」。
② **「上游不给预编译件」≠「不能生成」**：构建脚本支持显式 target 覆盖，sm_87+cuda12 一次通过（K5）。
③ **分解导出补丁是对「TRT-11-only 算子墙」的通用武器**：K4（visual 的 `TRT_Attention`）与 K6（policy 的
ONNX `Attention`/`RotaryEmbedding`）同源同解；K6 是 policy 链唯一通路（gen 无 FMHA backend 可选）。

## 二、分解注意力 / RoPE 导出补丁

K4（visual）与 K6（policy）同源：TRTEdge 把注意力/RoPE 经 `trt::attention_onnx`/`trt::rope_onnx` custom op
lower 成**只有 TRT≥11 parser 认识**的算子。解法 = 在导出前把它们**用基础 ONNX 算子分解掉**，任何 TRT 版本
可解析，TRT 用 Myelin 自行融合：

- **decomposed_attention**：GQA 折批 grouped matmul + arange 下三角 causal 掩码，覆盖 und 的 is_causal=True
  自注意力与 gen 的 is_causal=False 对 cat([und,gen]) 的交叉注意力（GQA H=16/Hkv=8）。
- **decomposed_rope**：rotate_half；`attention_pos_id` 是恒等 gather（cosmos3PolicyRunner.cpp:257）⇒ 逐 token
  直接对齐，但仍用 `torch.gather` 保留 position_ids 防被 DCE 导致 runtime 找不到 binding。
- **门控与对拍**：env `USE_DECOMPOSED_POLICY=1`，两个 modeling 文件 import 换包装器（additive，不动已工作的
  reasoning 路径）。分解实现先经 `test_policy_decomposed.py` 数值对拍（attention vs
  `F.scaled_dot_product_attention` err~1e-7、rope vs LLaMA 全维公式 err=0）再导出——避免 ~1h 导出后才发现语义 bug。

脚本：`scripts/orin/apply_policy_decomposed_patch.py`（policy）、`scripts/orin/apply_decomposed_attn_patch.py`
（visual）、`scripts/orin/test_policy_decomposed.py`（对拍）。

## 三、动作表示 / 归一化 / state（cosmos-framework 权威定义）

- **表示**：Edge 用 `midtrain` 10 维 = [eef_pos(3) + rot6d(6) + gripper(1)]、**相对位姿**；后处理链
  `denormalize_action` → `pose_rel_to_abs`（需当前 state 做参照）。
- **归一化**：quantile(q01/q99)→[−1,1]，stats 在 `droid_lerobot_stats.json`（dim0-2 位置 delta≈±0.015 米、
  dim3-8 rot6d、dim9 gripper[0,1]）；反归一化 `0.5(a+1)(q99−q01)+q01`。
- **state 注入定论**：Edge 是 **vision+language-only**（gen 图 = 16 行、无 +1 state 行 ⇒ `use_state=False`）。
  TRTEdge PolicyRunner 把 action_latent 全填高斯噪声、mask 全 1、只注入 frame-0 图像 latent + und K/V，无本体
  状态——**设计如此，非 bug**。这正是 policy 单帧「帧无关」边界的机理（见 `eval/README.md`）。

> AV 行车轨迹另走 **inverse-dynamics** 模式（起始帧+视频→ego 轨迹，无需单独 AV policy 权重），但该模式
> **仅 PyTorch 框架路径、非 turnkey**，未 TRTEdge 化（列后续）。

## 四、Policy 动作链部署

Policy-DROID 四组件引擎（und_prefill 2.7G / gen 2.7G / vae_encoder 292M / text_tokenizer 17M，均为分解注意力版
绕过 K6）部署打通，`cosmos3_policy_inference` 图+prompt→action 冒烟通过（action [1,16,10]、5154 ms/round）。
**conditioning 弱耦合边界**（W3 帧敏感性 / W4 五轴对照：seed 主导 52%、帧/prompt/CFG 三轴仅 1–2%，根因未定）
见 [`eval/README.md`](../eval/README.md)。
