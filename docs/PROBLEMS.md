# Cosmos3-Edge 工程问题记录（§K / §L）

> 2026-10-29 从 alpamayo-edge 仓 `docs/PROBLEMS.md`（统一问题记录 A–M）迁入——§K / §L 即
> Cosmos3-Edge 部署与 policy 动作链的工程记录，保留原 K/L 编号以便与历史文档互引。该记录的其余部分——
> §A–J（CR2-8B 量化部署）与 §M（Alpamayo-R1 action 专家，K6 同源墙第三次打通）——留在
> [alpamayo-edge](https://github.com/ranpin/alpamayo-edge/blob/main/docs/PROBLEMS.md)。
> 部署摘要见 [DEPLOYMENT.md](DEPLOYMENT.md)。

---

## K. Cosmos3-Edge on JetPack 6.2：六道墙与打通路径（2026-10-16，W0–W2 实测）

背景：TRTEdge 0.10.x 官方支持 Cosmos3-Edge，但 Orin JetPack 6.2（cuda 12.6 / TRT 10.7 / sm_87）
在支持矩阵里只是 **Compatible 档**（Official 是 JetPack 7.2 / cuda13 / TRT 11）。实测代价 = 六道墙：

### K1. CuTe DSL 预编译件缺 sm_87+cuda12 ⇒ configure FATAL
上游 `kernelSrcs/build_cutedsl_tarballs.sh` 的 cuda13 targets 含 `aarch64:sm_87`，
cuda12 targets 只有 sm_110/121（**刻意排除，非遗漏**）。初解 `-DENABLE_CUTE_DSL=OFF`（见 K5 反噬）。

### K2. nlohmann 子模块缺失（depth-1 clone）
C++ 侧 `nlohmann/json.hpp` 编译失败；python 侧 pyproject `license-files` 匹配不到文件被
scikit-build-core 拒装。修：`git submodule update --init --recursive`；C++ 侧曾以单头文件 `-I` 绕过（等价）。

### K3. XQA JIT NVRTC 缺头 ⇒ llm_build 插件创建失败
0.10.x 的 decode XQA kernel 改为构建期嵌入头 + 运行期 NVRTC JIT；嵌入清单按 CUDA 13 写，
cuda12.6 的 `cuda_fp16.h → vector_types.h → crt/host_defines.h` 链缺失。补清单后又暴露
`vector_functions.h → cuda_runtime_api.h` 巨链 ⇒ **根治：`pluginJitCompiler.cpp` 注入
`-I$CUDA_HOME/include`**（NVRTC 从磁盘解析系统头，一处补丁覆盖全部 JIT kernel）。

### K4. ViTAttentionPlugin 无 legacy 后端 ⇒ visual_build 失败（head_dim 72）
`ViTFMHABackend` 枚举只剩 CuTe DSL 两项（legacy 被移除）；`USE_TRT_NATIVE_ATTN=1` 导出的
`TRT_Attention` 算子 parser 支持 **TRT 11.0 起**（TensorRT CHANGELOG 实证，ragged 11.1）。
解：分解注意力导出补丁（基础算子块对角 ragged attention，`scripts/orin/apply_decomposed_attn_patch.py`）。

### K5. prefill FMHA 无 legacy 后端 ⇒ 引擎能建、推理必挂
`ContextFMHABackend` 枚举同样只剩 CuTe DSL；`ENABLE_CUTE_DSL=OFF` 下 LLM 引擎构建成功
（XQA JIT 属 decode 路径）但运行期 prefill 报 "selected prefill kernel is unavailable (D=128, SM=87)"。
**根治（W1 突破）：用上游脚本自生成 `cutedsl_aarch64_sm_87_cuda12.tar.gz`**——
`CUDA_CTK_VERSION=12.6 CUTE_DSL_TARGETS=aarch64:sm_87 CUTE_DSL_KERNELS=fmha bash kernelSrcs/build_cutedsl_tarballs.sh`
（需 GPU 在场做 AOT 导出；nvidia-cutlass-dsl[cu12]==4.7.0 + cupy-cuda12x 均有 aarch64 wheel）
→ `-DENABLE_CUTE_DSL=fmha` 重配置重建 → **冒烟全绿**（单帧驾驶场景描述连贯、EOS）。

### K6. policy und_prefill/gen 导出 ONNX 标准 Attention/RotaryEmbedding ⇒ TRT 10.7 parser 不认（W2）
`--task policy` 的 und_prefill（理解 prefill）与 gen（扩散动作专家）把注意力/RoPE 经
`trt::attention_onnx`/`trt::rope_onnx` custom op lower 成 **ONNX 标准 `Attention`（opset 23+）
/`RotaryEmbedding` 算子**（und：28 Attention+84 RotaryEmbedding；gen：一批），这两个算子只有
**TRT≥11 parser** 认识；Orin TRT 10.7 建引擎报 "Cannot find plugin: Attention/RotaryEmbedding,
version: 1, namespace:."。与 K4 同源（都是 TRT-11 算子墙），但 K4 是 visual 的 `TRT_Attention`、
这里是 policy 的 ONNX 标准算子；且 K5 的 CuTe DSL 自生成对 policy 图无解（gen 走 custom-op
lowering，不是 FMHA backend 选择）。
**解（W2 突破）**：把 K4 的分解思路推广到 policy + RoPE——`scripts/orin/apply_policy_decomposed_patch.py`
在 `ops.py` 加 `decomposed_attention`（GQA 折批 grouped matmul + arange 下三角 causal 掩码，
覆盖 und 的 is_causal=True 自注意力与 gen 的 is_causal=False 对 cat([und,gen]) 的交叉注意力）
+ `decomposed_rope`（rotate_half；`attention_pos_id` 是恒等 gather（cosmos3PolicyRunner.cpp:257
"identity gather indices"）⇒ 逐 token 直接对齐，但仍用 torch.gather 保留 position_ids 防被 DCE
导致 runtime 找不到 binding），env `USE_DECOMPOSED_POLICY=1` 门控、两个 modeling 文件 import
换包装器（additive，不动已工作的 reasoning 路径）。分解实现先经 `test_policy_decomposed.py`
数值对拍（attention vs `F.scaled_dot_product_attention` err~1e-7、rope vs LLaMA 全维公式 err=0）
再导出 ⇒ und_prefill/gen 引擎建成、`cosmos3_policy_inference` 冒烟 exit=0
（action [1,16,10] 全 finite、5154ms/round）。

**教训**：① Compatible 档的代价是具体、有限、可逐个击破的——六道墙各有工程解；
② **"上游不给预编译件"≠"不能生成"**：构建脚本支持显式 target 覆盖，sm_87+cuda12 生成一次通过；
③ **分解导出补丁是对「TRT-11-only 算子墙」的通用武器**：K4（visual 的 TRT_Attention）与
K6（policy 的 ONNX Attention/RotaryEmbedding）同源同解——用基础算子分解，任何 TRT 版本可解析；
K4 在 K5（自生成 CuTe DSL）解决后成冗余保险，K6 则是 policy 链唯一通路（gen 无 FMHA backend 可选）；
④ 值得向 TRTEdge 上游提 issue：cuda12 targets 补 aarch64:sm_87 + 嵌入头清单补 vector_* 链 +
policy 导出对 TRT<11 提供分解 fallback。

## L. Cosmos3-Edge policy 动作链：部署打通，但 turnkey 路径输出由 seed 主导、conditioning 弱耦合（2026-10-19 W3 / 2026-10-26 W4 实测）

W2 把 policy 四组件引擎建通、冒烟 exit=0（K6）。W3 要把它做成「真实 DROID 数据 → 动作」demo，
先只读侦察 NVIDIA 参考实现（`github.com/NVIDIA/cosmos` + `cosmos-framework`）搞清动作的后处理与
归一化，再用真实数据实测。**W3 初判：turnkey 单帧路径「帧无关」。W4 复测修正：不是「帧无关」，
而是帧/文本/CFG 三个 conditioning 轴一致地只有 1–2%、输出由 diffusion seed 主导（52%）——整条
conditioning 通路弱耦合，根因未定（我们的分解补丁接线 vs 模型属性），见 L4。**

### L1. 动作表示 / 归一化 / 后处理（cosmos-framework 权威定义，已查清）
- **表示**：Edge 用 `midtrain` 10 维 = [eef_pos(3) + rot6d(6) + gripper(1)]、**相对位姿**
  （`action_policy_server_robolab.py:607-612`；action_space 默认 joint_pos=8 维，Edge=midtrain=10 维）。
- **归一化**：quantile(q01/q99)→[−1,1]，stats 在 `cosmos_framework/data/generator/action/normalizer_stats/droid_lerobot_stats.json`
  （10 维物理量：dim0-2 位置 delta≈±0.015 米、dim3-8 rot6d、dim9 gripper[0,1]）；反归一化
  `0.5(a+1)(q99−q01)+q01`（`action_normalization.py`）。
- **后处理链**：`denormalize_action` → `pose_rel_to_abs`（需当前 state 做参照）。
- **state 注入定论**：Edge 是 **vision+language-only**（`use_state_rows = 1 if use_state else 0`，
  Edge gen 图=16 行、无 +1 state 行 ⇒ use_state=False）。TRTEdge PolicyRunner 把 action_latent 全填
  高斯噪声、action_noisy_mask 全 1、只注入 frame-0 图像 latent + und K/V，无本体状态——**设计如此，非 bug**。

### L2. 实测：单帧反归一化 + 跨帧复测（reverify 确认）
- 真实 DROID 帧（`lerobot/droid_100` ep0 "Put the marker in the pot"）+ 指令喂 goat 引擎，输出反归一化后：
  位置 delta 小（~mm）、有净位移但沿途抖动、rot6d 合法（|v|=1）近恒等、gripper 反归一化超 [0,1]
  （invert+denorm 约定存疑）；4/16/40 步扩散输出范围不变（**非欠去噪**，`DEFAULT_NUM_INFERENCE_STEPS=4`）。
- **跨帧复测（换独立变量=帧，f20/60/100/140）**：输出跨帧近乎相同（mean|fA−fB|≈0.0076–0.0116、max≈0.06）
  ⇒ **单帧 turnkey 路径「帧无关」**：帧内能出连贯位移，但完全不随输入帧变化。vision-only 单帧
  conditioning 不足以产生帧敏感动作。

### L3. 为何「预测 vs GT」对照不成立（表示动物园）
社区 `lerobot/droid_100` GT 是 **7 维笛卡尔 euler** [x,y,z,roll,pitch,yaw,gripper]、lerobot 自归一化 [−1,1]；
Edge 输出是 **10 维 midtrain rot6d**、NVIDIA quantile 归一化；cosmos cookbook 的 droid 样例又是
**8 维关节空间** [joint_pos(7)+gripper(1)]。三套表示 + 三套归一化 + 时间分辨率不同（policy 5fps vs GT 15fps）
⇒ 直接对照是四重错配，「对不上」是必然，非引擎错。

### L4. W4 复测：不是「帧无关」，是「全 conditioning 弱耦合、输出由 seed 主导」（2026-10-26）

W3 只换了「帧」一个变量，且用的是二进制默认 `--guidance 1.0`。读 `cosmos3_policy_inference.cpp`
发现 **1.0 = CFG 关闭**，源码注明「**3.0 matches the reference**」——W3 是在偏离参考配置下测的。
W4 补测三个 W3 没碰的轴（官方 UMI 帧，`engines/cosmos3_policy_droid`，steps 4）：

| 变化的轴 | mean\|diff\| | 占 mean\|action\|=0.7387 |
|---|---|---|
| 确定性底（同参数三次重跑，**逐位相同**） | 0.000000 | 0% |
| guidance 1.0→5.0 | 0.0070 | 0.94% |
| 跨帧（W3） | 0.0094 | 1.27% |
| 跨 prompt（4 条语义不同指令） | 0.0129 | 1.74% |
| **SEED（对照组）** | **0.383** | **51.9%** |
| *参照：chunk 内相邻两步* | *0.389* | *52.7%* |

**SEED / conditioning = 39×**。「去噪不够」已排除：cross-prompt 敏感性 steps 4→16→40 =
0.0129→0.0322→0.0287（**平台化**），steps16≈steps40（扩散已收敛），seed 效应仍 0.45。

⇒ **修正 W3 定性**：不是「vision-only 单帧 conditioning 弱」，而是**帧/文本/CFG 三个轴一致地只有
1–2%**——整条 conditioning 通路没有有效耦合，输出基本是「seed 去噪出的通用轨迹」。三轴**均匀**衰减
更像系统性 attenuation（scale/mask/接线），**「我们的 gen cross-attention 分解补丁 attenuate 了
conditioning」从猜想升级为领先嫌疑**（reasoner 侧同塔同技术却明确输入敏感：n=184 得 42.4%、3.4× chance）。
**根因未定**：补丁接线 vs 模型属性，决定性判据（PyTorch 参考对照）需 Orin 装 cosmos-framework +
GPU torch（实测 `export_venv` 是 CPU-only torch 且无 cosmos 模块），非快速对照。

**附带发现（action space 配置）**：跑 TRTEdge 自带 `robolab/selftest.py` 失败于
`cannot reshape array of size 160 into shape (32,8)`——参考客户端 `Cosmos3JointposClient` 硬编码
**joint_pos 8 维 / chunk 32**，我们引擎是 **midtrain 10 维 / chunk 16**（与 cosmos-framework 的
DROID 定义一致）。`config.json` 无 action_space/dim/chunk 键（只有 `max_action_dim=64`、
`num_embodiment_domains=32`）⇒ action space 由导出侧选、非 checkpoint 决定。selftest 不适用我们的
配置，**既未证明引擎坏、也未解释 seed 主导**，但留下一个未验证变量（换 joint_pos/chunk32 重导出）。

复现：`eval/policy/w4_conditioning_sensitivity.py`（证据 `eval/policy/w4_cfg/` 16 个 action.json
已入库，一键重算 T0 确定性/T1 guidance/T2 prompt/T3 seed/T4-T5 steps 全部差异；帧用官方 UMI 样例，
无需 droid 数据；取数命令见脚本头注）。

**教训**：① turnkey 工具（`cosmos3_policy_inference`）是**简化路径**——vision+language-only、单帧、无 state
锚点，与 cosmos-framework 参考 server（state-conditioned、多帧历史）输入契约不同；拿 turnkey 输出直接比
参考 GT 前，必须先核对**表示 / 归一化 / state / 时间**四者是否对齐。② **反直觉的单次实测要换独立变量复测
再定性，且要换「结论所依赖的那个」变量**：W3 换帧得「帧无关」，W4 换 conditioning 强度（guidance）与
文本才发现真相是「全 conditioning 弱耦合、seed 主导」——只换帧会漏掉它（见
[[reverify-surprising-measurements]] 纪律）。③ **先查工具默认值是否等于参考配置**：W3/W2 都用了
`--guidance` 默认 1.0（CFG 关），而参考是 3.0——偏离参考配置的「能力边界」结论要先排除配置伪影。
④ 测出边界本身有价值，但**「非引擎 bug」这类排除性断言要有对照证据**：W4 之前说「非引擎 bug」只凭
算子数值对拍，没验导出图接线，W4 后降级为「根因未定」。复现：`eval/policy/w3_frame_sensitivity.py`
（证据 `eval/policy/w3_actions/`）与 `eval/policy/w4_conditioning_sensitivity.py`（证据 `eval/policy/w4_cfg/`）。

