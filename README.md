# Cosmos3-Edge

**NVIDIA Cosmos3-Edge（4B 全模态世界模型）在 Jetson AGX Orin 上的边缘部署实证**——用 TensorRT-Edge-LLM
把 reasoner 与扩散动作专家落成 Orin 引擎，覆盖两条能力线：**AV 驾驶场景理解（reasoning）** 与
**DROID 机器人操作（policy 动作链）**。全部为 Orin 实机推理输出，非模拟。

🎬 **在线 Demo（reasoning 侧·真实驾驶场景自由描述）**：<https://ranpin.github.io/cosmos3-edge/cosmos3-demo/>

> 本仓从 `ranpin/alpamayo-edge` 拆出，聚焦 Cosmos3-Edge 世界模型。**AV VLA（Alpamayo-R1）轨迹预测**
> 与 **CR2-8B 量化**、以及 Cosmos3 vs CR2-8B 的**跨代精度对比**仍在 alpamayo-edge（那是"一套 TRTEdge
> 管线部署多代 NVIDIA Physical AI 模型"的研究主线）。

## 模型

Cosmos3-Edge = **2B VL reasoner + rectified-flow 扩散动作专家**（MoT 单塔，4B 全模态世界模型）。
- **Reasoning**：图 + 文 → 场景理解 / 自由描述（AV 驾驶域）。
- **Policy**：图 + 指令 + 状态 → 动作 chunk（官方权重仅 DROID 机器人操作域）。

## 两条能力线

### 1. AV Reasoning —— 已验证 works

- **n=184 驾驶域基准：42.4% 正确率**（Wilson 95%CI [35.5, 49.6]），随机基线 12.5% ⇒ **3.4× chance**，
  有真实场景理解力；184/184 全 EOS、零解析失败、字母分布均衡无坍缩。
- **在线 demo**：4 个公开 sample clip 的自由场景描述（商业区白天 / 施工锥桶 / 夜间左转道 / 夜间 stop sign 行人），
  真实 Orin 推理输出。
- 协议与 CR2-8B 评测完全一致（同帧、同题、同标答、贪心、cap 12、6 帧 @448），唯一变量是模型；
  **跨代对比结论**（新一代 2B 在驾驶域显著输给上一代 8B 量化版）见 alpamayo-edge。

### 2. DROID Policy 动作链 —— 部署打通 + 诚实边界

- 四组件引擎全部建成（und_prefill 2.7G / gen 2.7G / vae_encoder 292M / text_tokenizer 17M），
  `cosmos3_policy_inference` 图 + prompt → action 冒烟通过：action shape **[1,16,10]**、160 值全 finite、
  逐维有结构（非噪声），**5154 ms/policy round**（goat）。
- 🔴 **诚实边界（W3 实测 / W4 复测修正）**：W2 冒烟是**部署管线验证**（分解引擎端到端跑通、产出结构合法动作），
  **不是动作语义正确性验证**。W4 复测：turnkey 单帧路径输出**由 diffusion seed 主导（52%）**，
  帧 / 文本 prompt / CFG 三个 conditioning 轴一致地只有 **1–2%**（seed 是 prompt 效应的 ~30×），
  40 步去噪也不收敛 ⇒ 输出基本是「seed 去噪出的通用轨迹」，conditioning 弱耦合，**根因未定**
  （我们的 gen cross-attention 分解补丁接线 vs 模型属性）。且 Edge 输出（10 维 midtrain rot6d + quantile
  归一化）与社区 lerobot GT（7 维笛卡尔）表示/归一化不同、无法直接对照。⇒ policy 定位为
  **「部署打通 + conditioning 弱耦合边界（根因未定）」**，demo 锚定已验证 works 的 reasoning 侧。

## 边缘部署要点

- **TRTEdge 0.10.1**，Jetson AGX Orin（sm_87 / JetPack 6.2 / CUDA 12.6 / TRT 10.7）。
- **JetPack 6.2 兼容墙**逐个打通；核心一道：policy 的 und_prefill/gen 把注意力/RoPE lower 成 ONNX 标准
  `Attention`/`RotaryEmbedding` 算子（只有 TRT≥11 parser 认识），Orin TRT 10.7 报 "Cannot find plugin"
  ⇒ 用**分解注意力补丁**（`scripts/orin/apply_policy_decomposed_patch.py`：GQA 折批 + arange 下三角 causal
  掩码 + rotate_half RoPE，恒等 gather 保 position_ids 防 DCE）打穿，先经 `test_policy_decomposed.py`
  数值对拍（attention vs SDPA err~1e-7、rope vs LLaMA 公式 err=0）再导出。
- ViT 注意力在 JetPack 6.2 无 legacy 后端（CuTe DSL 预编译件只有 sm_87+cuda13），是 reasoning visual
  引擎的另一道墙。**完整六道兼容墙、分解注意力/RoPE 补丁与动作表示见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**。

## 仓库结构

```
eval/policy/        DROID 操作评测：W3 帧敏感性 + W4 conditioning 敏感性（脚本 + 证据 JSON）
scripts/orin/       Orin 部署脚本：TRTEdge build / reasoning+policy engine build / 分解注意力补丁
site/cosmos3-demo/  reasoning 自由描述 demo（gh-pages 源，4 公开 clip）
docs/DEPLOYMENT.md  部署细节：六道 JetPack 6.2 兼容墙 + 分解注意力/RoPE 补丁 + 动作表示
docs/PROBLEMS.md    工程问题记录（§K/§L，迁自 alpamayo-edge）：现象 → 根因 → 解决 → 教训
```

## 许可与数据来源

- 代码 **MIT**（见 LICENSE）。
- reasoning demo 帧取自 **dgural 公开样本**（NVIDIA PhysicalAI-AV 的公开 100-clip 子集）。
- n=184 基准的逐 clip 数据涉及 gated `nvidia/PhysicalAI-Autonomous-Vehicles`；其完整 benchmark 与跨代
  对比留在 alpamayo-edge 仓，本仓只引用 Cosmos3 自身的聚合准确率。
