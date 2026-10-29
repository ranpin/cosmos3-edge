# Cosmos3-Edge 评测

`nvidia/Cosmos3-Edge`（4B 全模态世界模型 = 2B VL reasoner + rectified-flow 扩散专家，MoT 单塔）
在 Jetson AGX Orin 上的两条能力线评测。部署链路与 JetPack 6.2 兼容墙、policy 动作链能力边界的
完整技术记录见 alpamayo-edge 仓 `docs/PROBLEMS.md` §K/§L（本研究主线的内部技术文档）。

🎬 **在线 Demo（reasoning 侧自由场景描述，真实 Orin 推理输出）**：
<https://ranpin.github.io/cosmos3-edge/cosmos3-demo/>（源在 `site/cosmos3-demo/`）

## Reasoner 侧：n=184 驾驶域基准（2026-10-16，W2）

**Cosmos3-Edge 2B FP16（goat）：42.4% 正确率**，Wilson 95%CI [35.5, 49.6]，E2E ~0.91s/请求。

- 协议：6 帧 @448、贪心、cap 12，与 CR2-8B 评测完全一致（同帧、同题、同标答），唯一变量是模型。
- 184/184 全 EOS、零解析失败、字母分布均衡（A17 B22 C20 D23 E29 F26 G19 H28，无坍缩）。
- 随机基线 12.5% ⇒ 2B reasoner 达 **3.4× chance**，有真实场景理解力。
- 引擎：FP16，llm 3.7G + visual 940M（分解注意力版）。

> **跨代对比**（Cosmos3-Edge 2B vs 上一代 CR2-8B 量化版：新一代 2B 在驾驶域显著输给上一代 8B 的
> INT8/INT4，−9.2 pts、p=0.0115）属"CR2-8B→Cosmos3→R1"研究主线叙事，**完整 n=184 benchmark + 跨代
> 对比表 + 逐 clip 数据留在 [alpamayo-edge](https://github.com/ranpin/alpamayo-edge)**（涉 gated
> PhysicalAI-AV，不在本仓重复）。本仓只引用 Cosmos3 自身的聚合准确率。

## Policy 侧：DROID 动作链端到端打通（2026-10-16 W2 打通 / 2026-10-19 W3 实测边界）

Policy-DROID（`nvidia/Cosmos3-Edge-Policy-DROID`，机器人操作 WAM）四组件引擎全部建成，
`cosmos3_policy_inference` 图+prompt→action.json 冒烟通过（goat）。

| 组件 | 引擎 | 说明 |
|---|---|---|
| und_prefill | 2.7G | 理解 prefill（28 层，分解注意力版） |
| gen | 2.7G | rectified-flow 扩散动作专家（分解注意力版） |
| vae_encoder | 292M | 观测编码 |
| text_tokenizer | 17M | + embed_tokens 513M（staged） |

**第六道墙（K6）与打通**：policy 的 und_prefill/gen 把注意力/RoPE lower 成 **ONNX 标准
`Attention`/`RotaryEmbedding` 算子**，只有 TRT≥11 parser 认识；Orin TRT 10.7 报
"Cannot find plugin: Attention/RotaryEmbedding"。解法同 K4 思路、推广到 RoPE：
`scripts/orin/apply_policy_decomposed_patch.py` 在 `ops.py` 加 `decomposed_attention`
（GQA 折批 + arange 下三角 causal 掩码，覆盖 und 的 is_causal=True 自注意力与 gen 的
is_causal=False 交叉注意力）+ `decomposed_rope`（rotate_half，恒等 gather 保留
position_ids 防 DCE），env `USE_DECOMPOSED_POLICY=1` 门控。分解实现先经
`test_policy_decomposed.py` 数值对拍（attention vs SDPA err~1e-7、rope vs LLaMA 全维
公式 err=0）再导出。

**冒烟结果**（`policy/policy_smoke_action.json`，官方 UMI 首帧图 + "pick up the object"，4 步扩散）：
exit=0，action shape **[1,16,10]**，160 值全 finite、范围 [−2.13, 2.77]，逐维分布有结构
（d3≈−1.93±0.15、d5≈+2.15±0.14 紧聚类，非噪声）；**延迟 5154 ms/policy round**（goat）。

> **诚实边界（W3 实测 2026-10-19 / W4 复测修正 2026-10-26）**：W2 冒烟是**部署管线验证**——分解引擎
> 端到端跑通、产出结构合法的动作；**不是动作语义正确性验证**。W3 用真实 DROID 帧实测单帧路径「帧无关」；
> **W4 复测修正**：不止帧——**帧/文本 prompt/CFG 三个 conditioning 轴一致地只有 1–2%，而 diffusion seed
> 主导输出（52%，是 prompt 效应的 ~30×）**，40 步也不收敛出来。即 turnkey 路径输出基本是「seed 去噪出的
> 通用轨迹」，conditioning 弱耦合，**根因未定**（我们的 gen cross-attention 分解补丁接线 vs 模型属性）。
> 且 Edge 输出（10 维 midtrain rot6d + quantile 归一化）与社区 lerobot GT（7 维笛卡尔）表示/归一化不同、
> 无法直接对照。⇒ demo 锚定已验证 works 的 reasoning 侧，policy 动作链定位为「部署打通 + conditioning
> 弱耦合边界（根因未定）」。

## 文件

| 路径 | 说明 |
|---|---|
| `policy/w3_frame_sensitivity.py` | W3 帧敏感性复现（换帧 f20/60/100/140，证「帧无关」） |
| `policy/w4_conditioning_sensitivity.py` | W4 conditioning 敏感性复现（帧/prompt/CFG/seed/steps 五轴对照） |
| `policy/w3_actions/` | W3 四帧动作输出证据（f{20,60,100,140}_act.json） |
| `policy/w4_cfg/` | W4 各轴 action.json 证据（det/t1–t5，16 个） |
| `policy/policy_smoke_action.json` | W2 policy 冒烟动作输出（1×16×10，端到端打通凭证） |
| `policy/droid_action_stats.json` | DROID quantile 归一化 stats（取自 cosmos-framework） |
