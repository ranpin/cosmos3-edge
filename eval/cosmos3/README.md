# Cosmos3-Edge 评测（Phase W · W2）—— reasoner n=184 跨代对比

`nvidia/Cosmos3-Edge`（4B 全模态世界模型 = 2B VL reasoner + rectified-flow 扩散专家，MoT 单塔）。
部署链路与六道 JetPack 6.2 兼容墙见本仓 `docs/PROBLEMS.md` §K/§L 与 `docs/DEPLOYMENT.md`。
呈现页：<https://ranpin.github.io/cosmos3-edge/cosmos3.html>（跨代精度对比专篇）。

> **迁移说明（2026-11-03）**：本目录原在 alpamayo-edge（2026-10-29 拆仓时跨代对比曾留彼处）；
> 现按「cosmos3 的尽量都迁去 cosmos3-edge」迁入本仓。CR2-8B 侧对照数据以 `ref_cr2/` 副本随迁
> （manifest / driving RESULTS / CR2 FP16 n184 原始输出 / score_mc.py，逐字节拷贝自
> alpamayo-edge，canonical 仍在彼处），本目录因此可自包含重渲。policy 动作链（W3/W4）在
> 本仓 `eval/policy/`；reasoning 自由描述 demo 见
> <https://ranpin.github.io/cosmos3-edge/cosmos3-demo/>。

## Reasoner 侧：n=184 驾驶域基准（2026-10-16，W2）

**协议与 alpamayo-edge 主页 §1.11/§1.12 完全一致**（同帧、同题、同标答、贪心、cap 12、6 帧 @448），
唯一变量是模型。输入直接复用 `bench/ood_eval/ood_input_goat.json`（messages 格式模型无关，
runtime 用引擎目录内自己的 chat template）。

| 模型 | 正确率 | Wilson 95%CI | vs CR2-8B FP16 | E2E/请求 |
|---|---|---|---|---|
| CR2-8B FP16（goat） | 51.6% | [44.5,58.7] | — | ~1.2s |
| CR2-8B INT4（dog） | 50.5% | [43.4,57.7] | −1.1（p=0.815 ns） | 0.68s |
| CR2-8B INT8（dog） | 47.8% | [40.7,55.0] | −3.8（p=0.143 ns） | 0.50s |
| **Cosmos3-Edge 2B FP16（goat）** | **42.4%** | [35.5,49.6] | **−9.2 pts（b/c=29/12，p=0.0115 显著）** | **0.91s** |

- 184/184 全 EOS、零解析失败、字母分布均衡（A17 B22 C20 D23 E29 F26 G19 H28，无坍缩）。
- 随机基线 12.5% ⇒ 2B reasoner 达 3.4× chance，有真实场景理解力。
- **结论（跨代验证项目核心发现）**：新一代 2B（Cosmos3，2026-07）在驾驶域显著输给上一代 8B
  的量化版（CR2 INT8/INT4）——「缩小模型不免费、量化几乎免费」在第二个模型家族上成立。
- 引擎：FP16，llm 3.7G + visual 940M（分解注意力版；插件版 visual 亦已建成，性能对比待做）。

## 文件

| 文件 | 说明 |
|---|---|
| `cosmos3_edge_fp16_outputs_n184.json` | reasoner n=184 原始输出（实测记录） |
| `cosmos3_viz.py` | 跨代对比图 + policy 帧不敏感图生成（`FIG_THEME=dark` 出暗色变体，输出 `site/figures[/dark]/`） |
| `ref_cr2/` | CR2-8B 侧对照数据副本（manifest_n184 / driving_results / orin_fp16_outputs_n184 / score_mc.py），逐字节拷贝自 alpamayo-edge |
| 打分 | `ref_cr2/score_mc.py` 的解析/Wilson/McNemar（与 alpamayo-edge 同源同实现） |
