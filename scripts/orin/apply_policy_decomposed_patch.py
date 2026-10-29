#!/usr/bin/env python3
"""Patch TRTEdge 0.10.1 cosmos3 policy exporter: decomposed attention + RoPE.

Why (W2 实测链, 2026-10-16):
  policy 的 und_prefill / gen 两个 ONNX 图把注意力与 RoPE 导出成 ONNX 标准算子
  ``Attention`` / ``RotaryEmbedding``（trt::attention_onnx / trt::rope_onnx 的
  lowering 目标）。这两个算子只有 TRT >= 11 的 parser 认识；Orin JetPack 6.2
  是 TRT 10.7，构建时报
      "Cannot find plugin: Attention/RotaryEmbedding, version: 1, namespace:."
  ⇒ 与 W1 visual 同一道版本墙。解法同 K4：用基础 ONNX 算子（MatMul/Softmax/
  Where/GatherElements）把 attention 与 RoPE 分解掉，任何 TRT 版本可解析，
  TRT 用 Myelin 自行融合。

  语义钉死（补丁正确性依据）：
  - attention_pos_id 是恒等 gather（0..S-1）：cosmos3PolicyRunner.cpp:257
    "identity gather indices 0..genLen-1"。故 RoPE 的 gather 是 no-op，但仍用
    torch.gather 保留 position_ids 作为图输入（否则被 DCE，runtime 找不到
    attention_pos_id binding 会硬失败）。
  - rope_rotary_cos_sin 形状 [B,S,128]：cos=[..., :64]、sin=[..., 64:]，
    rotate_half 约定（head_dim=128, half=64）。
  - GQA：und/gen 均 H=16 / Hkv=8（2:1），分解里把 rep 折进 batch 维，不显式
    repeat k/v。
  - und_prefill: is_causal=True（自注意力，S==Sk）；gen: is_causal=False
    （对 cat([und,gen]) 全注意力）。两种都覆盖。

Usage (on device, inside trtedge-0.10.1/):
    python3 apply_policy_decomposed_patch.py          # 打补丁 (幂等)
    python3 apply_policy_decomposed_patch.py --diff   # 只打印将做的修改
Then re-export policy with:
    USE_DECOMPOSED_POLICY=1 tensorrt-edgellm-export <ckpt> <out> --task policy
"""
import sys

OPS = "tensorrt_edgellm/models/ops.py"
UND = "tensorrt_edgellm/models/cosmos3/modeling_und_prefill.py"
GEN = "tensorrt_edgellm/models/cosmos3/modeling_gen.py"

# 追加到 ops.py 末尾（rope_onnx / attention_onnx 两个 custom_op 之后）。
HELPERS = '''

# ---------------------------------------------------------------------------
# W2 policy decomposed attention / RoPE  (JetPack 6.2 / TRT 10.7 compat)
# ---------------------------------------------------------------------------
# trt::attention_onnx / trt::rope_onnx 会 lower 成 ONNX 标准 Attention /
# RotaryEmbedding，只有 TRT>=11 parser 能解析；Orin JetPack 6.2 (TRT 10.7)
# 构建 und_prefill/gen 引擎时报 "Cannot find plugin: Attention/RotaryEmbedding"。
# 导出时设 USE_DECOMPOSED_POLICY=1，把两个算子改走基础 ONNX 算子分解，
# 任何 TRT 版本可解析（与 W1 visual 的分解注意力同一套路）。


def _use_decomposed_policy() -> bool:
    import os
    return os.environ.get("USE_DECOMPOSED_POLICY") == "1"


def decomposed_rope(x, cos, sin, position_ids):
    """rotate_half RoPE, plain ops. x [B,H,S,D]; cos/sin [B*S, D//2].

    attention_pos_id 对 und_prefill/gen 都是恒等 gather（0..S-1），故逐 token
    直接对齐即可；但仍用 torch.gather 保留 position_ids 作为图输入（no-op），
    避免被 DCE 掉导致 runtime 找不到 attention_pos_id binding。
    """
    b, h, s, d = x.shape
    half = d // 2
    cos_t = cos.reshape(b, s, half)
    sin_t = sin.reshape(b, s, half)
    pos = position_ids.to(torch.int64).unsqueeze(-1).expand(b, s, half)
    cos_g = torch.gather(cos_t, 1, pos).unsqueeze(1)
    sin_g = torch.gather(sin_t, 1, pos).unsqueeze(1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos_g - x2 * sin_g, x2 * cos_g + x1 * sin_g),
                     dim=-1)


def decomposed_attention(query, key, value, attn_mask, is_causal, scale):
    """GQA-aware attention, plain ops. q [B,H,S,D]; k/v [B,Hkv,Sk,D].

    把 rep=H//Hkv 折进 batch 维做 grouped matmul，不显式 repeat k/v。
    is_causal=True 时用 arange 比较构造下三角 keep 掩码（避免 Trilu 算子）。
    """
    b, hq, s, d = query.shape
    hkv = key.shape[1]
    sk = key.shape[2]
    rep = hq // hkv
    q = query.view(b, hkv, rep, s, d)
    k = key.unsqueeze(2)
    v = value.unsqueeze(2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        iq = torch.arange(s, device=query.device).unsqueeze(1)
        ik = torch.arange(sk, device=query.device).unsqueeze(0)
        keep = ik <= (iq + (sk - s))
        neg = torch.finfo(scores.dtype).min
        scores = torch.where(keep.unsqueeze(0).unsqueeze(0), scores,
                             torch.full_like(scores, neg))
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.reshape(b, hq, s, d)


def policy_rope(x, cos, sin, position_ids):
    if _use_decomposed_policy():
        return decomposed_rope(x, cos, sin, position_ids)
    return rope_onnx(x, cos, sin, position_ids)


def policy_attention(query, key, value, attn_mask, is_causal, scale):
    if _use_decomposed_policy():
        return decomposed_attention(query, key, value, attn_mask, is_causal,
                                    scale)
    return attention_onnx(query, key, value, attn_mask, is_causal, scale)

'''

# 两个 modeling 文件里的局部 import（各出现一次），换成走包装器。
IMPORT_OLD = "from tensorrt_edgellm.models.ops import attention_onnx, rope_onnx"
IMPORT_NEW = ("from tensorrt_edgellm.models.ops import "
              "policy_attention as attention_onnx, policy_rope as rope_onnx")


def patch_ops(diff_only):
    src = open(OPS).read()
    if "def decomposed_rope(" in src:
        print("ops.py already patched")
        return src
    patched = src.rstrip("\n") + HELPERS
    if diff_only:
        _diff(src, patched, OPS)
        return src
    open(OPS, "w").write(patched)
    print("PATCHED", OPS)
    return patched


def patch_modeling(path, diff_only):
    src = open(path).read()
    if IMPORT_NEW in src:
        print(path, "already patched")
        return
    n = src.count(IMPORT_OLD)
    if n != 1:
        print("ANCHOR count=%d in %s — 需人工适配" % (n, path))
        sys.exit(1)
    patched = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    if diff_only:
        _diff(src, patched, path)
        return
    open(path, "w").write(patched)
    print("PATCHED", path)


def _diff(a, b, name):
    import difflib
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                     fromfile=name, tofile=name + ".patched",
                                     lineterm="", n=2):
        print(line)


def main():
    diff_only = "--diff" in sys.argv
    patch_ops(diff_only)
    patch_modeling(UND, diff_only)
    patch_modeling(GEN, diff_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
