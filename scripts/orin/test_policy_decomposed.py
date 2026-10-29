#!/usr/bin/env python3
"""Numerically validate the W2 policy decomposed attention/RoPE before export.

导出 und_prefill+gen 要跑约 1 小时；若分解实现有语义 bug（GQA 头序 / causal
方向 / rope 公式），建引擎时才暴露就太贵。本测试在 CPU 上用小张量对拍：
  - decomposed_attention  vs  F.scaled_dot_product_attention（真 ground truth，
    覆盖 und_prefill 的 is_causal=True 自注意力 + gen 的 is_causal=False 交叉注意力，
    两者都是 GQA H=16/Hkv=8）
  - decomposed_rope       vs  LLaMA 全维公式 q*cos_full + rotate_half(q)*sin_full
    （独立代码路径、同数学；cos_full=cat([cos_half,cos_half])）

判据打印实际值 vs 期望值（max abs err），阈值 1e-3（fp32）。
Usage (goat, inside trtedge-0.10.1/, export_venv):
    USE_DECOMPOSED_POLICY=1 python3 ../scripts/test_policy_decomposed.py
"""
import os
import sys

os.environ.setdefault("USE_DECOMPOSED_POLICY", "1")

import torch
import torch.nn.functional as F

from tensorrt_edgellm.models.ops import (decomposed_attention, decomposed_rope,
                                         policy_attention, policy_rope)

TOL = 1e-3
ok = True


def check(name, got, ref):
    global ok
    err = (got - ref).abs().max().item()
    shape_ok = got.shape == ref.shape
    passed = err < TOL and shape_ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: "
          f"max_abs_err={err:.3e} (tol {TOL:.0e}), "
          f"shape got={tuple(got.shape)} ref={tuple(ref.shape)}")


def ref_repeat_kv(x, rep):
    """HF GQA repeat_kv: kv head h//rep serves query head h (interleave)."""
    b, hkv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, hkv, rep, s,
                                      d).reshape(b, hkv * rep, s, d)


def test_attention(b, h, hkv, s_q, s_kv, is_causal, label):
    d = 128
    torch.manual_seed(0)
    q = torch.randn(b, h, s_q, d, dtype=torch.float32)
    k = torch.randn(b, hkv, s_kv, d, dtype=torch.float32)
    v = torch.randn(b, hkv, s_kv, d, dtype=torch.float32)
    scale = 1.0
    got = decomposed_attention(q, k, v, None, is_causal, scale)
    k_exp = ref_repeat_kv(k, h // hkv)
    v_exp = ref_repeat_kv(v, h // hkv)
    ref = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal,
                                         scale=scale)
    check(f"attention/{label}", got, ref)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def test_rope(b, h, s, label):
    d = 128
    half = d // 2
    torch.manual_seed(1)
    x = torch.randn(b, h, s, d, dtype=torch.float32)
    # packed rope_rotary_cos_sin [B,S,D] = [cos_half | sin_half]
    cos_half = torch.randn(b, s, half, dtype=torch.float32)
    sin_half = torch.randn(b, s, half, dtype=torch.float32)
    cos_in = cos_half.reshape(-1, half)  # [B*S, half] as modeling passes it
    sin_in = sin_half.reshape(-1, half)
    pos = torch.arange(s, dtype=torch.int32).unsqueeze(0).expand(b, -1)
    got = decomposed_rope(x, cos_in, sin_in, pos)
    # reference: full-dim LLaMA formula, gather by pos (identity here)
    idx = pos.long().unsqueeze(-1).expand(b, s, half)
    cos_g = cos_half.gather(1, idx)
    sin_g = sin_half.gather(1, idx)
    cos_full = torch.cat([cos_g, cos_g], dim=-1).unsqueeze(1)  # [B,1,S,D]
    sin_full = torch.cat([sin_g, sin_g], dim=-1).unsqueeze(1)
    ref = x * cos_full + rotate_half(x) * sin_full
    check(f"rope/{label}", got, ref)


def test_dispatch():
    """policy_* wrappers must route to decomposed when env=1."""
    d, half, b, h, hkv, s = 128, 64, 1, 16, 8, 5
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, hkv, s, d)
    v = torch.randn(b, hkv, s, d)
    a1 = policy_attention(q, k, v, None, True, 1.0)
    a2 = decomposed_attention(q, k, v, None, True, 1.0)
    check("dispatch/attention", a1, a2)
    x = torch.randn(b, h, s, d)
    cos = torch.randn(b * s, half)
    sin = torch.randn(b * s, half)
    pos = torch.arange(s, dtype=torch.int32).unsqueeze(0)
    r1 = policy_rope(x, cos, sin, pos)
    r2 = decomposed_rope(x, cos, sin, pos)
    check("dispatch/rope", r1, r2)


print("=== W2 policy decomposed op validation ===")
# und_prefill: causal self-attn, S==Sk, GQA 16/8
test_attention(1, 16, 8, 13, 13, True, "und_prefill_causal_gqa")
# gen: non-causal cross-attn over cat([und,gen]), Sk>Sq, GQA 16/8, batch 2
test_attention(2, 16, 8, 5, 18, False, "gen_cross_gqa")
# rope: und_prefill (b1) + gen (b2)
test_rope(1, 16, 13, "und_prefill")
test_rope(2, 16, 5, "gen")
test_dispatch()
print("=== " + ("ALL PASS" if ok else "SOME FAIL") + " ===")
sys.exit(0 if ok else 1)
