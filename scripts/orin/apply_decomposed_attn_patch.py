#!/usr/bin/env python3
"""Patch TRTEdge 0.10.1 cosmos3 visual exporter: add decomposed-attention path.

Why (W1 实测链, 2026-10-16):
  JetPack 6.2 (TRT 10.7 / CUDA 12.6) 上 visual 引擎的两条内建注意力路径全被版本锁死:
  1. ViTAttentionPlugin → 只走 CuTe DSL FMHA (0.10.x 移除了 legacy 后端),
     而 sm_87 的 CuTe DSL 预编译件只有 cuda13 版 (上游 build 脚本对 cuda12
     的 target 列表刻意不含 aarch64:sm_87);
  2. USE_TRT_NATIVE_ATTN=1 → 导出 TRT_Attention 算子, 但 TRT parser 对它的
     支持是 11.0 才加入 (TensorRT CHANGELOG), ragged 支持 11.1 —— 10.7 报
     "No checker registered for op: TRT_Attention".
  ⇒ 第三条路: 用基础 ONNX 算子分解实现块对角 ragged self-attention
     (Range/Slice/Compare/MatMul/Where/Softmax), 任何 TRT 版本可解析。
     TRT 会用 Myelin 自行融合; 448/720p 档 scores (H,T,T) 内存无压力,
     原生 1080p×6 (T=12240) 约 4.8GB fp16 scores —— 高分辨率档慎用 (已注明)。

Usage (on device, inside trtedge-0.10.1/):
    python3 apply_decomposed_attn_patch.py          # 打补丁 (幂等)
    python3 apply_decomposed_attn_patch.py --diff   # 只打印将做的修改
Then export with:
    USE_DECOMPOSED_ATTN=1 tensorrt-edgellm-export <ckpt> <out> --task reasoning
"""
import sys

TARGET = "tensorrt_edgellm/models/cosmos3_reasoner/modeling_cosmos3_reasoner_visual.py"

HELPERS = '''

def _is_decomposed_attn_enabled():
    """USE_DECOMPOSED_ATTN=1: plain-op attention for TRT < 11 / no-CuTe-DSL builds."""
    import os
    return os.environ.get("USE_DECOMPOSED_ATTN") == "1"


def decomposed_ragged_attention(query_states, key_states, value_states,
                                cu_seqlens, num_heads, head_size,
                                attention_scale):
    """Block-diagonal ragged self-attention with basic ops only.

    query/key/value_states: [T, num_heads, head_size]
    cu_seqlens: [F+1] segment boundaries (int32/int64)
    Returns [T, num_heads, head_size].

    Mask is built from cu_seqlens via broadcast comparisons + a small
    (T,F)@(F,T) matmul, so the ONNX graph contains only Range/Slice/Cast/
    Greater/Less/And/MatMul/Where/Softmax/Transpose - parseable by any TRT.
    Memory: scores are (H,T,T); fine for T<=~6k, heavy at T=12240 (1080p x6).
    """
    total = query_states.shape[0]
    idx = torch.arange(total, device=query_states.device)
    starts = cu_seqlens[:-1].to(torch.int64)
    ends = cu_seqlens[1:].to(torch.int64)
    idx_col = idx.unsqueeze(1)
    in_seg = (idx_col >= starts.unsqueeze(0)) & (idx_col < ends.unsqueeze(0))
    seg_f = in_seg.to(query_states.dtype)
    same_seg = torch.matmul(seg_f, seg_f.transpose(0, 1)) > 0.5
    q = query_states.permute(1, 0, 2)
    k = key_states.permute(1, 0, 2)
    v = value_states.permute(1, 0, 2)
    scores = torch.matmul(q, k.transpose(1, 2)) * attention_scale
    neg = torch.finfo(scores.dtype).min
    scores = torch.where(same_seg.unsqueeze(0), scores, torch.full_like(scores, neg))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.permute(1, 0, 2)

'''

# 在 attention 分支处插入 elif（锚定 plugin 分支的调用头）
ANCHOR = """        if self._use_trt_attn:
            attn_output = trt_ragged_attention(
                q,
                k,
                v,
                cu_seqlens,
                kv_lengths,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale)
        else:
            attn_output = vit_attention_plugin("""

REPLACEMENT = """        if self._use_trt_attn:
            attn_output = trt_ragged_attention(
                q,
                k,
                v,
                cu_seqlens,
                kv_lengths,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale)
        elif _is_decomposed_attn_enabled():
            attn_output = decomposed_ragged_attention(
                q,
                k,
                v,
                cu_seqlens,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale)
        else:
            attn_output = vit_attention_plugin("""


def main():
    diff_only = "--diff" in sys.argv
    src = open(TARGET).read()
    if "decomposed_ragged_attention" in src:
        print("already patched")
        return 0
    if ANCHOR not in src:
        print("ANCHOR NOT FOUND — 上游代码结构变了，需人工适配")
        return 1
    # helpers 插到第一个 class 定义之前
    cls_idx = src.find("\nclass ")
    assert cls_idx > 0
    patched = src[:cls_idx] + HELPERS + src[cls_idx:]
    patched = patched.replace(ANCHOR, REPLACEMENT, 1)
    if diff_only:
        import difflib
        for line in difflib.unified_diff(src.splitlines(), patched.splitlines(),
                                         lineterm="", n=2):
            print(line)
        return 0
    open(TARGET, "w").write(patched)
    print("PATCHED", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
