#!/bin/bash
# W2 policy 分解重建（goat）：用 USE_DECOMPOSED_POLICY=1 重导 und_prefill+gen
# （分解注意力/RoPE，绕开 TRT 10.7 不认 ONNX Attention/RotaryEmbedding 的墙），
# 然后建引擎。分段执行以在 99% 满的 goat 盘上保持磁盘余量。
#
# 前置（已完成）：apply_policy_decomposed_patch.py 已打补丁（ops.py 加
# decomposed_rope/decomposed_attention + policy_* 包装器；两个 modeling 文件
# import 换包装器），且 test_policy_decomposed.py 数值校验 6 项全过
# （attention 对拍 SDPA err~1e-7，rope 对拍 LLaMA 全维公式 err=0）。
#
# 分段磁盘账（起始 free 6.3G）：
#   A 删废 ONNX(und3.7+gen2.7) → ~12.7G
#   B 导 und(3.7) → 9G；建 und 引擎(3.7) → 5.3G；删 und model.onnx*(2.8) → 8.1G
#   C 导 gen(2.7) → 5.4G；建 gen 引擎(1.3) → 4.1G；删 gen model.onnx*(2.7) → 6.8G
# 引擎建成即自包含，ONNX 权重可从 checkpoint 再生，故建完即删 model.onnx*。
A=/home/nvidia/chenrunbin/alpamayo-edge
export PATH=/usr/local/cuda/bin:$PATH
export EDGELLM_PLUGIN_PATH=$A/trtedge-0.10.1/build_orin/libNvInfer_edgallm_plugin.so
cd "$A" || exit 1
mkdir -p logs
PB=$A/trtedge-0.10.1/build_orin/experimental_models/cosmos3/examples/cosmos3_policy_build
O=$A/onnx/cosmos3_policy_droid
E=$A/engines/cosmos3_policy_droid
CKPT=$A/models/Cosmos3-Edge-Policy-DROID
T=$A/trtedge-0.10.1

free_now() { df -h /home/nvidia | tail -1 | awk '{print "  disk_free="$4}'; }

echo "=== start $(date +%T) ==="; free_now

echo "=== Phase A: 删废 plugin 版 ONNX (und_prefill+gen) $(date +%T) ==="
rm -rf "$O/und_prefill" "$O/gen"
echo "  deleted"; free_now

echo "=== Phase B1: export und_prefill (decomposed) $(date +%T) ==="
( while true; do free -m | awk '/Mem/{print $3}'; sleep 5; done > /tmp/mem_w2d_und.log ) &
MON=$!
cd "$T" && source export_venv/bin/activate
USE_DECOMPOSED_POLICY=1 tensorrt-edgellm-export "$CKPT" "$O" --task policy \
  --components und_prefill > "$A/logs/w2d_export_und.log" 2>&1
rc=$?
deactivate; cd "$A"; kill $MON 2>/dev/null
echo "  und export exit=$rc peak_mem=$(sort -n /tmp/mem_w2d_und.log | tail -1)MB $(date +%T)"; free_now
if [ $rc -ne 0 ]; then echo "EXPORT_UND_FAILED"; tail -30 "$A/logs/w2d_export_und.log"; exit 1; fi

echo "=== Phase B2: build und_prefill engine $(date +%T) ==="
"$PB" --onnxDir "$O" --engineDir "$E" --component und_prefill \
  > "$A/logs/w2d_build_und.log" 2>&1
rc=$?
echo "  und build exit=$rc $(date +%T)"; free_now
if [ $rc -ne 0 ]; then echo "BUILD_UND_FAILED"; tail -30 "$A/logs/w2d_build_und.log"; exit 1; fi
rm -f "$O/und_prefill/model.onnx" "$O/und_prefill/model.onnx.data"
echo "  freed und model.onnx*"; free_now

echo "=== Phase C1: export gen (decomposed) $(date +%T) ==="
( while true; do free -m | awk '/Mem/{print $3}'; sleep 5; done > /tmp/mem_w2d_gen.log ) &
MON=$!
cd "$T" && source export_venv/bin/activate
USE_DECOMPOSED_POLICY=1 tensorrt-edgellm-export "$CKPT" "$O" --task policy \
  --components gen > "$A/logs/w2d_export_gen.log" 2>&1
rc=$?
deactivate; cd "$A"; kill $MON 2>/dev/null
echo "  gen export exit=$rc peak_mem=$(sort -n /tmp/mem_w2d_gen.log | tail -1)MB $(date +%T)"; free_now
if [ $rc -ne 0 ]; then echo "EXPORT_GEN_FAILED"; tail -30 "$A/logs/w2d_export_gen.log"; exit 1; fi

echo "=== Phase C2: build gen engine $(date +%T) ==="
"$PB" --onnxDir "$O" --engineDir "$E" --component gen \
  > "$A/logs/w2d_build_gen.log" 2>&1
rc=$?
echo "  gen build exit=$rc $(date +%T)"; free_now
if [ $rc -ne 0 ]; then echo "BUILD_GEN_FAILED"; tail -30 "$A/logs/w2d_build_gen.log"; exit 1; fi
rm -f "$O/gen/model.onnx" "$O/gen/model.onnx.data"
echo "  freed gen model.onnx*"; free_now

echo "=== Phase D: final $(date +%T) ==="
du -sh "$E"/* 2>/dev/null
free_now
echo "W2D_POLICY_DECOMPOSED_REBUILD_DONE $(date +%T)"
