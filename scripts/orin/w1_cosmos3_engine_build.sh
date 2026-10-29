#!/bin/bash
# W1：Cosmos3-Edge reasoning 引擎构建（FP16 先行）——用 dog 自建的 TRTEdge 0.10.1。
# 参数口径照上游 docs/source/user_guide/examples/vla/cosmos3.md 的 reasoning 示例
# （maxInputLen 2048 / KV 4096），batch 1 与本项目一贯口径一致。
# 插件路径必须指向 0.10.1 自建 build（共享安装是 0.9.0，不认识 cosmos3）。
A=/home/vision/chenrunbin/alpamayo-edge
T=$A/trtedge-0.10.1/build_orin
export EDGELLM_PLUGIN_PATH=$T/libNvInfer_edgellm_plugin.so
export PATH=/usr/local/cuda/bin:$PATH
O=$A/onnx/cosmos3_edge/reasoning
E=$A/engines/cosmos3_edge/reasoning
mkdir -p "$A/logs"

echo "=== llm_build FP16 $(date +%T) ==="
( while true; do free -m | awk '/Mem/{print $3}'; sleep 2; done > /tmp/mem_w1_llm.log ) &
MON=$!
$T/examples/llm/llm_build --onnxDir $O/llm --engineDir $E/llm \
  --maxInputLen 2048 --maxBatchSize 1 --maxKVCacheCapacity 4096 \
  > $A/logs/w1_llm_build.log 2>&1
rc=$?; kill $MON 2>/dev/null
echo "llm_build exit=$rc peak=$(sort -n /tmp/mem_w1_llm.log | tail -1)MB $(date +%T)"
if [ $rc -ne 0 ]; then tail -20 $A/logs/w1_llm_build.log; echo LLM_BUILD_FAILED; exit 1; fi

echo "=== visual_build $(date +%T) ==="
$T/examples/multimodal/visual_build --onnxDir $O/visual --engineDir $E \
  > $A/logs/w1_visual_build.log 2>&1
rc=$?
echo "visual_build exit=$rc $(date +%T)"
if [ $rc -ne 0 ]; then tail -20 $A/logs/w1_visual_build.log; echo VISUAL_BUILD_FAILED; exit 1; fi

du -sh $E/llm $E/visual 2>/dev/null
echo "W1_BUILD_DONE $(date +%T)"
