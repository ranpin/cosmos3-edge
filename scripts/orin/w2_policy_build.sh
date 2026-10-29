#!/bin/bash
# W2 policy 引擎构建链（goat）：und_prefill → gen → 重导出（补 vae）→ vae 引擎。
# 背景：首次 policy 导出 gen/und_prefill/text_tokenizer 成功，vae_encoder 被杀
# （exit=120，疑似内存压力）；cosmos3_policy_build 支持 --component 分组件构建。
A=/home/nvidia/chenrunbin/alpamayo-edge
export PATH=/usr/local/cuda/bin:$PATH
export EDGELLM_PLUGIN_PATH=$A/trtedge-0.10.1/build_orin/libNvInfer_edgellm_plugin.so
cd "$A" || exit 1
PB=trtedge-0.10.1/build_orin/experimental_models/cosmos3/examples/cosmos3_policy_build
O=onnx/cosmos3_policy_droid
E=engines/cosmos3_policy_droid

echo "=== build und_prefill $(date +%T) ==="
$PB --onnxDir $O --engineDir $E --component und_prefill > logs/w2_build_und.log 2>&1
echo "und_prefill exit=$? $(date +%T)"

echo "=== build gen $(date +%T) ==="
$PB --onnxDir $O --engineDir $E --component gen > logs/w2_build_gen.log 2>&1
echo "gen exit=$? $(date +%T)"

echo "=== retry policy export (补 vae_encoder) $(date +%T) ==="
cd trtedge-0.10.1 && source export_venv/bin/activate
tensorrt-edgellm-export ../models/Cosmos3-Edge-Policy-DROID ../onnx/cosmos3_policy_droid --task policy \
  > ../logs/w2_export_policy2.log 2>&1
echo "export2 exit=$? $(date +%T)"
deactivate
cd "$A"

if [ -d "$O/vae_encoder" ]; then
  echo "=== build vae_encoder $(date +%T) ==="
  $PB --onnxDir $O --engineDir $E --component vae_encoder > logs/w2_build_vae.log 2>&1
  echo "vae exit=$? $(date +%T)"
else
  echo "vae_encoder 仍缺失"
fi
du -sh $E/* 2>/dev/null
echo "W2_POLICY_BUILD_CHAIN_DONE $(date +%T)"
