#!/bin/bash
# W1-goat：TRTEdge 0.10.1 C++ 构建 + Cosmos3-Edge reasoning 引擎构建（全部在 goat）。
# 背景：dog 的 GPU 统一内存被 system_agent 服务占 26.5G（2026-10-15 18:07 起），
# 构建与推理都无余量 ⇒ 改在 goat（61G，~28G 可用）构建；引擎 sm_87+TRT 10.7.0.23
# 与 dog 同版互通（2026-07 已验证）。
# 与 dog 版 w1 脚本同源，两处差异：① 路径在 goat；② 需先打 vector_types.h 补丁
# （CUDA 12.6 的 cuda_fp16.h 依赖链，CUDA 13 重组掉了；dog 上同款补丁已验证）。
A=/home/nvidia/chenrunbin/alpamayo-edge
export PATH="$HOME/.local/bin:$PATH"
export PATH=/usr/local/cuda/bin:$PATH
cd "$A" || exit 1
mkdir -p logs

echo "=== patch gen_cpp_header.py $(date +%T) ==="
cd trtedge-0.10.1 || { echo NO_CLONE; exit 1; }
python3 - << 'EOF'
p = "kernelSrcs/jit_utils/gen_cpp_header.py"
s = open(p).read()
if "'vector_types.h'" in s:
    print("already patched")
else:
    old = """REQUIRED_CUDA_HEADERS = [
    'cuda_bf16.h',
    'cuda_bf16.hpp',
    'cuda_fp16.h',
    'cuda_fp16.hpp',
]"""
    new = """REQUIRED_CUDA_HEADERS = [
    'cuda_bf16.h',
    'cuda_bf16.hpp',
    'cuda_fp16.h',
    'cuda_fp16.hpp',
    # CUDA 12.6 (JetPack 6.2) compat: cuda_fp16.h includes vector_types.h,
    # which includes crt/host_defines.h. CUDA 13 reorganized these away.
    'vector_types.h',
    'crt/host_defines.h',
]"""
    assert old in s, "pattern not found"
    open(p, "w").write(s.replace(old, new))
    print("patched")
EOF

echo "=== configure $(date +%T) ==="
# nlohmann：goat 的 clone 带子模块（3rdParty/nlohmannJson），CMake 自动纳入
# include 路径；dog 当时缺子模块才用 -I deps/include 单头文件绕过。
CXX_EXTRA=""
if [ ! -d 3rdParty/nlohmannJson/include ]; then
  mkdir -p "$A/deps/include/nlohmann"
  curl -sL --max-time 120 -o "$A/deps/include/nlohmann/json.hpp" \
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
    || { echo JSON_FETCH_FAILED; exit 1; }
  CXX_EXTRA="-I$A/deps/include"
fi
cmake -B build_orin -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=OFF \
  -DCMAKE_CXX_FLAGS="$CXX_EXTRA" \
  -DBUILD_EXPERIMENTAL_MODELS=ON > "$A/logs/w1g_configure.log" 2>&1
rc=$?
echo "configure exit=$rc"
if [ $rc -ne 0 ]; then tail -25 "$A/logs/w1g_configure.log"; echo CONFIGURE_FAILED; exit 1; fi

echo "=== build $(date +%T) ==="
cmake --build build_orin -j8 > "$A/logs/w1g_build.log" 2>&1
rc=$?
echo "build exit=$rc $(date +%T)"
if [ $rc -ne 0 ]; then tail -30 "$A/logs/w1g_build.log"; echo BUILD_FAILED; exit 1; fi

T=$A/trtedge-0.10.1/build_orin
export EDGELLM_PLUGIN_PATH=$T/libNvInfer_edgellm_plugin.so
O=$A/onnx/cosmos3_edge/reasoning
E=$A/engines/cosmos3_edge/reasoning

echo "=== llm_build FP16 $(date +%T) ==="
( while true; do free -m | awk '/Mem/{print $3}'; sleep 2; done > /tmp/mem_w1g_llm.log ) &
MON=$!
$T/examples/llm/llm_build --onnxDir $O/llm --engineDir $E/llm \
  --maxInputLen 2048 --maxBatchSize 1 --maxKVCacheCapacity 4096 \
  > $A/logs/w1g_llm_build.log 2>&1
rc=$?; kill $MON 2>/dev/null
echo "llm_build exit=$rc peak=$(sort -n /tmp/mem_w1g_llm.log | tail -1)MB $(date +%T)"
if [ $rc -ne 0 ]; then tail -20 $A/logs/w1g_llm_build.log; echo LLM_BUILD_FAILED; exit 1; fi

echo "=== visual_build $(date +%T) ==="
$T/examples/multimodal/visual_build --onnxDir $O/visual --engineDir $E \
  > $A/logs/w1g_visual_build.log 2>&1
rc=$?
echo "visual_build exit=$rc $(date +%T)"
if [ $rc -ne 0 ]; then tail -20 $A/logs/w1g_visual_build.log; echo VISUAL_BUILD_FAILED; exit 1; fi

du -sh $E/llm $E/visual 2>/dev/null
echo "W1G_DONE $(date +%T)"
