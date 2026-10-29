#!/bin/bash
# W0b：dog 自有目录构建 TRTEdge v0.10.1（Cosmos3-Edge 官方支持的最低版本系）。
# 原则：共享安装 /home/vision/TensorRT-Edge-LLM（0.9.0）一律不动——其他用户依赖它，
# 且它是全部已发布数字的可复现锚点。新 build 在 chenrunbin/alpamayo-edge/trtedge-0.10.1/。
# 配方沿用 7 月验证过的 Orin 构建（EMBEDDED_TARGET=jetson-orin，FMHA 教训见 orin_build_notes.md），
# 新增 -DBUILD_EXPERIMENTAL_MODELS=ON（cosmos3 policy 管线所需，上游 docs 明示）。
# 已知风险：dog cmake 3.22.1 可能低于 0.10.x 要求（configure 阶段即暴露，日志留档）。
A=/home/vision/chenrunbin/alpamayo-edge
export PATH=/usr/local/cuda/bin:$PATH
cd "$A" || exit 1
mkdir -p logs

echo "=== clone v0.10.1 $(date +%T) ==="
if [ -d trtedge-0.10.1/.git ]; then
  echo "clone exists, reuse"
else
  rm -rf trtedge-0.10.1
  git clone --depth 1 --branch v0.10.1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git trtedge-0.10.1 \
    || { echo CLONE_FAILED; exit 1; }
fi
cd trtedge-0.10.1 || exit 1
grep -m1 "cmake_minimum_required" CMakeLists.txt

# 0.10.x 新增 nlohmann/json 依赖（actionBuilder 等），dog 系统无此头文件且
# sudo 有密码锁不能 apt ⇒ 单头文件版落自有 deps 目录，-I 注入（GitHub dog 可达）。
DEPS=$A/deps
if [ ! -f "$DEPS/include/nlohmann/json.hpp" ]; then
  echo "=== fetch nlohmann/json single header ==="
  mkdir -p "$DEPS/include/nlohmann"
  curl -sL --max-time 120 -o "$DEPS/include/nlohmann/json.hpp" \
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
    || { echo JSON_FETCH_FAILED; exit 1; }
  head -c 200 "$DEPS/include/nlohmann/json.hpp" | grep -q nlohmann \
    || { echo JSON_FETCH_BAD_CONTENT; head -c 200 "$DEPS/include/nlohmann/json.hpp"; exit 1; }
fi

echo "=== configure $(date +%T) ==="
# ENABLE_CUTE_DSL=OFF：0.10.x 默认 "fmha"，但 CuTe DSL 预编译件只有 sm_87+cuda13
# （JetPack 7.2），dog 是 cuda12.6 ⇒ 必须关掉走传统 FMHA 路径（0.9.0 同款，
# configure 日志确认 legacy FMHA 保留 sm_87）。这就是支持矩阵里 JetPack 6.2
# 只是 Compatible 档的具体代价之一。
cmake -B build_orin -S . \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=OFF \
  -DCMAKE_CXX_FLAGS="-I$DEPS/include" \
  -DBUILD_EXPERIMENTAL_MODELS=ON > "$A/logs/w0b_configure.log" 2>&1
rc=$?
echo "configure exit=$rc"
if [ $rc -ne 0 ]; then tail -25 "$A/logs/w0b_configure.log"; echo CONFIGURE_FAILED; exit 1; fi

echo "=== build $(date +%T) ==="
cmake --build build_orin -j8 > "$A/logs/w0b_build.log" 2>&1
rc=$?
echo "build exit=$rc $(date +%T)"
if [ $rc -ne 0 ]; then tail -30 "$A/logs/w0b_build.log"; echo BUILD_FAILED; exit 1; fi

echo "=== artifacts ==="
ls build_orin/examples/llm/ 2>/dev/null | head -8
ls build_orin/experimental_models/cosmos3/examples/ 2>/dev/null | head -8
ls build_orin/libNvInfer_edgellm_plugin.so 2>/dev/null
echo "W0B_DONE $(date +%T)"
