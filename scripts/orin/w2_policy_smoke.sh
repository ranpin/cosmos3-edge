#!/bin/bash
# W2 policy 冒烟测试（goat）：四组件引擎全建成后，用官方样例图 + 操作指令跑
# cosmos3_policy_inference，验证 und_prefill → gen 扩散去噪 → vae → 动作 全链
# 在「分解注意力」引擎上端到端跑通（这些引擎绕过了 TRT 10.7 不认 ONNX
# Attention/RotaryEmbedding 的墙）。判据：exit=0 + 产出结构合法的 action.json。
#
# conditioning 图用官方 example_action_fd_umi_first_frame.png（256×256 RGB，
# 机器人操作首帧）；domain 用引擎 canonical 的 droid_lerobot（domain_id=8 已
# baked 进 action_proj）。冒烟只验管线跑通与输出格式，不验动作语义正确性。
A=/home/nvidia/chenrunbin/alpamayo-edge
export PATH=/usr/local/cuda/bin:$PATH
export EDGELLM_PLUGIN_PATH=$A/trtedge-0.10.1/build_orin/libNvInfer_edgellm_plugin.so
cd "$A" || exit 1
mkdir -p eval/cosmos3 logs
PI=$A/trtedge-0.10.1/build_orin/experimental_models/cosmos3/examples/cosmos3_policy_inference
IMG=$A/models/Cosmos3-Edge/assets/example_action_fd_umi_first_frame.png
OUT=$A/eval/cosmos3/policy_smoke_action.json

echo "=== plugin lib ==="; ls -la "$EDGELLM_PLUGIN_PATH" 2>&1 | head -1
echo "=== image ==="; ls -la "$IMG" 2>&1 | head -1
echo "=== free mem before ==="; free -m | awk '/Mem/{print "  used="$3"MB avail="$7"MB"}'

echo "=== run policy inference $(date +%T) ==="
"$PI" --engineDir "$A/engines/cosmos3_policy_droid" \
      --image "$IMG" \
      --prompt "pick up the object" \
      --output "$OUT" \
      --steps 4 --seed 0 --iters 1 --warmup 1 \
      > logs/w2d_smoke.log 2>&1
rc=$?
echo "  inference exit=$rc $(date +%T)"
echo "=== free mem after ==="; free -m | awk '/Mem/{print "  used="$3"MB avail="$7"MB"}'

if [ $rc -ne 0 ]; then
  echo "SMOKE_FAILED — log tail:"; tail -40 logs/w2d_smoke.log; exit 1
fi

echo "=== inference log tail ==="; tail -20 logs/w2d_smoke.log
echo "=== action.json ==="; ls -la "$OUT" 2>&1
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if isinstance(d, dict):
    print("keys:", list(d.keys()))
    for k in ("prompt","domain_name","view_point","fps","action_chunk_size",
              "num_chunks","action_shape_per_chunk"):
        if k in d: print(f"  {k} = {d[k]}")
    ac = d.get("action_chunks")
    if ac is not None:
        import itertools
        flat = list(itertools.chain.from_iterable(
            c if isinstance(c[0], list) else [c] for c in ac))
        nums = [x for row in (ac[0] if isinstance(ac[0][0], list) else ac) for x in row] \
            if isinstance(ac[0][0], list) else ac[0]
        print(f"  action_chunks: {len(ac)} chunk(s), first chunk rows={len(ac[0])}")
        print(f"  first action vec (dim={len(ac[0][0])}): {[round(x,4) for x in ac[0][0]]}")
        allv = [x for ch in ac for row in ch for x in row]
        print(f"  value range: [{min(allv):.4f}, {max(allv):.4f}]  (normalized ~[-1,1] expected)")
else:
    print("type:", type(d), "preview:", str(d)[:300])
PY
echo "W2D_POLICY_SMOKE_DONE $(date +%T)"
