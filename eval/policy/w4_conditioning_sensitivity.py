#!/usr/bin/env python3
"""W4 conditioning-sensitivity repro (PROBLEMS §L addendum).

W3 concluded the Cosmos3-Edge policy turnkey path is FRAME-INSENSITIVE. But W3
was run at the binary's DEFAULT --guidance 1.0, which the TRTEdge source
documents as "CFG off (single conditional forward). 3.0 matches the reference."
Classifier-free guidance is precisely the knob that amplifies conditioning, so
W3's conclusion was drawn under a non-reference config.

W4 varies the two conditioning axes W3 never touched, on the official UMI
conditioning frame (models/Cosmos3-Edge/assets/example_action_fd_umi_first_frame.png):

  T0  determinism   same frame/prompt/guidance/seed, run twice  -> noise floor
  T1  guidance      1.0 / 3.0 / 5.0, same frame + prompt        -> does CFG matter?
  T2  prompt        4 semantically distinct instructions @ g=3.0 -> is the
                    UND->GEN cross-attention (our decomposed patch) carrying text?
  T3  seed          0 / 1 / 2 / 3, everything else fixed        -> what DOES drive
                    the output? (control: the one axis that is not conditioning)
  T4  steps         4 vs 16 @ g=3.0, and cross-prompt at 16     -> does more
                    denoising sharpen conditioning?

T3 is the control that makes T1/T2 interpretable: if varying the seed moves the
action far more than varying the observation, the instruction, or the CFG scale,
then the output is seed-dominated and the conditioning path is not effectively
coupled -- which is a different (and stronger) claim than W3's "frame-insensitive".

T2 doubles as an end-to-end check of apply_policy_decomposed_patch.py:
test_policy_decomposed.py validated the OPERATOR numerically (attention vs SDPA
err~1e-7), but never validated the WIRING in the exported graph. If text
conditioning is dead, the gen cross-attention is not receiving the UND K/V.

Evidence (committed, KB-scale): w4_cfg/*.json — raw normalized action chunks
[16,10] = [eef_pos(3) + rot6d(6) + gripper(1)], goat, engines/cosmos3_policy_droid,
--steps 4 --iters 1 --warmup 1.

Regeneration (goat; engines already built, no rebuild/re-export needed):
  A=/home/nvidia/chenrunbin/alpamayo-edge
  export EDGELLM_PLUGIN_PATH=$A/trtedge-0.10.1/build_orin/libNvInfer_edgellm_plugin.so
  PI=$A/trtedge-0.10.1/build_orin/experimental_models/cosmos3/examples/cosmos3_policy_inference
  IMG=$A/models/Cosmos3-Edge/assets/example_action_fd_umi_first_frame.png
  $PI --engineDir $A/engines/cosmos3_policy_droid --image "$IMG" \
      --prompt "<instruction>" --output <name>.json \
      --steps 4 --seed 0 --guidance <g> --iters 1 --warmup 1
  # det_a/det_b/t1_g10: g=1.0 "pick up the object"   (det_a == det_b == t1_g10)
  # t1_g30 / t1_g50:    g=3.0 / 5.0 "pick up the object"
  # t2_open/pour/push:  g=3.0 "open the drawer" / "pour water into the cup"
  #                     / "push the block to the left"

Usage:
    .venv/bin/python eval/cosmos3/w4_conditioning_sensitivity.py
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W4 = os.path.join(HERE, "w4_cfg")
W3 = os.path.join(HERE, "w3_actions")

PROMPTS = {
    "t1_g30": "pick up the object",
    "t2_open": "open the drawer",
    "t2_pour": "pour water into the cup",
    "t2_push": "push the block to the left",
}
POS = slice(0, 3)    # eef position delta
ROT6D = slice(3, 9)  # 6D rotation
GRIP = 9             # gripper


def load(d, name):
    return np.array(json.load(open(os.path.join(d, f"{name}.json")))["action"],
                    dtype=float)


def diff(a, b):
    d = np.abs(a - b)
    return d.mean(), d.max()


def report(label, a, b):
    m, x = diff(a, b)
    print(f"    {label:<34s} mean|d|={m:.6f}  max|d|={x:.6f}")
    return m


def main():
    acts = {n: load(W4, n) for n in
            ["det_a", "det_b", "t1_g10", "t1_g30", "t1_g50",
             "t2_open", "t2_pour", "t2_push",
             "t3_s1", "t3_s2", "t3_s3",
             "t4_st16_pick", "t4_st16_pour",
             "t5_st40_pick", "t5_st40_pour", "t5_st40_s1"]}
    print("=== W4 conditioning-sensitivity repro (PROBLEMS §L addendum) ===")
    for n, a in acts.items():
        p = json.load(open(os.path.join(W4, f"{n}.json"))).get("prompt", "?")
        print(f"  {n:9s} shape={a.shape} range=[{a.min():+.4f},{a.max():+.4f}]"
              f"  prompt={p!r}")

    # Signal scale: diffs are only meaningful relative to the action's own spread.
    ref = acts["t1_g30"]
    spread = np.abs(ref).mean()
    within = np.abs(np.diff(ref, axis=0)).mean()
    print(f"\n[0] signal scale (t1_g30): mean|action|={spread:.4f}  "
          f"mean|step-to-step within chunk|={within:.4f}")

    print("\n[T0] determinism — identical args run twice (noise floor)")
    same = np.array_equal(acts["det_a"], acts["det_b"])
    floor = report("det_a vs det_b", acts["det_a"], acts["det_b"])
    report("det_a vs t1_g10 (3rd run)", acts["det_a"], acts["t1_g10"])
    print(f"    bitwise identical: {same}")

    print("\n[T1] guidance sweep — same frame, same prompt, seed 0")
    g = {}
    g["g1.0 vs g3.0"] = report("g1.0 vs g3.0", acts["t1_g10"], acts["t1_g30"])
    g["g1.0 vs g5.0"] = report("g1.0 vs g5.0", acts["t1_g10"], acts["t1_g50"])
    g["g3.0 vs g5.0"] = report("g3.0 vs g5.0", acts["t1_g30"], acts["t1_g50"])

    print("\n[T2] prompt sensitivity — guidance 3.0, same frame, seed 0")
    p = {}
    for n in ["t2_open", "t2_pour", "t2_push"]:
        p[f"pick_up vs {n}"] = report(f"pick_up vs {n[3:]}", ref, acts[n])
    for a, b in itertools.combinations(["t2_open", "t2_pour", "t2_push"], 2):
        p[f"{a} vs {b}"] = report(f"{a[3:]} vs {b[3:]}", acts[a], acts[b])

    # W3 cross-frame baseline, recomputed from committed evidence for a fair ruler.
    print("\n[W3] cross-frame baseline (recomputed from w3_actions/)")
    w3 = []
    if os.path.isdir(W3):
        fr = {n: load(W3, f"f{n}_act") for n in [20, 60, 100, 140]}
        for a, b in itertools.combinations(fr, 2):
            w3.append(report(f"f{a} vs f{b}", fr[a], fr[b]))
    else:
        print("    (w3_actions/ not found)")

    print("\n[T3] seed sweep — CONTROL: everything fixed except the diffusion seed")
    s = {}
    for n in ["t3_s1", "t3_s2", "t3_s3"]:
        s[f"seed0 vs {n[-2:]}"] = report(f"seed0 vs {n[4:]}", ref, acts[n])
    for a, b in itertools.combinations(["t3_s1", "t3_s2", "t3_s3"], 2):
        s[f"{a[4:]} vs {b[4:]}"] = report(f"{a[4:]} vs {b[4:]}", acts[a], acts[b])

    print("\n[T4] steps sweep — does more denoising sharpen conditioning?")
    st = {}
    st["steps4 vs steps16 (pick)"] = report("steps4 vs steps16 (pick)",
                                            ref, acts["t4_st16_pick"])
    st["cross-prompt @ steps16"] = report("pick vs pour @ steps16",
                                          acts["t4_st16_pick"],
                                          acts["t4_st16_pour"])
    print(f"    (cf. cross-prompt @ steps4 = {p['pick_up vs t2_pour']:.6f})")

    print("\n[T5] steps=40 — conditioning trend + does seed still dominate?")
    t5 = {}
    t5["cross-prompt @ steps40"] = report("pick vs pour @ steps40",
                                          acts["t5_st40_pick"],
                                          acts["t5_st40_pour"])
    t5["seed @ steps40"] = report("seed0 vs seed1 @ steps40",
                                  acts["t5_st40_pick"], acts["t5_st40_s1"])
    t5["steps16 vs steps40"] = report("steps16 vs steps40 (pick)",
                                      acts["t4_st16_pick"], acts["t5_st40_pick"])
    print("    cross-prompt trend:  steps4={:.6f} -> steps16={:.6f} -> "
          "steps40={:.6f}".format(p["pick_up vs t2_pour"],
                                  st["cross-prompt @ steps16"],
                                  t5["cross-prompt @ steps40"]))
    print("    seed trend:          steps4={:.6f} -> steps40={:.6f}".format(
        float(np.mean(list(s.values()))), t5["seed @ steps40"]))
    print(f"    seed/prompt @ steps40 = "
          f"{t5['seed @ steps40'] / t5['cross-prompt @ steps40']:.1f}x")

    print("\n[3] per-dimension mean|diff|, pick_up vs pour (g=3.0)")
    d = np.abs(ref - acts["t2_pour"]).mean(axis=0)
    names = ["pos_x", "pos_y", "pos_z",
             "r1x", "r1y", "r1z", "r2x", "r2y", "r2z", "grip"]
    for n, v in zip(names, d):
        print(f"    {n:6s} {v:.6f}")

    print("\n=== VERDICT ===")
    axes = [("determinism floor", floor),
            ("guidance  (T1)", float(np.mean(list(g.values())))),
            ("frame     (W3)", float(np.mean(w3)) if w3 else float("nan")),
            ("prompt    (T2)", float(np.mean(list(p.values())))),
            ("steps     (T4)", st["steps4 vs steps16 (pick)"]),
            ("SEED      (T3)", float(np.mean(list(s.values()))))]
    for label, val in axes:
        print(f"  {label:<20s} mean|diff|={val:.6f}   "
              f"= {100.0 * val / spread:6.2f}% of mean|action|={spread:.4f}")
    seed_m = float(np.mean(list(s.values())))
    cond = [np.mean(list(g.values())), np.mean(list(p.values()))] + \
           ([np.mean(w3)] if w3 else [])
    cond_m = float(np.mean(cond))
    print(f"\n  within-chunk step-to-step = {within:.6f} "
          f"({100.0 * within / spread:.1f}% of mean|action|)")
    print(f"  SEED / conditioning ratio = {seed_m / cond_m:.1f}x")
    print(f"  within-chunk / prompt     = "
          f"{within / p['pick_up vs t2_pour']:.1f}x")


if __name__ == "__main__":
    main()
