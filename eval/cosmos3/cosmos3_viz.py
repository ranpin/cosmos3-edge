#!/usr/bin/env python3
"""Phase W figures (site/cosmos3.html): cross-generation accuracy + policy frame-insensitivity.

Both figures derive every number from committed data — nothing is hardcoded:
  * Cosmos3-Edge 2B accuracy is re-scored here from the raw n=184 outputs
    (cosmos3_edge_fp16_outputs_n184.json) against ref_cr2/manifest_n184.json,
    reusing score_mc's parse_choice / wilson / mcnemar_exact so it is
    methodologically identical to RESULTS.json.
  * CR2-8B FP16/INT8/INT4 numbers come from ref_cr2/driving_results.json.
  * Policy frame-insensitivity comes from eval/policy/w3_actions/f{20,60,100,140}_act.json.

ref_cr2/ = verbatim copies of the CR2-side reference data from
ranpin/alpamayo-edge (manifest / driving RESULTS / CR2 FP16 n184 outputs /
score_mc.py) so these figures re-render self-contained; canonical CR2 data
lives in alpamayo-edge.

Run:  python eval/cosmos3/cosmos3_viz.py   (FIG_THEME=dark for the dark variant)
Out:  site/figures/cosmos3_crossgen.png
      site/figures/cosmos3_policy_frame_insensitive.png   (+ site/figures/dark/)
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))     # paper_style
sys.path.insert(0, os.path.join(HERE, "ref_cr2"))  # score_mc (CR2 reference copy)
from paper_style import (apply, bar_kw, label_bars, C, DARK, ANNOT, C4, C8,
                         CF, WIDTH_FULL)  # noqa: E402
from score_mc import load_run, mcnemar_exact, parse_choice, wilson  # noqa: E402

apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(ROOT, "site", "figures", "dark") if DARK else os.path.join(ROOT, "site", "figures")
os.makedirs(OUT, exist_ok=True)
C3 = C("#7c3aed")  # Cosmos3-Edge violet (distinct from the CR2-8B trio)


def score_cosmos3():
    man = json.load(open(os.path.join(HERE, "ref_cr2", "manifest_n184.json")))
    items = man["items"]
    gens = load_run(os.path.join(HERE, "cosmos3_edge_fp16_outputs_n184.json"))
    assert len(gens) == len(items), f"{len(gens)} vs {len(items)}"
    choices = [parse_choice(g, e["options"]) for e, g in zip(items, gens)]
    correct = [c == e["answer"] for c, e in zip(choices, items)]
    k = sum(correct)
    n = len(items)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "acc": 100 * k / n, "ci": (lo, hi),
            "unparsed": sum(1 for c in choices if c is None),
            "correct": correct}


def fig_crossgen():
    drv = json.load(open(os.path.join(HERE, "ref_cr2", "driving_results.json")))
    c3 = score_cosmos3()
    fp16 = drv["runs"]["fp16"]

    # McNemar of Cosmos3 vs CR2-8B FP16 (same 184 items, same manifest)
    ref = fp16 and None  # placeholder; recompute from per-item correctness below
    man = json.load(open(os.path.join(HERE, "ref_cr2", "manifest_n184.json")))
    items = man["items"]
    fp16_gens = load_run(os.path.join(HERE, "ref_cr2",
                                      "orin_fp16_outputs_n184.json"))
    fp16_choices = [parse_choice(g, e["options"]) for e, g in zip(items, fp16_gens)]
    fp16_correct = [c == e["answer"] for c, e in zip(fp16_choices, items)]
    b = sum(1 for f, q in zip(fp16_correct, c3["correct"]) if f and not q)
    cc = sum(1 for f, q in zip(fp16_correct, c3["correct"]) if q and not f)
    p = mcnemar_exact(b, cc)

    names = ["CR2-8B\nFP16", "CR2-8B\nINT8", "CR2-8B\nINT4", "Cosmos3-Edge\n2B FP16"]
    accs = [drv["runs"]["fp16"]["overall"], drv["runs"]["int8"]["overall"],
            drv["runs"]["int4"]["overall"], c3["acc"]]
    cis = [drv["runs"]["fp16"]["ci"], drv["runs"]["int8"]["ci"],
           drv["runs"]["int4"]["ci"], list(c3["ci"])]
    cols = [CF, C8, C4, C3]
    errs = [[a - lo for a, (lo, hi) in zip(accs, cis)],
            [hi - a for a, (lo, hi) in zip(accs, cis)]]

    fig, ax = plt.subplots(figsize=(WIDTH_FULL * 0.62, 3.4))
    x = np.arange(len(names))
    bars = ax.bar(x, accs, width=0.62, color=cols, **bar_kw())
    ax.errorbar(x, accs, yerr=errs, fmt="none", ecolor=C("#333"), elinewidth=1,
                capsize=4, capthick=1, zorder=5)
    ax.axhline(drv["chance_pct"], color=C("#999"), ls=":", lw=1)
    ax.text(len(names) - 0.42, drv["chance_pct"] + 1.0, f"随机 {drv['chance_pct']}%",
            ha="right", fontsize=ANNOT, color=C("#777"))
    label_bars(ax, bars, fmt="{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("8 选 1 正确率 (%)")
    ax.set_ylim(0, 68)
    ax.set_title("驾驶域 n=184 · 误差棒 = Wilson 95%CI")
    # cross-gen gap annotation
    gap = accs[0] - accs[3]
    ax.annotate("", xy=(3, accs[3] + errs[1][3] + 3.5), xytext=(0, accs[0] + errs[1][0] + 3.5),
                arrowprops=dict(arrowstyle="<->", color=C("#555"), lw=0.9))
    ax.text(1.5, max(accs[0], accs[3]) + errs[1][0] + 5.0,
            f"跨代差 −{gap:.1f} pts\n(McNemar b/c={b}/{cc}, p={p:.4f})",
            ha="center", fontsize=ANNOT, color=C("#333"))
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT, "cosmos3_crossgen.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  cosmos3 re-scored: {c3['acc']:.1f}% [{c3['ci'][0]:.1f},{c3['ci'][1]:.1f}] "
          f"k={c3['k']}/{c3['n']} unparsed={c3['unparsed']}")
    print(f"  vs CR2-8B FP16: b/c={b}/{cc} p={p:.4f}")


def load_actions():
    acts = {}
    for n in (20, 60, 100, 140):
        d = json.load(open(os.path.join(ROOT, "eval", "policy", "w3_actions", f"f{n}_act.json")))
        acts[n] = np.asarray(d["action"], dtype=float)  # [16,10]
    return acts


def fig_policy(acts):
    frames = sorted(acts)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 3.3),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # Panel A: within-frame cumulative end-effector displacement (dims 0-2 are
    # per-step position deltas) — each frame shows coherent net displacement.
    steps = np.arange(acts[frames[0]].shape[0])
    for i, n in enumerate(frames):
        pos = np.cumsum(acts[n][:, 0:3], axis=0)
        disp = np.linalg.norm(pos, axis=1)
        axA.plot(steps, disp, marker="o", ms=3, lw=1.2, label=f"frame {n}")
    axA.set_xlabel("扩散动作步 (16 步 chunk)")
    axA.set_ylabel("累计位移范数 (归一化)")
    axA.set_title("帧内：每帧都有连贯净位移")
    axA.legend(ncol=2)

    # Panel B: pairwise cross-frame mean|diff| over all 160 action values,
    # against the within-frame displacement scale — cross-frame diff is ~2 orders
    # smaller than the motion the chunk itself produces.
    pairs = list(itertools.combinations(frames, 2))
    diffs = [np.abs(acts[a] - acts[b]).mean() for a, b in pairs]
    within = np.mean([np.linalg.norm(np.cumsum(acts[n][:, 0:3], axis=0)[-1]) for n in frames])
    xs = np.arange(len(pairs))
    bars = axB.bar(xs, diffs, width=0.6, color=C("#94a3b8"), **bar_kw())
    label_bars(axB, bars, fmt="{:.4f}")
    axB.axhline(within, color=C3, ls="--", lw=1.2)
    axB.text(len(pairs) - 0.45, within, f"帧内位移范数 ≈ {within:.2f}",
             ha="right", va="bottom", fontsize=ANNOT, color=C3)
    axB.set_xticks(xs)
    axB.set_xticklabels([f"{a}v{b}" for a, b in pairs], rotation=0)
    axB.set_ylabel("跨帧 mean|Δaction| (160 值)")
    axB.set_title("跨帧：换帧输出几乎不变")
    axB.set_ylim(0, max(within * 1.25, max(diffs) * 1.4))

    fig.tight_layout()
    out = os.path.join(OUT, "cosmos3_policy_frame_insensitive.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  cross-frame mean|diff|: {[f'{d:.4f}' for d in diffs]}")
    print(f"  within-frame displacement norm mean: {within:.3f}")


if __name__ == "__main__":
    fig_crossgen()
    fig_policy(load_actions())
