#!/usr/bin/env python3
"""W3 cross-frame frame-sensitivity repro (PROBLEMS §L).

Reproduces the finding: the Cosmos3-Edge policy single-frame turnkey path is
FRAME-INSENSITIVE — it outputs nearly identical action chunks for different
input frames of the same episode, while each chunk still has coherent
within-frame displacement. This is a vision-only single-frame conditioning
capability boundary, NOT an engine bug.

Evidence (committed, KB-scale):
  w3_actions/f{20,60,100,140}_act.json — cosmos3_policy_inference output on 4
    frames of lerobot/droid_100 ep0 ("Put the marker in the pot"), goat,
    engines/cosmos3_policy_droid, --steps 4 --seed 0. Each is the raw normalized
    action chunk [16,10] = [eef_pos(3) + rot6d(6) + gripper(1)].
  droid_action_stats.json — cosmos-framework quantile stats (q01/q99), from
    cosmos_framework/data/generator/action/normalizer_stats/droid_lerobot_stats.json

Regeneration (frames + video are gitignored, rebuildable from lerobot/droid_100):
  # 1. pull ep0 exterior video (aggregated file, all 100 eps concatenated):
  #    hf-mirror.com/datasets/lerobot/droid_100/resolve/main/videos/
  #      observation.images.exterior_image_1_left/chunk-000/file-000.mp4
  # 2. extract frames (ep0 = video frames 0-165 @15fps; -ss AFTER -i = frame-accurate):
  #    ffmpeg -i file-000.mp4 -ss {1.333,4.0,6.667,9.333} -vframes 1 f{20,60,100,140}.png
  # 3. on goat, per frame:
  #    cosmos3_policy_inference --engineDir engines/cosmos3_policy_droid
  #      --image f<N>.png --prompt "Put the marker in the pot"
  #      --output f<N>_act.json --steps 4 --seed 0 --iters 1 --warmup 1

Usage:
    .venv/bin/python eval/cosmos3/w3_frame_sensitivity.py
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIONS = os.path.join(HERE, "w3_actions")
STATS = os.path.join(HERE, "droid_action_stats.json")
FRAMES = [20, 60, 100, 140]
# quantile denormalize: physical = 0.5*(norm+1)*(q99-q01) + q01
POS = slice(0, 3)      # eef position delta (meters after denorm)
ROT6D = slice(3, 9)    # 6D rotation (two 3-vectors, Gram-Schmidt -> R)
GRIP = 9               # gripper


def denormalize(a, q01, q99):
    return 0.5 * (a + 1.0) * (q99 - q01) + q01


def main():
    stats = json.load(open(STATS))
    q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
    acts = {n: np.array(json.load(open(os.path.join(ACTIONS, f"f{n}_act.json")))
                        ["action"], dtype=float) for n in FRAMES}

    print("=== W3 frame-sensitivity repro (PROBLEMS §L) ===")
    print("shapes:", {n: a.shape for n, a in acts.items()})

    # --- Finding 1: cross-frame near-identity (frame-insensitive) ---
    print("\n[1] cross-frame action diff (normalized space; ~0 => frame-insensitive)")
    diffs = []
    for a, b in itertools.combinations(FRAMES, 2):
        d = np.abs(acts[a] - acts[b])
        diffs.append(d.mean())
        print(f"    mean|f{a}-f{b}| = {d.mean():.4f}   max|.| = {d.max():.4f}")
    print(f"    => cross-frame mean diff range [{min(diffs):.4f}, {max(diffs):.4f}]"
          f"  (§L: ≈0.009, i.e. frames barely change the output)")

    # --- Finding 2: within-frame coherent displacement (not pure noise) ---
    print("\n[2] within-frame denormalized eef-position trajectory (meters)")
    for n in FRAMES:
        pos = denormalize(acts[n], q01, q99)[:, POS]
        net = pos[-1] - pos[0]
        step_std = np.diff(pos, axis=0).std(axis=0)
        print(f"    f{n}: step0={np.round(pos[0],4)} step15={np.round(pos[-1],4)} "
              f"net={np.round(net,4)} step-diff-std={np.round(step_std,4)}")

    # --- Finding 3: rot6d validity + gripper out-of-range ---
    print("\n[3] rot6d validity (|v| should ≈1) + gripper range")
    a0 = denormalize(acts[FRAMES[0]], q01, q99)
    v1, v2 = a0[0, 3:6], a0[0, 6:9]
    grip = denormalize(acts[FRAMES[0]], q01, q99)[:, GRIP]
    print(f"    rot6d step0 |v1|={np.linalg.norm(v1):.3f} |v2|={np.linalg.norm(v2):.3f}")
    print(f"    gripper denorm range [{grip.min():.3f}, {grip.max():.3f}] "
          f"(>1 => invert/denorm convention mismatch, see §L2)")

    print("\n=> CONCLUSION: single-frame turnkey policy is frame-insensitive "
          "(cross-frame diff ≈0.009) yet produces coherent within-frame "
          "displacement; vision-only single-frame conditioning boundary, not an "
          "engine bug. See docs/PROBLEMS.md §L.")


if __name__ == "__main__":
    main()
