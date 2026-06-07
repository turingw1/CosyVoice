"""Quick health-check on a train.log.jsonl.

Reads the JSONL log produced by train_svd_flow.py and prints:
  - first / last / mid values per loss
  - trend (decreasing? plateau? diverging?)
  - any NaN/inf occurrences
  - simple Pass/Warn/Fail verdict per training health rubric
"""
import argparse
import json
import math
import sys
from statistics import mean


def trend(xs, ratio=0.5):
    """Compare last-N% mean to first-N% mean. Returns ratio last/first."""
    if not xs:
        return None
    n = max(1, int(len(xs) * ratio))
    a = mean(xs[:n])
    b = mean(xs[-n:])
    if a == 0:
        return float("inf") if b > 0 else 1.0
    return b / a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="train.log.jsonl path")
    args = ap.parse_args()

    rows = []
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print("no log rows found")
        sys.exit(1)

    keys = ["loss", "L_fm", "L_sem", "L_emo",
            "cos_A_target_mean", "A_over_v_ratio", "B_over_v_ratio",
            "grad_norm"]

    print(f"{'metric':<20s} {'first':>12s} {'mid':>12s} {'last':>12s} {'last/first':>12s}  verdict")
    for k in keys:
        xs = [r[k] for r in rows if k in r]
        if not xs:
            continue
        nan_count = sum(1 for x in xs if not math.isfinite(x))
        if nan_count > 0:
            print(f"{k:<20s} {'-':>12s} {'-':>12s} {'-':>12s} {'-':>12s}  FAIL ({nan_count} non-finite)")
            continue
        mid = xs[len(xs) // 2]
        ratio = trend(xs)
        first = xs[0]; last = xs[-1]

        if k in ("loss", "L_fm", "L_sem", "L_emo", "grad_norm"):
            # want to decrease; "last/first" near 1 = no progress, < 0.8 = good
            if ratio < 0.7:
                verdict = "GOOD"
            elif ratio < 0.95:
                verdict = "OK"
            elif ratio < 1.1:
                verdict = "PLATEAU"
            else:
                verdict = "WARN (rising)"
        elif k == "cos_A_target_mean":
            # want to increase from start
            if last - first > 0.2:
                verdict = "GOOD"
            elif last - first > 0.05:
                verdict = "OK"
            elif last - first > -0.05:
                verdict = "PLATEAU"
            else:
                verdict = "WARN (dropping)"
        else:
            # ratios — just informational
            verdict = "info"
        print(f"{k:<20s} {first:>12.4f} {mid:>12.4f} {last:>12.4f} {ratio:>12.3f}  {verdict}")

    # Overall
    print()
    if any(not math.isfinite(r.get("loss", 0)) for r in rows):
        print("[OVERALL] FAIL: non-finite loss observed.")
    else:
        l_fm_xs = [r["L_fm"] for r in rows if "L_fm" in r]
        cos_xs = [r.get("cos_A_target_mean") for r in rows if "cos_A_target_mean" in r]
        ok_fm = (l_fm_xs and trend(l_fm_xs) < 1.05)
        ok_emo = (cos_xs and cos_xs[-1] > cos_xs[0] - 0.02)
        if ok_fm and ok_emo:
            print("[OVERALL] PASS — L_fm not diverging and L_emo direction not collapsing.")
        elif ok_fm:
            print("[OVERALL] PARTIAL — L_fm stable but L_emo direction not improving.")
        else:
            print("[OVERALL] FAIL — training unstable; investigate.")
    print(f"[INFO] {len(rows)} steps logged; last step = {rows[-1].get('step')}")


if __name__ == "__main__":
    main()
