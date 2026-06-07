"""Recompute the cross-sample stats + pass_report from a failed step0 log.

The probe_v1 run crashed after all per-pair forwards but before writing JSON,
due to a bug in singular_spectrum_summary on a (1, N, 80) tensor. We don't
need to re-run the GPU forwards — we just re-parse the log lines and recompute
the aggregate. The fixed probe script handles it correctly going forward.
"""
import re, json, sys
from collections import defaultdict

LOG = "/test1208/zw/ctm_emotion_tts/logs/svd_v2/step0_v1.log"

# Parse log lines like:
# [Happy/0002_text00063] ||r||/||v_neu||=0.485 cos(v_e,v_n)=0.882 cos(A,r)=0.267 cos(B,v_n)=0.889
pat = re.compile(
    r"\[(?P<emo>\w+)/(?P<key>[\w]+)\] "
    r"\|\|r\|\|/\|\|v_neu\|\|=(?P<r>[\d.+-]+) "
    r"cos\(v_e,v_n\)=(?P<cos_vn>[\d.+-]+) "
    r"cos\(A,r\)=(?P<cos_Ar>[\d.+-]+) "
    r"cos\(B,v_n\)=(?P<cos_Bn>[\d.+-]+)"
)

per_pair = []
with open(LOG) as f:
    for line in f:
        m = pat.search(line)
        if m:
            per_pair.append({
                "emotion": m["emo"],
                "key": m["key"],
                "r_over_vneu": float(m["r"]),
                "cos_v_emo_v_neu": float(m["cos_vn"]),
                "cos_svdA_trueR": float(m["cos_Ar"]),
                "cos_svdB_vneu": float(m["cos_Bn"]),
            })

# Aggregate
by_emo = defaultdict(list)
for p in per_pair:
    by_emo[p["emotion"]].append(p)

mean = lambda xs: (sum(xs) / len(xs)) if xs else None

per_emotion_stats = {}
for emo, items in by_emo.items():
    per_emotion_stats[emo] = {
        "n": len(items),
        "mean_r_over_vneu": mean([p["r_over_vneu"] for p in items]),
        "mean_cos_v_emo_v_neu": mean([p["cos_v_emo_v_neu"] for p in items]),
        "mean_cos_svdA_trueR": mean([p["cos_svdA_trueR"] for p in items]),
        "mean_cos_svdB_vneu": mean([p["cos_svdB_vneu"] for p in items]),
    }

overall = {
    "n": len(per_pair),
    "mean_r_over_vneu": mean([p["r_over_vneu"] for p in per_pair]),
    "mean_cos_v_emo_v_neu": mean([p["cos_v_emo_v_neu"] for p in per_pair]),
    "mean_cos_svdA_trueR": mean([p["cos_svdA_trueR"] for p in per_pair]),
    "mean_cos_svdB_vneu": mean([p["cos_svdB_vneu"] for p in per_pair]),
}

pass_report = {
    "test1_energy_pass": overall["mean_r_over_vneu"] >= 0.05,
    "test1_energy_mean": overall["mean_r_over_vneu"],
    "test4_A_aligns_r_pass": overall["mean_cos_svdA_trueR"] >= 0.20,
    "test4_A_aligns_r_mean": overall["mean_cos_svdA_trueR"],
    "test5_B_aligns_vneu_pass": overall["mean_cos_svdB_vneu"] >= 0.80,
    "test5_B_aligns_vneu_mean": overall["mean_cos_svdB_vneu"],
}

out = {
    "source": "recomputed from log",
    "log_path": LOG,
    "n_pairs": len(per_pair),
    "per_pair": per_pair,
    "per_emotion_stats": per_emotion_stats,
    "overall": overall,
    "pass_report": pass_report,
}

OUT = "/test1208/zw/ctm_emotion_tts/outputs/svd_v2/step0/probe_v1_recovered.json"
with open(OUT, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"wrote {OUT}")
print(json.dumps({"overall": overall, "pass_report": pass_report}, indent=2))
