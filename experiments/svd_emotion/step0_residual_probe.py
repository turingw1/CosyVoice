"""Step 0: zero-cost residual structure probe on existing CosyVoice3.

Goal (see plan.md section 10):
Before any training, verify on real ESD parallel data whether the v2 hypothesis
is even plausible. Three checks per pair (Neutral vs target emotion):

  Test 1 (energy):       ||v_emo - v_neu||_F / ||v_neu||_F  >= 5%
                         If much smaller, flow ignores emotion conditioning and
                         the SVD path is dead.

  Test 2 (consistency):  cos(time_avg(r_i), time_avg(r_j))  for same-emotion >> for different
                         If they're equal, emotion direction is not stable.

  Test 3 (low-rank):     SVD of stacked residuals  ->  top-10 captures >= 50% energy
                         If much smaller, k=16 won't suffice.

Plus a bonus diagnostic:
  Test 4 (svd-alignment): cos(SVD-residual of v_emo,  true r = v_emo - v_neu)
                         Tells us whether the SVD residual of v_emo already
                         "naturally" points along the emotion direction even
                         without any training intervention.

All forwards use the wav's own speech tokens (extracted by the speech tokenizer)
as flow inputs, so we are probing the trained flow's response to real emo vs
real neu utterances with no LLM generation step.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/test1208/zw/ctm_emotion_tts")
COSYVOICE_REPO = ROOT / "repos" / "CosyVoice"
MODEL_DIR = Path("/home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512")

# sys.path is set by importing modules; do it explicitly so this script can be
# run directly without `pip install -e .`
sys.path.insert(0, str(COSYVOICE_REPO))
sys.path.insert(0, str(COSYVOICE_REPO / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel                              # noqa: E402
from cosyvoice.utils.mask import make_pad_mask                              # noqa: E402

from experiments.svd_emotion.svd_decompose import (                         # noqa: E402
    cosine_flat,
    energy_ratio,
    singular_spectrum_summary,
    svd_decompose,
)


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


def build_model_input(cosyvoice, wav_path: str, text: str) -> dict:
    """frontend_zero_shot extracts the wav's own speech tokens, mel, and x-vec.

    For Step 0 we feed the wav as both reference and content source, so the
    flow sees real (token, mel) pairs from the recording.
    """
    return cosyvoice.frontend.frontend_zero_shot(
        text, "", wav_path, cosyvoice.sample_rate, ""
    )


@torch.inference_mode()
def flow_estimator_at_t(cosyvoice, model_input: dict, t_value: float,
                        z: torch.Tensor | None = None) -> dict:
    """Run a single DiT estimator forward at flow time s = t_value.

    Uses the wav's own mel as x_1 (CFM training-style perturbation), then calls
    the DiT estimator exactly once. Returns the predicted velocity v_hat (the
    quantity that L_fm trains against) plus diagnostic tensors.
    """
    device = cosyvoice.model.device
    flow = cosyvoice.model.flow

    token = model_input["flow_prompt_speech_token"].to(device)              # (1, N_tok)
    feat = model_input["prompt_speech_feat"].to(device)                     # (1, K_feat, 80)
    embedding = model_input["flow_embedding"].to(device)

    # speaker projection (matches inference path)
    embedding = F.normalize(embedding, dim=1)
    spks = flow.spk_embed_affine_layer(embedding)                          # (1, 80)

    # token -> embedding -> pre_lookahead -> repeat_interleave for mel rate
    mask_tok = (~make_pad_mask(torch.tensor([token.shape[1]], device=device))).unsqueeze(-1).to(spks)
    token_emb = flow.input_embedding(torch.clamp(token, min=0)) * mask_tok
    h = flow.pre_lookahead_layer(token_emb)                                 # (1, N_tok, 80)
    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)                    # (1, ~K, 80)

    # Align h length with feat length (they can disagree by 1-2 frames)
    K = min(h.shape[1], feat.shape[1])
    h = h[:, :K]
    x_1 = feat[:, :K].transpose(1, 2).contiguous()                          # (1, 80, K)

    mu = h.transpose(1, 2).contiguous()                                     # (1, 80, K)
    cond = torch.zeros_like(x_1)                                            # zero out prompt portion
    flow_mask = (~make_pad_mask(torch.tensor([K], device=device))).to(spks).unsqueeze(1)

    # CFM perturbation: y = (1 - (1-σ)t) z + t x_1
    sigma_min = 1e-6
    if z is None:
        z = torch.randn_like(x_1)
    t_scalar = float(t_value)
    y = (1.0 - (1.0 - sigma_min) * t_scalar) * z + t_scalar * x_1
    t_tensor = torch.full([1], t_scalar, device=device, dtype=spks.dtype)
    u_true = x_1 - (1.0 - sigma_min) * z

    v_hat = flow.decoder.forward_estimator(
        y, flow_mask, mu, t_tensor, spks, cond, streaming=False
    ).float()

    return {
        "v_hat": v_hat,         # (1, 80, K)
        "u_true": u_true.float(),
        "K": K,
        "y": y.float(),
        "z": z.float(),
    }


# ---------------------------------------------------------------------------
# Main probe loop
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL from build_esd_manifest.py")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--emotions", default="Happy,Sad,Angry,Surprise",
                    help="Comma-separated target emotions (each paired with Neutral)")
    ap.add_argument("--per_emotion", type=int, default=5,
                    help="Number of (neu, emo) pairs to sample per emotion")
    ap.add_argument("--t_value", type=float, default=0.5, help="Flow sampling time s")
    ap.add_argument("--k_svd", type=int, default=16, help="SVD top-k cutoff for decomposition test")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu", type=str, default="5", help="CUDA_VISIBLE_DEVICES value (default 5)")
    ap.add_argument("--filter_lang", type=str, default="",
                    help="If set ('en' or 'zh'), restrict to that language")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]

    # ------ Load manifest ------
    records = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if args.filter_lang and r.get("lang") != args.filter_lang:
                continue
            records.append(r)
    print(f"[INFO] manifest contains {len(records)} records "
          f"(filter_lang={args.filter_lang or 'all'})")

    selected = []
    for emo in target_emotions:
        cands = [r for r in records
                 if "Neutral" in r["emotions"] and emo in r["emotions"]]
        random.shuffle(cands)
        for r in cands[:args.per_emotion]:
            selected.append((emo, r))
    print(f"[INFO] selected {len(selected)} pairs across {len(target_emotions)} emotions")

    # ------ Load CosyVoice3 ------
    print(f"[INFO] loading CosyVoice3 from {MODEL_DIR}")
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR), fp16=False)
    device = cosyvoice.model.device
    print(f"[INFO] device={device}")

    # ------ Run forwards ------
    per_pair = []
    avg_r_by_emo = {emo: [] for emo in target_emotions}      # list of (80,) cpu tensors

    for emo, rec in selected:
        try:
            mi_neu = build_model_input(cosyvoice, rec["emotions"]["Neutral"], rec["text"])
            mi_emo = build_model_input(cosyvoice, rec["emotions"][emo], rec["text"])

            res_neu = flow_estimator_at_t(cosyvoice, mi_neu, args.t_value)
            res_emo = flow_estimator_at_t(cosyvoice, mi_emo, args.t_value)

            v_neu, v_emo = res_neu["v_hat"], res_emo["v_hat"]
            K_min = min(v_neu.shape[-1], v_emo.shape[-1])
            v_neu_c = v_neu[..., :K_min]
            v_emo_c = v_emo[..., :K_min]
            r = v_emo_c - v_neu_c                            # (1, 80, K_min)

            # Test 2 (consistency): per-channel time average residual
            r_avg = r[0].mean(dim=-1).cpu()                  # (80,)

            # Test 3 spectrum (single sample)
            spec_v_emo = singular_spectrum_summary(v_emo_c[0])
            spec_r     = singular_spectrum_summary(r[0])

            # Test 4: do svd_decompose on v_emo, compare residual A vs true r
            A, B, _ = svd_decompose(v_emo_c, args.k_svd)
            cos_A_r   = cosine_flat(A[0], r[0])
            cos_B_neu = cosine_flat(B[0], v_neu_c[0])        # ideally B ~ semantic baseline ~ v_neu

            entry = {
                "key": rec["key"],
                "speaker": rec["speaker"],
                "emotion": emo,
                "lang": rec.get("lang"),
                "K_neu": int(v_neu.shape[-1]),
                "K_emo": int(v_emo.shape[-1]),
                "K_min": int(K_min),
                "v_neu_norm": float(torch.linalg.vector_norm(v_neu_c).item()),
                "v_emo_norm": float(torch.linalg.vector_norm(v_emo_c).item()),
                "r_norm": float(torch.linalg.vector_norm(r).item()),
                "test1_r_over_vneu": energy_ratio(r, v_neu_c),
                "test1_r_over_vemo": energy_ratio(r, v_emo_c),
                "cos_v_emo_v_neu": cosine_flat(v_emo_c, v_neu_c),
                "test3_v_emo_singular_spectrum": spec_v_emo,
                "test3_r_singular_spectrum": spec_r,
                "test4_cos_svdA_trueR": cos_A_r,
                "test4_cos_svdB_vneu":  cos_B_neu,
            }
            per_pair.append(entry)
            avg_r_by_emo[emo].append(r_avg)
            print(f"[{emo}/{rec['key']}] "
                  f"||r||/||v_neu||={entry['test1_r_over_vneu']:.3f} "
                  f"cos(v_e,v_n)={entry['cos_v_emo_v_neu']:.3f} "
                  f"cos(A,r)={cos_A_r:.3f} "
                  f"cos(B,v_n)={cos_B_neu:.3f}")
        except Exception as e:
            print(f"[ERR] {rec['key']} {emo}: {type(e).__name__}: {e}")
            continue

    # ------ Cross-sample stats: within-emotion vs between-emotion ------
    within = {}
    for emo, vecs in avg_r_by_emo.items():
        if len(vecs) < 2:
            within[emo] = {"n": len(vecs), "note": "insufficient samples"}
            continue
        cs = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cs.append(F.cosine_similarity(vecs[i].unsqueeze(0),
                                              vecs[j].unsqueeze(0), dim=-1).item())
        within[emo] = {
            "n": len(vecs),
            "mean_cos": sum(cs) / len(cs),
            "min_cos":  min(cs),
            "max_cos":  max(cs),
        }

    between = {}
    emos_with_data = [e for e in target_emotions if len(avg_r_by_emo[e]) >= 1]
    for i, e1 in enumerate(emos_with_data):
        for e2 in emos_with_data[i + 1:]:
            cs = []
            for v1 in avg_r_by_emo[e1]:
                for v2 in avg_r_by_emo[e2]:
                    cs.append(F.cosine_similarity(v1.unsqueeze(0),
                                                  v2.unsqueeze(0), dim=-1).item())
            if cs:
                between[f"{e1}_vs_{e2}"] = {
                    "n_pairs": len(cs),
                    "mean_cos": sum(cs) / len(cs),
                }

    # ------ Stacked residual SVD (across all samples, time-averaged) ------
    all_r_avg = [v for vecs in avg_r_by_emo.values() for v in vecs]
    if all_r_avg:
        R = torch.stack(all_r_avg, dim=0)                    # (N, 80)
        stacked_spec = singular_spectrum_summary(R)
    else:
        stacked_spec = {}

    # ------ Pass / fail summary against plan.md gating thresholds ------
    mean_ratio = (sum(p["test1_r_over_vneu"] for p in per_pair) /
                  max(len(per_pair), 1))
    mean_within = (sum(w["mean_cos"] for w in within.values() if "mean_cos" in w) /
                   max(sum(1 for w in within.values() if "mean_cos" in w), 1))
    mean_between = (sum(b["mean_cos"] for b in between.values()) /
                    max(len(between), 1)) if between else 0.0
    top_k_frac = stacked_spec.get("top_8_energy_frac",
                                  stacked_spec.get("top_4_energy_frac", 0.0))

    pass_report = {
        "test1_energy_pass":      mean_ratio >= 0.05,
        "test1_energy_mean":      mean_ratio,
        "test2_direction_pass":   (mean_within - mean_between) >= 0.10,
        "test2_within_mean":      mean_within,
        "test2_between_mean":     mean_between,
        "test3_lowrank_pass":     top_k_frac >= 0.50,
        "test3_top_energy_frac":  top_k_frac,
    }

    out_data = {
        "config": {
            "manifest": args.manifest,
            "t_value": args.t_value,
            "k_svd": args.k_svd,
            "per_emotion": args.per_emotion,
            "emotions": target_emotions,
            "filter_lang": args.filter_lang,
            "seed": args.seed,
            "gpu": args.gpu,
        },
        "n_pairs_processed": len(per_pair),
        "per_pair": per_pair,
        "within_emotion_consistency": within,
        "between_emotion_consistency": between,
        "stacked_residual_spectrum": stacked_spec,
        "pass_report": pass_report,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print("---")
    print(f"[OK] wrote {args.out}")
    print("Pass report:")
    print(json.dumps(pass_report, indent=2))


if __name__ == "__main__":
    main()
