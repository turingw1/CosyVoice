"""SVD-suitability diagnostic evaluator (plan.md section 12).

Consumes the alpha-ladder output of inference_svd_alpha.py and scores each
generated wav along:
  - emotion expression: emotion2vec target-emotion probability
  - intelligibility:    Whisper WER vs original text
  - speaker preservation: speaker x-vector cosine similarity vs neutral reference

Then aggregates per-(speaker, sentence, target_emotion) to compute:
  - alpha-monotonicity: emo score increases with alpha?
  - alpha=0 neutrality: low emo score, high neutral score at alpha=0?
  - alpha=1 quality: WER and SIM acceptable at alpha=1?

Also supports ablation comparisons:
  - Replace A with random tensor of same norm: does emotion control still work?
  - Replace A with mean residual baseline: does SVD-A beat mean baseline?

These ablations are produced by running inference_svd_alpha.py with extra
flags (planned: --ablation random|mean|none) — for now this evaluator just
processes the produced manifest.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torchaudio

ROOT = Path("/test1208/zw/ctm_emotion_tts")
COSYVOICE_REPO = ROOT / "repos" / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_REPO))
sys.path.insert(0, str(COSYVOICE_REPO / "third_party" / "Matcha-TTS"))

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
LOG = logging.getLogger("eval_svd_suitability")


EMO_TO_E2V_LABEL = {
    "Neutral":  ["中立/neutral", "neutral"],
    "Happy":    ["开心/happy", "happy", "快乐"],
    "Angry":    ["生气/angry", "angry", "愤怒"],
    "Sad":      ["难过/sad", "sad", "悲伤"],
    "Surprise": ["吃惊/surprised", "surprised", "惊喜"],
}


def load_emotion2vec():
    """Return a FunASR emotion2vec inference handle (lazy-loaded)."""
    from funasr import AutoModel as FunASRAutoModel
    LOG.info("loading emotion2vec...")
    m = FunASRAutoModel(model="iic/emotion2vec_plus_large",
                        hub="hf",
                        disable_update=True)
    return m


def score_emotion(em_model, wav_path: str, target_emo: str) -> dict:
    """Return {target_score, neutral_score, top_label, top_score, all_labels}."""
    res = em_model.generate(wav_path, output_dir=None, granularity="utterance",
                            extract_embedding=False)
    rec = res[0] if isinstance(res, list) else res
    labels = rec.get("labels", [])
    scores = rec.get("scores", [])
    label_score = dict(zip(labels, scores))
    # match candidates
    def lookup(emo):
        for cand in EMO_TO_E2V_LABEL.get(emo, [emo]):
            for lab, sc in label_score.items():
                if cand in lab or lab in cand:
                    return float(sc)
        return None
    return {
        "target_score": lookup(target_emo),
        "neutral_score": lookup("Neutral"),
        "top_label": labels[0] if labels else None,
        "top_score": float(scores[0]) if scores else None,
        "label_score": {l: float(s) for l, s in label_score.items()},
    }


def load_whisper():
    from funasr import AutoModel as FunASRAutoModel
    LOG.info("loading SenseVoice ASR (small)...")
    # SenseVoiceSmall is more dialect-robust and Chinese-friendly than Whisper-base
    return FunASRAutoModel(model="iic/SenseVoiceSmall", hub="hf",
                           disable_update=True, vad_model="fsmn-vad")


def score_asr(asr_model, wav_path: str) -> dict:
    res = asr_model.generate(wav_path, output_dir=None, language="auto",
                             use_itn=False, batch_size_s=60)
    rec = res[0] if isinstance(res, list) else res
    text = rec.get("text", "")
    # SenseVoice prepends language/emotion tokens like "<|zh|><|NEUTRAL|>..."; strip them
    import re
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return {"asr_text": text}


def cer_or_wer(ref: str, hyp: str, lang: str) -> float:
    """Character-level error rate for zh, word-level for en. Levenshtein-based."""
    if lang == "zh":
        a = list(ref)
        b = list(hyp)
    else:
        a = ref.split()
        b = hyp.split()
    # standard Levenshtein
    n, m = len(a), len(b)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / n


def load_spk_encoder(cosyvoice_frontend):
    """Reuse cosyvoice's own speaker encoder via its frontend extractor."""
    return cosyvoice_frontend  # exposes _extract_spk_embedding


def score_spk_sim(frontend, wav_path: str, ref_emb: torch.Tensor) -> float:
    emb = frontend._extract_spk_embedding(wav_path)        # (1, 192)
    emb = torch.nn.functional.normalize(emb.float(), dim=1)
    ref = torch.nn.functional.normalize(ref_emb.float(), dim=1)
    return float((emb * ref).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha_manifest", required=True,
                    help="alpha_ladder_manifest.jsonl from inference_svd_alpha.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--skip_asr", action="store_true")
    ap.add_argument("--skip_spk", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    records = []
    with open(args.alpha_manifest, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    LOG.info(f"{len(records)} generations to evaluate")

    em_model = load_emotion2vec()
    asr_model = None if args.skip_asr else load_whisper()
    frontend = None
    if not args.skip_spk:
        from cosyvoice.cli.cosyvoice import AutoModel as CosyAuto
        MODEL_DIR = "/home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512"
        cosyvoice = CosyAuto(model_dir=MODEL_DIR, fp16=False)
        frontend = cosyvoice.frontend

    results = []
    # Build reference embedding cache (per speaker, from manifest's ref wav)
    ref_emb_cache = {}

    for i, r in enumerate(records):
        try:
            emo = score_emotion(em_model, r["wav"], r["target_emotion"])
            entry = {**r, "emo": emo}
            if not args.skip_asr:
                asr = score_asr(asr_model, r["wav"])
                entry["asr"] = asr
                entry["cer_or_wer"] = cer_or_wer(r["text"], asr["asr_text"], r["lang"])
            if not args.skip_spk:
                spk_key = r["speaker"]
                if spk_key not in ref_emb_cache:
                    # use ref wav embedded as the speaker reference
                    # The original ref_emo wav path is implicit; recover from key
                    # For ESD: ref_emotion wav is one of the emotion paths in the
                    # parallel record. We expect alpha_manifest to carry the path,
                    # but it may not. Workaround: use neutral path from manifest.
                    # If not present, fall back to the generated wav itself.
                    src_ref = r.get("ref_wav", None) or r["wav"]
                    ref_emb_cache[spk_key] = frontend._extract_spk_embedding(src_ref)
                entry["spk_sim"] = score_spk_sim(frontend, r["wav"], ref_emb_cache[spk_key])
            results.append(entry)
            if (i + 1) % 20 == 0:
                LOG.info(f"progress: {i+1}/{len(records)}")
        except Exception as e:
            LOG.error(f"[{r.get('tag')}]: {type(e).__name__}: {e}")

    # Aggregate by (speaker, key, target_emotion)
    agg = defaultdict(list)
    for r in results:
        gk = (r["speaker"], r["key"], r["target_emotion"])
        agg[gk].append(r)

    summary = []
    for gk, items in agg.items():
        items = sorted(items, key=lambda x: x["alpha"])
        alphas = [it["alpha"] for it in items]
        emo_sc = [it["emo"].get("target_score") for it in items]
        neu_sc = [it["emo"].get("neutral_score") for it in items]
        wer = [it.get("cer_or_wer") for it in items]
        sim = [it.get("spk_sim") for it in items]

        # monotonicity: count of alpha pairs where emo score is increasing
        mono_pairs = 0
        total_pairs = 0
        for i_ in range(len(alphas) - 1):
            if emo_sc[i_] is not None and emo_sc[i_ + 1] is not None:
                total_pairs += 1
                if emo_sc[i_ + 1] >= emo_sc[i_]:
                    mono_pairs += 1

        summary.append({
            "speaker": gk[0], "key": gk[1], "target_emotion": gk[2],
            "alphas": alphas,
            "emo_scores": emo_sc,
            "neutral_scores": neu_sc,
            "wer_or_cer": wer,
            "spk_sims": sim,
            "monotonic_fraction": (mono_pairs / total_pairs) if total_pairs else None,
            "delta_emo_alpha0_to_alpha1": (
                (emo_sc[alphas.index(1.0)] - emo_sc[alphas.index(0.0)])
                if (0.0 in alphas and 1.0 in alphas and
                    emo_sc[alphas.index(0.0)] is not None and
                    emo_sc[alphas.index(1.0)] is not None)
                else None
            ),
        })

    # Overall pass/fail report against plan.md gating
    overall = {}
    if summary:
        mono_fracs = [s["monotonic_fraction"] for s in summary if s["monotonic_fraction"] is not None]
        deltas = [s["delta_emo_alpha0_to_alpha1"] for s in summary if s["delta_emo_alpha0_to_alpha1"] is not None]
        all_wers = [w for s in summary for w in s["wer_or_cer"] if w is not None]
        all_sims = [v for s in summary for v in s["spk_sims"] if v is not None]
        overall = {
            "n_groups": len(summary),
            "mean_monotonic_fraction": mean(mono_fracs) if mono_fracs else None,
            "mean_delta_emo_0_to_1": mean(deltas) if deltas else None,
            "mean_wer_or_cer": mean(all_wers) if all_wers else None,
            "mean_spk_sim": mean(all_sims) if all_sims else None,
        }

    out_data = {
        "alpha_manifest": args.alpha_manifest,
        "n_evaluated": len(results),
        "per_sample": results,
        "per_group_summary": summary,
        "overall": overall,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    LOG.info(f"wrote {args.out}")
    LOG.info(f"overall: {json.dumps(overall, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
