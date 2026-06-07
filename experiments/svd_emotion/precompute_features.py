"""Pre-extract speech tokens + mel feats + speaker x-vector for ESD wavs.

For each (key, emotion) wav in the manifest, run the CosyVoice frontend
extractors once and save:
  feature_dir/{key}_{emotion}.pt  with keys
    speech_token: (N_tok,) int32
    speech_feat:  (K, 80) float32
    embedding:    (192,) float32

This is required by ESDParallelDataset and avoids the heavy ONNX/Whisper
pipeline running every training step.

Run on GPU; safe to interrupt and resume (skips existing .pt files).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/test1208/zw/ctm_emotion_tts")
COSYVOICE_REPO = ROOT / "repos" / "CosyVoice"
MODEL_DIR = Path("/home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512")

sys.path.insert(0, str(COSYVOICE_REPO))
sys.path.insert(0, str(COSYVOICE_REPO / "third_party" / "Matcha-TTS"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True, help="Feature cache directory")
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, only process first N records (for smoke testing)")
    ap.add_argument("--emotions", default="Neutral,Happy,Angry,Sad,Surprise")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from cosyvoice.cli.cosyvoice import AutoModel

    cosyvoice = AutoModel(model_dir=str(MODEL_DIR), fp16=False)
    sr = cosyvoice.sample_rate
    frontend = cosyvoice.frontend

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_emos = [e.strip() for e in args.emotions.split(",") if e.strip()]

    records = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if args.limit > 0:
        records = records[:args.limit]

    n_total = sum(len(set(r["emotions"]) & set(target_emos)) for r in records)
    n_done = 0
    n_skip = 0
    n_err = 0
    print(f"[INFO] extracting features for ~{n_total} (key, emotion) wavs")

    for rec in records:
        key = rec["key"]
        for emo in target_emos:
            if emo not in rec["emotions"]:
                continue
            out_path = out_dir / f"{key}_{emo}.pt"
            if out_path.exists():
                n_skip += 1
                continue
            wav_path = rec["emotions"][emo]
            try:
                # frontend_zero_shot extracts everything we need from the wav itself.
                mi = frontend.frontend_zero_shot(rec["text"], "", wav_path, sr, "")
                feat = mi["prompt_speech_feat"][0].cpu()           # (K, 80)
                tok  = mi["flow_prompt_speech_token"][0].cpu().to(torch.int32)  # (N_tok,)
                emb  = mi["flow_embedding"][0].cpu()               # (192,)
                torch.save(
                    {"speech_token": tok, "speech_feat": feat, "embedding": emb},
                    out_path,
                )
                n_done += 1
                if n_done % 200 == 0:
                    print(f"[progress] done={n_done} skipped={n_skip} err={n_err}")
            except Exception as e:
                n_err += 1
                if n_err <= 10:
                    print(f"[ERR] {key} {emo}: {type(e).__name__}: {e}")
    print(f"[DONE] done={n_done} skipped={n_skip} err={n_err} dir={out_dir}")


if __name__ == "__main__":
    main()
