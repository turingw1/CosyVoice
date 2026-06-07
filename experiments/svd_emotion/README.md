# SVD Emotion Decomposition Experiment (v2)

Goal: train CosyVoice3's flow DiT so that its velocity output v ∈ R^{80×K},
when decomposed by SVD, has its **residual** (low-energy directions) align
with the emotion content of the utterance, and its **top-k reconstruction**
align with the semantic / neutral baseline. At inference time, scale the
residual by α to control emotion strength.

See `/Users/.../Voice/plan.md` (v2) for the design rationale and the
quantitative SVD-suitability gating criteria.

**Important convention** (do NOT flip):
- `B = top-k SVD` → semantic baseline (high energy)
- `A = v - B`     → emotion residual (low energy)

This direction is forced by energy distribution: emotion is a small perturbation
on top of speech content, so its energy lives in the small singular values,
not the large ones.

## Files

| File | Purpose |
|---|---|
| `svd_decompose.py` | Reusable SVD decomposition + CFM reconstruction helpers |
| `build_esd_manifest.py` | Parse extracted ESD into parallel JSONL manifest (text-grouped) |
| `precompute_features.py` | Pre-extract speech tokens + mel + x-vec per wav (cache for training) |
| `dataset.py` | Pair-loader + collate for `(emo, neu)` parallel training |
| `step0_residual_probe.py` | **Zero-cost** diagnostic on raw CosyVoice3 (run before training) |
| `train_svd_flow.py` | Fine-tune DiT with L_fm + L_sem + L_emo |
| `inference_svd_alpha.py` | Generate audio with `v_ctrl = α·A + B` per Euler step |
| `eval_svd_suitability.py` | emotion2vec / ASR / SIM scoring + SVD attribution report |

## Pipeline order

```
1. Extract ESD              # already done; data/ESD/raw/
2. Build manifest           # build_esd_manifest.py
3. Step 0 probe             # step0_residual_probe.py  ← gating decision
4. Precompute features      # precompute_features.py
5. Train                    # train_svd_flow.py --smoke_test then full
6. Inference α ladder       # inference_svd_alpha.py
7. Eval suitability         # eval_svd_suitability.py
```

## Step 0 gating thresholds (plan.md §10)

The probe writes a `pass_report` block; thresholds (configurable):

- `test1_energy_pass`:    `||v_emo - v_neu|| / ||v_neu|| >= 5%`
- `test2_direction_pass`: `mean_within_emo_cos - mean_between_emo_cos >= 0.10`
- `test3_lowrank_pass`:   top-8 (or top-10) singular fraction of stacked
                          residuals >= 0.50

Any failure ⇒ the SVD path is unlikely to work and the plan needs rethinking.

## Commands

```bash
# Step 0 probe (zero-cost diagnostic)
CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/step0_residual_probe.py \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --out /test1208/zw/ctm_emotion_tts/outputs/svd_v2/step0/probe_v1.json \
  --per_emotion 5 --gpu 5 --k_svd 16

# Pre-extract features (~35k wavs; takes a while)
CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/precompute_features.py \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --out_dir /test1208/zw/ctm_emotion_tts/data/esd_features --gpu 5

# Smoke-test training (100 samples, 200 steps, no ckpt)
CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/train_svd_flow.py \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --feature_dir /test1208/zw/ctm_emotion_tts/data/esd_features \
  --out_dir /test1208/zw/ctm_emotion_tts/outputs/svd_v2/train_smoke \
  --smoke_test --batch_size 2 --max_steps 200 --gpu 5

# Full training (after smoke test passes)
CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/train_svd_flow.py \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --feature_dir /test1208/zw/ctm_emotion_tts/data/esd_features \
  --out_dir /test1208/zw/ctm_emotion_tts/outputs/svd_v2/train \
  --epochs 5 --batch_size 4 --gpu 5

# Inference α ladder
CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/inference_svd_alpha.py \
  --ckpt /test1208/zw/ctm_emotion_tts/outputs/svd_v2/train/ckpts/dit_epoch5_stepN.pt \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --out_dir /test1208/zw/ctm_emotion_tts/outputs/svd_v2/eval/alpha_ladder \
  --n_samples 10 --alphas 0,0.25,0.5,0.75,1.0,1.25,1.5 --gpu 5

# Eval SVD suitability
CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/eval_svd_suitability.py \
  --alpha_manifest /test1208/zw/ctm_emotion_tts/outputs/svd_v2/eval/alpha_ladder/alpha_ladder_manifest.jsonl \
  --out /test1208/zw/ctm_emotion_tts/outputs/svd_v2/eval/suitability_report.json --gpu 5
```

## ESD layout reference

```
data/ESD/raw/
  0001/                   # 0001-0010 ZH, 0011-0020 EN
    Neutral/   0001_000001.wav ... 0001_000350.wav
    Angry/     0001_000351.wav ... 0001_000700.wav
    Happy/     0001_000701.wav ... 0001_001050.wav
    Sad/       0001_001051.wav ... 0001_001400.wav
    Surprise/  0001_001401.wav ... 0001_001750.wav
    0001.txt   # 1750 lines: <full_key>\t<text>\t<emotion_native_lang>
```

Parallel pairing is by TEXT, not by id: text X appears in 5 lines with 5 different
full_keys (one per emotion). The manifest builder groups by text per speaker.
