# SVD-Emotion 实验代码索引（plan v2）

> 用途：让你**精确定位**每个机制实现在哪里。每条引用都给出 `文件:行号`，可直接 GitHub 跳转或 IDE 打开。
>
> 仓库根：`/test1208/zw/ctm_emotion_tts/repos/CosyVoice` (branch `exp/flow-svd-emotion-v2`)
> 模块根：`experiments/svd_emotion/`
> 上线日期：2026-06-09

---

## 0. 一段话项目介绍

CosyVoice3 的 flow DiT 输出速度场 `v ∈ ℝ^{B×80×K}`。我们假设 SVD 分解 `v = U Σ V^T` 后：

- **top-k 重建 `B`** 携带"语义骨架"（neutral velocity 的基线）
- **残差 `A = v − B`** 携带"情感扰动"

通过给 DiT 加 3 项联合 loss 微调，让 SVD 分解后真实出现这个结构。推理时用 `v_ctrl = α·A + B` 缩放残差实现情感强度控制。

ESD 平行数据训练后，emotion2vec 实测 12 组样本里 70% α-邻对单调上升、α=0→α=1 情感分平均 +0.40 —— **plan 假设被实证支持**。

完整设计文档（含数学推导）：`/Users/gzwmac/vscode/gitproject/Voice/plan.md`（本地）

---

## 1. 整体数据流

```
原始 wav (例 Sad 录音)
   │
   │ [precompute] frontend.frontend_zero_shot()
   ▼
.pt 缓存 {speech_token, speech_feat, embedding}
   │
   │ [dataset] ESDParallelDataset 按 (key, emotion) 加载，组成 (emo, neu) 平行对
   ▼
batch
   │
   │ [train] build_flow_inputs side="emo" / "neu"
   ▼
两套 (mu, spks, cond, x_1, mask)
   │
   │ [train] cfm_perturb(x_1) → y, t, z, u_true (★ground truth velocity)
   ▼
v_hat = DiT(y, t, mu, spks, cond)             # emo 侧，带梯度
v_hat_neu = DiT(...neu inputs..., no_grad)    # neu 侧
   │
   │ [train] compute_svd_loss → L_fm + L_sem + L_emo
   ▼
backward → AdamW(DiT only)
```

推理时：

```
ref_wav → frontend (zero-shot) → LLM 生成 speech_token
   ↓
build flow inputs → svd_alpha_solve_euler:
   for each Euler step:
       v = DiT(...)
       A, B = svd_decompose(v, k=16)
       v_ctrl = α·A + B          # ★α 控制点
       x = x + dt·v_ctrl
mel → hift vocoder → wav
```

---

## 2. SVD 方向约定（重要，不要写反）

```
B = top-k 奇异分量重建      = 语义基线（高能量方向）
A = v − B                  = 情感残差（低能量方向）
```

物理依据：emo 是小扰动，`||u_emo − u_neu||_F ≈ 0.1·||u_neu||_F`。SVD 按能量排序，top-k 必然落在 u_neu 主导的高能量方向上；残差才轮到情感。Step 0 实证 `cos(B, v_neu) = 0.93`，证实了这个方向。

`svd_decompose.py:22` 的 docstring 里有完整论证，**不要改方向**。

---

## 3. 代码模块索引

### 3.1 `svd_decompose.py` — 核心 SVD 工具

| 功能 | 位置 |
|---|---|
| **`svd_decompose(v, k)`** — 返回 `(A, B, S)`，A=残差、B=top-k 重建 | `:22-54` |
| **`reconstruct_x1_cfm(y, t, v)`** — rectified flow 还原 `x_1 = y + (1-t)·v` | `:57-77` |
| **`cfm_forward_perturb(x_1, σ_min, t, z)`** — 构造 `(y, t, z, u_true)` | `:80-114` |
| `energy_ratio(numer, denom)` — Frobenius norm 比 | `:117-124` |
| `cosine_flat(a, b)` — flatten 后余弦 | `:127-133` |
| **`singular_spectrum_summary(v)`** — 奇异谱诊断（top-k 能量占比） | `:136-155` |

### 3.2 `build_esd_manifest.py` — ESD 数据解析

| 功能 | 位置 |
|---|---|
| `parse_transcript(txt_path)` — 解析 ESD 转录文件（3 列 tsv） | `:55-83` |
| `find_wav(emo_dir, full_key)` — 查找 wav 文件，支持 train/eval/test 子目录 | `:86-97` |
| **`main()`** — 按 **文本** 分组（不是按 wav id）生成平行 manifest | `:105-202` |
| 关键约定 ZH=0001-0010 / EN=0011-0020 | `:51-53` |
| ID 偏移：Neutral=000001+, Angry=000351+, Happy=000701+, ... | docstring `:17-20` |

### 3.3 `precompute_features.py` — 特征缓存生成（★ Ground truth 入口）

| 功能 | 位置 |
|---|---|
| 调 `frontend.frontend_zero_shot(text, "", wav, sr, "")` 提取 token/mel/x-vec | **`:79`** |
| `prompt_speech_feat` (K, 80) — mel ground truth | `:80` |
| `flow_prompt_speech_token` (N_tok,) — 该 wav 的 speech token | `:81` |
| `flow_embedding` (192,) — speaker x-vector | `:82` |
| 落盘到 `.pt`，schema `{speech_token, speech_feat, embedding}` | `:83-86` |

**这是 ground truth audio 转成 ML 张量的唯一入口**。一个 wav → 3 个张量，全靠 CosyVoice 自己的 frontend，无自定义逻辑。

### 3.4 `dataset.py` — 平行对 dataloader

| 功能 | 位置 |
|---|---|
| `ESDParallelDataset.__init__` — 读 manifest，扩展每个 (key, target_emo) 成一条记录 | `:53-78` |
| `_load_cache(key, emotion)` — 加载 `.pt` 文件 | `:91-99` |
| **`__getitem__(idx)`** — 同时加载 emo + neu 两侧特征，返回 dict | **`:101-131`** |
| Emotion ID 映射 (Neutral=0, Happy=1, Angry=2, Sad=3, Surprise=4) | `:23` |
| `collate_parallel(batch)` — 变长 token / mel 的 pad-collate | `:133-191` |

### 3.5 `step0_residual_probe.py` — 训练前零成本诊断

| 功能 | 位置 |
|---|---|
| `build_model_input(cosyvoice, wav, text)` — 调 frontend 提取 | `:64-73` |
| **`flow_estimator_at_t(cosyvoice, mi, t_value=0.5)`** — 在固定 s=0.5 跑一次 DiT 前向 | `:76-128` |
| 在 `:104-114` 构造 `y = (1-(1-σ)t)z + t·x_1` 并算 `u_true` | `:104-114` |
| 直接调 `flow.decoder.forward_estimator(y, mask, mu, t, spks, cond)` | `:117` |
| `main()` — 跑 20 个 (emo, neu) 对，统计能量比 / 跨样本 cos / SVD 谱 | `:137-335` |
| 三个 gate：energy_pass / direction_pass / lowrank_pass | `:303-318` |

### 3.6 `train_svd_flow.py` — 训练主入口（★ 最重要）

| 功能 | 位置 |
|---|---|
| `freeze_all_but_dit(cosyvoice)` — 冻结 LLM/HiFT/encoder，只训 DiT 22 层 transformer | `:57-78` |
| `enable_grad_checkpointing(estimator)` — 可选 grad ckpt 省内存 | `:81-104` |
| **`build_flow_inputs(flow, batch, device, side)`** — 数据→ (mu, spks, cond, x_1, mask) | **`:107-134`** |
| ↳ `x_1 = feat[:, :K].transpose(1, 2)` — **★ ground truth mel** | `:129` |
| ↳ `mu = flow.input_embedding → pre_lookahead → repeat_interleave` | `:123-130` |
| ↳ `spks = F.normalize → flow.spk_embed_affine_layer` | `:119-120` |
| **`cfm_perturb(x_1)`** — CFM 加噪 + 计算 ground truth velocity | **`:137-145`** |
| ↳ `y = (1-(1-σ)t)·z + t·x_1` | `:143` |
| ↳ `u_true = x_1 - (1-σ)·z` ★ | `:144` |
| `masked_mean_time(x, mask)` — 通道维平均 | `:148-151` |
| **`compute_svd_loss(...)`** — 3 项 loss | **`:154-220`** |
| ↳ **L_fm** masked MSE | `:189-191` |
| ↳ **L_sem** time-avg per-channel MSE | `:193-196` |
| ↳ **L_emo** velocity-space cosine on common crop | `:198-208` |
| `main()` — 数据载入 + 训练循环 | `:223-401` |
| ↳ emo 侧 with grad forward | `:333-336` |
| ↳ neu 侧 **no_grad** 第二次 forward | `:339-344` |
| ↳ 构造 `common_mask = min(K_emo, K_neu)` | `:351-355` |
| ↳ 调 `compute_svd_loss` + backward | `:357-369` |
| ↳ ckpt 每 epoch 保存 | `:392-395` |

### 3.7 `inference_svd_alpha.py` — α 控制推理

| 功能 | 位置 |
|---|---|
| **`svd_alpha_solve_euler(decoder, x_init, mu, mask, spks, cond, k_svd, alpha)`** | **`:62-137`** |
| ↳ 每 Euler step 做 SVD 分解 + α 缩放 | `:108-117` |
| ↳ `v_ctrl = α·A + B`（★ α 控制点） | `:114` |
| `synthesize_with_alpha(cosyvoice, ref_wav, ref_text, target_text, alpha)` | `:140-219` |
| ↳ 加 `<|endofprompt|>` prefix（CosyVoice3 强制要求） | `:159` |
| ↳ LLM 自回归生成 speech token | `:162-181` |
| ↳ flow encoder pipeline 同训练 | `:184-211` |
| ↳ HiFT vocoder（`finalize=True`，不是 `cache_source`） | `:215` |
| `main()` — ladder 推理入口 | `:220-310` |
| ↳ `--ref_emotion target` 用目标情绪 wav 作 ref（实验关键设计） | `:266-267, :278-280` |

### 3.8 `eval_svd_suitability.py` — emotion2vec 评分

| 功能 | 位置 |
|---|---|
| `load_emotion2vec()` — FunASR 加载 `emotion2vec_plus_large` | `:55-62` |
| `score_emotion(model, wav, target_emo)` — 出 `{target_score, neutral_score, label_score}` | `:65-87` |
| `load_whisper()` — SenseVoice ASR（有 triton 兼容问题，已 `--skip_asr`） | `:89-94` |
| `cer_or_wer(ref, hyp, lang)` — Levenshtein-based | `:108-128` |
| `main()` — 跑 emotion2vec + ASR + speaker SIM，aggregate per-group | `:143-272` |
| 单调度计算 `mono_pairs / total_pairs` | `:230-237` |

### 3.9 `summarize_train_log.py` — 日志摘要工具

| 功能 | 位置 |
|---|---|
| `trend(xs, ratio=0.5)` — 头部 vs 尾部均值比 | `:8-17` |
| `main()` — 输出 first/mid/last + verdict | `:20-107` |

---

## 4. Loss 设计精确定位

```
L_total = λ_fm·L_fm + λ_sem·L_sem + λ_emo·L_emo
权重: 1.0, 0.1, 0.5 (训练实测稳定)
```

| Loss 项 | 物理含义 | 代码 | Ground truth 来源 |
|---|---|---|---|
| **L_fm** | TTS 质量锚 | `train_svd_flow.py:189-191` | `u_true = x_1 - (1-σ)z` (`:144`) |
| **L_sem** | B 拉向 neutral mel 通道分布 | `train_svd_flow.py:193-196` | `neu_feat = inp_neu["x_1"]` (`:346`) |
| **L_emo** | A 对齐 (v_emo - v_neu) | `train_svd_flow.py:198-208` | `target_dir = v_hat - v_hat_neu` (`:202`) |

L_emo 的 3 次迭代设计史（重要）：

| 版本 | target 是什么 | 结果 | commit |
|---|---|---|---|
| v1 | `cos(time_avg(A), mel_emo_mean - mel_neu_mean)` | cos 卡 0.02（SVD DC 吸收） | `5b5a8fa` |
| v2 | `cos(vec(A), vec(emo_mel - interp(neu_mel)))` | cos 卡 0.03（线性插值毁 phone 对齐） | `b6e7517` 之前 |
| v3 | `cos(vec(A), vec(v_hat - v_hat_neu_no_grad))` ★ | cos 起点 0.23 → 0.32 | `b6e7517` ★ |

---

## 5. Ground truth audio 怎么放进训练

**关键流程图**（精确到行号）：

```
原始 wav 路径 (例 0003/Sad/0003_001271.wav)
   │
   │ ★ precompute_features.py:79
   │   frontend.frontend_zero_shot(text, "", wav_path, sr, "")
   ▼
mi["prompt_speech_feat"]    →  ★ ground truth mel  (K, 80)
mi["flow_prompt_speech_token"] →  speech token  (N_tok,)
mi["flow_embedding"]        →  x-vector  (192,)
   │
   │ 落盘 precompute_features.py:83-86
   ▼
data/esd_features/{key}_{emotion}.pt
   │
   │ dataset.py:91-99 _load_cache
   ▼
__getitem__ dict (含 speech_feat_emo / speech_feat_neu)  → dataset.py:117-129
   │
   │ collate_parallel pad 后变 batch
   ▼
batch["speech_feat_emo"]  shape (B, K_max, 80)
   │
   │ train_svd_flow.py:115 feat = batch[f"speech_feat_emo"]
   │ train_svd_flow.py:129 x_1 = feat[:, :K].transpose(1, 2)   ★ ground truth
   ▼
x_1  shape (B, 80, K)
   │
   │ train_svd_flow.py:143-144  cfm_perturb
   │   y      = (1 - (1-σ)t) z + t · x_1
   │   u_true = x_1 - (1-σ) z                  ★ ground truth velocity
   ▼
v_hat = DiT(y, t, mu, spks, cond)    train_svd_flow.py:334-336
   │
   │ compute_svd_loss:
   │   L_fm = MSE(v_hat, u_true)              ★ ground truth check
   ▼
backward
```

---

## 6. 推理时的代码路径

```
ref wav (例 0003/Sad/0003_001271.wav)
   │
   │ inference_svd_alpha.py:158
   │   frontend.frontend_zero_shot(target_text, prompt_text, ref_wav, sr, "")
   │   prompt_text = "You are a helpful assistant.<|endofprompt|>" + ref_text  (:159)
   ▼
mi = {text, prompt_text, llm_prompt_speech_token, flow_prompt_speech_token,
      prompt_speech_feat, llm_embedding, flow_embedding}
   │
   │ inference_svd_alpha.py:162-181  LLM 自回归 token generation
   ▼
gen_token = [生成的 speech token]
   │
   │ inference_svd_alpha.py:184-211  flow encoder
   │   concat(prompt_token, gen_token) → input_embedding → pre_lookahead → repeat
   │   → mu;   prompt_feat → cond[:K_prompt];   x_vec → spks
   ▼
mu, mask, spks, cond
   │
   │ inference_svd_alpha.py:213  svd_alpha_solve_euler
   │   ┌─ for step in 1..10:
   │   │     v_raw = DiT(x, mask, mu, s, spks, cond)        (:99-100)
   │   │     A, B = svd_decompose(v_raw, k=16)              (:104)
   │   │     v_ctrl = α·A + B                                (:114)   ★
   │   │     x = x + ds·v_ctrl                               (:116)
   │   └─ return x
   ▼
mel = x[:, :, K_prompt:]                  (:217)
   │
   │ inference_svd_alpha.py:215  hift.inference(speech_feat=mel, finalize=True)
   ▼
waveform (1, samples)
```

---

## 7. 实证结果路径

| 数据 | 路径 |
|---|---|
| Step 0 诊断 (训练前) | `outputs/svd_v2/step0/probe_v1_recovered.json` |
| 训练日志 (1074 行 jsonl) | `outputs/svd_v2/train_full/train.log.jsonl` |
| 5 个 epoch ckpts | `outputs/svd_v2/train_full/ckpts/dit_epoch{1..5}_step*.pt` |
| α-ladder 推理 wav (84 wav) | `outputs/svd_v2/eval/alpha_ladder_emoref/` |
| emotion2vec 主报告 ★ | `outputs/svd_v2/eval/suitability_report_emoref.json` |
| 反面对照 wav + 报告 | `outputs/svd_v2/eval/alpha_ladder/` + `suitability_report.json` |

---

## 8. 运行命令快速参考

### Step 0 诊断
```bash
cd /test1208/zw/ctm_emotion_tts/repos/CosyVoice
CUDA_VISIBLE_DEVICES=5 envs/ctm-cosyvoice/bin/python \
  experiments/svd_emotion/step0_residual_probe.py \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --out /test1208/zw/ctm_emotion_tts/outputs/svd_v2/step0/probe_v2.json \
  --per_emotion 5 --gpu 5
```

### 训练（全量 ESD，5 epoch ~2 小时）
```bash
tmux new-session -d -s train_full "CUDA_VISIBLE_DEVICES=5 \
  /test1208/zw/ctm_emotion_tts/envs/ctm-cosyvoice/bin/python \
  experiments/svd_emotion/train_svd_flow.py \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --feature_dir /test1208/zw/ctm_emotion_tts/data/esd_features \
  --out_dir /test1208/zw/ctm_emotion_tts/outputs/svd_v2/train_run2 \
  --epochs 5 --batch_size 2 --lambda_emo 0.5 --gpu 5"
```

### α ladder 推理 + 评估
```bash
CUDA_VISIBLE_DEVICES=5 envs/ctm-cosyvoice/bin/python \
  experiments/svd_emotion/inference_svd_alpha.py \
  --ckpt /test1208/zw/ctm_emotion_tts/outputs/svd_v2/train_full/ckpts/dit_epoch5_step53700.pt \
  --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \
  --out_dir /test1208/zw/ctm_emotion_tts/outputs/svd_v2/eval/alpha_ladder_v2 \
  --emotions Happy,Sad,Angry,Surprise --n_samples 3 \
  --alphas 0,0.25,0.5,0.75,1.0,1.25,1.5 --ref_emotion target --gpu 5

CUDA_VISIBLE_DEVICES=5 envs/ctm-cosyvoice/bin/python \
  experiments/svd_emotion/eval_svd_suitability.py \
  --alpha_manifest .../alpha_ladder_v2/alpha_ladder_manifest.jsonl \
  --out .../alpha_ladder_v2/suitability_report.json --skip_asr --skip_spk --gpu 5
```

---

## 9. 已知陷阱（前人踩过的坑）

| 陷阱 | 原因 | 修复位置 |
|---|---|---|
| `CosyVoice3Model.parameters()` AttributeError | 不是 nn.Module | `train_svd_flow.py:57-78` |
| OOM at backward with GPU 半空 | 多卡 + fp32 + 全 DiT | `train_svd_flow.py:294-296` 加 `--amp_dtype bf16` |
| `cusolverDnCreate failed` | GPU 已满 cusolver 装不下 workspace | 释放 GPU 或 `torch.backends.cuda.preferred_linalg_library('magma')` |
| `<|endofprompt|>` AssertionError | CosyVoice3 frontend 必须有 | `inference_svd_alpha.py:159` |
| `hift.inference() got unexpected cache_source` | `CausalHiFTGenerator` 用 `finalize` | `inference_svd_alpha.py:215` |
| α-ladder 用 Neutral ref，全 0 分 | LLM 用 neutral ref 产 neutral token → 无情感残差 | 必须 `--ref_emotion target` |
| L_emo 时间平均 cos 卡 0 | SVD top-k 结构性吸收 DC | 改全张量 + velocity-space target |
| L_emo 用 mel-diff target 仍卡 0 | linear interp 毁 phone 对齐 | 改双前向 v_hat_emo - v_hat_neu |

---

## 10. GitHub 浏览

> [github.com/turingw1/CosyVoice/tree/exp/flow-svd-emotion-v2/experiments/svd_emotion](https://github.com/turingw1/CosyVoice/tree/exp/flow-svd-emotion-v2/experiments/svd_emotion)

每条 `:行号` 引用都能直接对应到 GitHub 上的高亮行。
