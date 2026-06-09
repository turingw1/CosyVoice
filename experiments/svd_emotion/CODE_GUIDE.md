# SVD-Emotion 代码索引

> 仓库：`/test1208/zw/ctm_emotion_tts/repos/CosyVoice` branch `exp/flow-svd-emotion-v2`
> 模块根：`experiments/svd_emotion/`
> GitHub：https://github.com/turingw1/CosyVoice/tree/exp/flow-svd-emotion-v2/experiments/svd_emotion

## 一句话

SVD 分解 DiT 速度场 v，top-k=语义、残差=情感；3 项 loss 训 DiT 让结构出现；推理时 `v_ctrl = α·A + B` 控情感强度。

## Loss

| 项 | 公式 | 位置 |
|---|---|---|
| L_fm | `MSE(v_hat, u_true)` | `train_svd_flow.py:189-191` |
| L_sem | `MSE(time_avg(B_top), time_avg(neu_feat))` | `train_svd_flow.py:193-196` |
| L_emo | `1 - cos(A_crop, v_hat - v_hat_neu)` | `train_svd_flow.py:198-208` |
| 加权求和 | `1.0·L_fm + 0.1·L_sem + 0.5·L_emo` | `train_svd_flow.py:210` |

## Ground truth audio 进训练

| 步骤 | 位置 |
|---|---|
| wav → speech_token + mel + x-vec | `precompute_features.py:79-86` |
| 缓存读取 | `dataset.py:91-99` |
| `(emo, neu)` 平行对返回 | `dataset.py:101-131` |
| token → mu (encoder pipeline) | `train_svd_flow.py:119-130` |
| **`x_1 = feat` (★ ground truth mel)** | `train_svd_flow.py:129` |
| **`u_true = x_1 - (1-σ)z` (★ ground truth velocity)** | `train_svd_flow.py:144` |
| emo 侧 forward (with grad) | `train_svd_flow.py:333-336` |
| neu 侧 forward (no_grad) | `train_svd_flow.py:339-344` |
| 调 loss + backward | `train_svd_flow.py:357-369` |

## Reference token 来源

**训练**：用 wav 自己当 zero-shot prompt，`frontend_zero_shot(text, "", wav_path, sr, "")` 同时拿到 `flow_prompt_speech_token` + `prompt_speech_feat` + `flow_embedding`。见 `precompute_features.py:79-82`。

**推理**：同 frontend 调用，但加 `<|endofprompt|>` prefix；LLM 用 ref 的 token 作 prefix 自回归续写。见 `inference_svd_alpha.py:158-181`。

## SVD + α 控制

| 步骤 | 位置 |
|---|---|
| 核心 SVD 函数 | `svd_decompose.py:22-54` |
| 推理 Euler 循环 | `inference_svd_alpha.py:62-137` |
| **`v_ctrl = α·A + B` (★ α 控制点)** | `inference_svd_alpha.py:114` |
| `--ref_emotion target` 走目标情绪 wav 作 ref | `inference_svd_alpha.py:266-267, 278-280` |

## 其他

| 文件 | 用途 |
|---|---|
| `build_esd_manifest.py` | ESD 平行数据解析 |
| `step0_residual_probe.py` | 训练前诊断 |
| `eval_svd_suitability.py` | emotion2vec 评分 |
| `summarize_train_log.py` | 训练日志摘要 |
