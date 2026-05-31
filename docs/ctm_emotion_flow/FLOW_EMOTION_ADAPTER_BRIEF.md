# 情绪 Flow Adapter - 简要报告

日期：2026-05-31  
仓库：`/test1208/zw/ctm_emotion_tts/repos/CosyVoice`  
分支：`exp/emotion-flow-separation-probe`

## 目标

规划一个只训练小模块、不训练完整 CosyVoice 的 flow 情绪控制实验：

```text
v_base = frozen_flow_velocity(x_s, text, speaker, neutral_ref)
delta_v = adapter(emotion_ref - neutral_ref, x_s, s, text, speaker)
v_new = v_base + alpha * delta_v
```

目标不是证明语义和情绪 flow 已经完美分解，而是验证：在 flow 生成阶段加一个可训练 residual velocity channel，是否能产生实际可听、可量化的情绪强度控制。

## 插入点

CosyVoice3 当前路径：

```text
CausalMaskedDiffWithDiT -> CausalConditionalCFM -> DiT -> HiFT vocoder
```

`CausalConditionalCFM.solve_euler()` 每个 Euler 步都会从 DiT estimator 得到 `[B,80,K]` 的 velocity-like tensor，并在 CFG 组合后执行：

```text
x_next = x_s + ds * dphi_dt
```

adapter 应该插在 CFG 后、Euler 更新前：

```text
v_new = v_base + alpha * generated_region_mask * delta_v
```

只作用于生成区间，不作用于 prompt/reference mel。

## 推荐设计

只训练：

- `EmotionVelocityAdapter`：小型 Conv1d/FiLM residual 模块。
- 可选的 pooled emotion classifier head：只用于约束 residual 的情绪结构。

冻结：

- LLM/token generator。
- flow encoder/pre-lookahead。
- DiT estimator。
- vocoder。

第一版 loss：

```text
x_s^emo = (1 - (1 - sigma_min) * s) * z + s * x_1^emo
u_s^emo = x_1^emo - (1 - sigma_min) * z
v_base = stopgrad(frozen_estimator(x_s^emo, neutral_ref))
delta_target = stopgrad(u_s^emo - v_base)
L = MSE(delta_v, delta_target)
```

再加小权重约束：

- neutral pair 时 residual 接近 0。
- residual norm 不爆炸。
- pooled residual 能区分 angry / happy / sad。

## 数据

第一阶段使用自行生成的平行数据：

- 同一句文本。
- 同一个说话人 reference。
- neutral audio。
- angry / happy / sad audio。

每个 wav 跑 emotion2vec，只保留目标情绪明显强于 neutral 的 pair。synthetic tiny overfit 成功后，再切到 ESD 等真实平行情绪数据。

## 评估

生成 alpha ladder：

```text
alpha = 0, 0.25, 0.5, 0.75, 1.0, 1.25
```

记录：

- emotion2vec 目标情绪分数是否随 alpha 上升。
- transcript/content 是否稳定。
- speaker similarity 是否稳定。
- duration / energy 是否异常漂移。
- residual norm ratio。
- `delta_v` 和 `v_base` 的 cosine similarity。

必须做 ablation：

- random emotion ref。
- wrong emotion label。
- `emotion_delta_cond=0`。
- negative alpha。
- direct mel-delta baseline。

## 后续实现文件

计划新增：

- `cosyvoice/flow/emotion_adapter.py`
- `cosyvoice/flow/emotion_guided_flow.py`
- `examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py`
- `examples/ctm_emotion_flow/prepare_parallel_features.py`
- `examples/ctm_emotion_flow/train_flow_emotion_adapter.py`
- `examples/ctm_emotion_flow/eval_emotion_adapter.py`
- `examples/ctm_emotion_flow/config/adapter_tiny.yaml`
- `tests/ctm_emotion_flow/test_emotion_adapter.py`
- `tests/ctm_emotion_flow/test_adapter_loss.py`
- `tests/ctm_emotion_flow/test_alpha_injection.py`

## 判断标准

继续推进的条件：

- tiny overfit 中 residual loss 能下降。
- `alpha=0` 保持 base 行为。
- 目标情绪分数随 alpha 有趋势变化。
- 内容和说话人没有明显崩溃。
- ablation 不能轻易复现同样效果。

失败也有价值：如果 loss 不下降、alpha 不单调、或情绪变化主要来自内容/说话人崩坏，就说明当前 residual 分解假设不可靠，应停止扩大训练。
