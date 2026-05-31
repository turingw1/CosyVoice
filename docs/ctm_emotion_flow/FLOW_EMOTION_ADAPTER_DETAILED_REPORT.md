# 情绪 Flow Adapter 训练设计 - 详细报告

日期：2026-05-31  
仓库：`/test1208/zw/ctm_emotion_tts/repos/CosyVoice`  
分支：`exp/emotion-flow-separation-probe`

## 0. 本文边界

本文只规划训练代码和验证路线，不汇报训练结果，也不启动训练。

硬性边界：

- 本阶段不训练完整 CosyVoice。
- 冻结 LLM、flow encoder、DiT estimator、HiFT vocoder。
- 只训练一个很小的情绪速度残差模块。
- 不声称语义 flow 和情绪 flow 已被完美分解。
- CTM / emotion residual 只作为经验性的 emotion-biased residual guidance。
- flow 时间统一记为 `s`，语音/mel/token 帧时间统一记为 `k`。
- 路径统一使用 `/test1208/zw/ctm_emotion_tts`。

核心问题：

> 在同一句文本、同一个说话人、neutral audio 与 emotional audio 成对的条件下，能不能训练出一个小模块，在 flow 生成阶段提供一个可调强度的情绪速度残差，并且这个残差对最终音频有实际情绪控制意义？

用户给出的目标形式是：

```text
v_base = flow(x_s, text, speaker, neutral_ref)
delta_v = adapter(emotion_ref - neutral_ref)
v_new = v_base + alpha * delta_v
```

本文把它落到 CosyVoice3 当前代码中，形成一个 adapter-only 的可验证训练方案。

## 1. 方法调研基础

### 1.1 Flow Matching

Flow Matching 的关键思想是直接回归从噪声到数据分布路径上的速度场，而不是显式模拟扩散反向过程。相关论文：

- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747

对本实验真正有用的是这个局部监督形式：

```text
x_s = (1 - (1 - sigma_min) * s) * z + s * x_1
u_s = x_1 - (1 - sigma_min) * z
loss = || v_theta(x_s, s, condition) - u_s ||^2
```

CosyVoice 的 `compute_loss()` 正在使用同类形式。因此我们可以不改 flow 的大框架，只在 frozen velocity 旁边训练一个残差。

### 1.2 Conditional Flow Matching 在 TTS 中的使用

Matcha-TTS 用 optimal-transport conditional flow matching 做非自回归 mel 生成，证明 CFM/ODE decoder 可以在少量采样步里生成可用语音：

- Matcha-TTS paper: https://arxiv.org/abs/2309.03199
- Matcha-TTS code: https://github.com/shivammehta25/Matcha-TTS

F5-TTS 使用 flow matching 和 DiT 风格结构做 zero-shot TTS，说明 flow matching + transformer 作为语音生成主干已经是可行路线：

- F5-TTS paper: https://arxiv.org/abs/2410.06885
- F5-TTS code: https://github.com/SWivid/F5-TTS

Voicebox 也是关键参考，它使用非自回归 flow-matching 模型，并通过文本和音频上下文控制 speech infilling：

- Voicebox paper: https://arxiv.org/abs/2306.15687

这些工作支持一个工程判断：flow velocity 是值得干预的位置。但它们不能证明情绪和语义天然线性可分。

### 1.3 Guidance 与可控生成

Classifier-Free Diffusion Guidance 通过组合有条件预测和无条件预测来控制生成质量和条件跟随程度：

- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598

CosyVoice 当前 `CausalConditionalCFM.solve_euler()` 已经在 inference 中做了 CFG 风格组合：

```text
dphi_dt = (1 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt
```

本方案的 adapter 与 CFG 思路相近：都在速度场层面做控制。但不同点是，adapter 的 residual 来自 neutral/emotion reference 的对比，而不是 conditional/unconditional 的差。

### 1.4 情绪、风格和参考音频

Global Style Tokens 证明 reference-derived style embedding 可以在一定程度上控制说话风格：

- GST paper: https://arxiv.org/abs/1803.09017

StyleTTS 2 把 style 作为潜变量，用 diffusion 等方法增强风格建模：

- StyleTTS 2 paper: https://arxiv.org/abs/2306.07691

真实平行情绪数据方面，ESD 最适合作为后续验证集，因为它包含同文本、同说话人的 neutral / happy / angry / sad / surprise 情绪录音：

- ESD paper: https://arxiv.org/abs/2105.14762

本阶段建议先用 CosyVoice 自己生成 synthetic parallel data，因为它便宜、可控、模型分布一致。只有 synthetic tiny overfit 和评估脚本跑通后，再切到 ESD 或其他真实平行情绪数据。

## 2. 当前 CosyVoice 代码依据

当前远程分支：

```text
exp/emotion-flow-separation-probe
```

CosyVoice3 checkpoint 配置中的 flow 结构：

- `flow`：`cosyvoice.flow.flow.CausalMaskedDiffWithDiT`
- `decoder`：`cosyvoice.flow.flow_matching.CausalConditionalCFM`
- `estimator`：`cosyvoice.flow.DiT.dit.DiT`
- mel/velocity 通道维度：80
- DiT：`dim=1024`，`depth=22`，`heads=16`，`out_channels=80`
- inference：Euler solver，`inference_cfg_rate=0.7`

关键代码入口：

- `cosyvoice/cli/model.py:425-448`
  - `CosyVoice3Model.token2wav()` 调用 `self.flow.inference(...)`，然后把 mel 交给 HiFT vocoder。
- `cosyvoice/flow/flow.py:369-414`
  - `CausalMaskedDiffWithDiT.inference()` 拼接 prompt token 和 target token，生成文本/语义条件 `h`，构造 prompt mel 条件 `cond`，调用 decoder，最后裁掉 prompt mel。
- `cosyvoice/flow/flow_matching.py:71-124`
  - `solve_euler()` 每个 Euler 步调用 `forward_estimator(...)`，做 CFG 组合，然后执行 `x = x + dt * dphi_dt`。
- `cosyvoice/flow/flow_matching.py:155-193`
  - `compute_loss()` 随机采样 flow time `s`，采样噪声 `z`，构造 noised mel `y` 和目标速度 `u`，用 MSE 训练 estimator。
- `cosyvoice/flow/DiT/dit.py:145-170`
  - `DiT.forward()` 接收 `(x, mask, mu, t, spks, cond, streaming)`，输出 `[B,80,K]` 的 velocity-like tensor。

因此最干净的插入点是：base DiT velocity 计算完成、CFG 组合完成之后，在 Euler 更新之前：

```text
v_base = guided_base_velocity(x_s, mask, mu, s, speaker, neutral_cond)
delta_v = adapter(x_s, mask, mu, s, speaker, neutral_cond, emotion_delta_cond)
v_new = v_base + alpha * generated_region_mask * delta_v
x_next = x_s + ds * v_new
```

`generated_region_mask` 很重要：不要把 residual 加到 prompt/reference mel 区域，只对生成区间生效。

## 3. 可检验假设

弱假设，也就是本阶段真正要验证的内容：

> 对冻结的 CosyVoice3 flow，使用同文本同说话人的 neutral/emotional 平行数据，可以训练一个小 adapter，使 `alpha` 增大时目标情绪得分上升，同时文本内容和说话人特征尽量保持稳定。

强假设，本阶段不能声称：

> 模型内部已经把语义 flow 和情绪 flow 完全分解成两个独立因子。

即使实验成功，`delta_v` 也可能混合了 F0、能量、节奏、时长、频谱包络等因素。它最多能说明：在当前模型和数据分布下，有一个可用的 emotion-biased velocity residual。

## 4. 方案比较

### 方案 A：只做 inference residual probe

沿用已有 demo：source/reference、target-emotion reference、hard switch、crossfade。

优点：

- 不训练，风险最低。
- 能快速听到 reference 切换效果。

缺点：

- 不能回答 residual 是否可训练。
- 第一轮 demo 中 angry 等情绪区分不稳定。

结论：保留为 smoke test，不作为主路线。

### 方案 B：adapter-only residual flow 训练

冻结 CosyVoice，训练一个小模块预测 `delta_v`。

优点：

- 符合用户给出的公式。
- 风险小，不污染完整模型。
- 可以先在极小 synthetic set 上 overfit，失败也有清晰结论。
- 可以自然得到 `alpha` 强度控制。

缺点：

- 可能学到能量、时长等捷径。
- synthetic emotion 标签可能有噪声。
- 需要严谨 ablation，避免假阳性。

结论：推荐本阶段使用。

### 方案 C：全 flow fine-tune 或 LoRA

优点：

- 容量更大。

缺点：

- 违反当前边界。
- 结果更难归因到 residual 通道。
- 更容易损坏内容和说话人一致性。

结论：本阶段不做。

## 5. 数据设计

### 5.1 第一阶段使用 synthetic parallel data

每条训练样本：

```json
{
  "utt_id": "synth_000001_happy",
  "speaker_id": "spk_0001",
  "text": "同一句文本",
  "neutral_wav": "/test1208/zw/ctm_emotion_tts/data/parallel_synth/spk_0001/text_0001/neutral.wav",
  "emotion_wav": "/test1208/zw/ctm_emotion_tts/data/parallel_synth/spk_0001/text_0001/happy.wav",
  "emotion": "happy",
  "emotion_intensity": 1.0,
  "prompt_wav": "/test1208/zw/ctm_emotion_tts/data/references/spk_0001.wav",
  "generator": "CosyVoice3.inference_instruct2",
  "sample_rate": 24000
}
```

生成原则：

- 同一个 text。
- 同一个 speaker reference。
- 先生成 neutral，再生成 angry / happy / sad。
- 记录 instruction、seed、prompt_wav、输出 wav 路径。
- 每个 wav 跑 emotion2vec。
- 只保留 target emotion 明显强于 neutral 的 pair。

建议最小 smoke 规模：

- 1 个 speaker reference。
- 4 条文本。
- 3 个情绪：angry / happy / sad。
- 12 条 emotional pair，加 4 条 neutral control。

建议第一轮 tiny overfit 规模：

- 3 个 speaker/reference。
- 20 条文本。
- 3 个情绪。
- 180 条 emotional tuple，加 60 条 neutral control。

### 5.2 第二阶段再使用真实平行数据

ESD 是后续首选真实 benchmark，因为它具有同文本同说话人的多情绪录音。真实数据阶段应该等 synthetic tiny overfit、alpha ladder、ablation 都跑通之后再开始。

## 6. 特征准备

每个 wav 需要缓存：

- CosyVoice 使用的 mel feature。
- speech token 或生成 `mu` 所需的 token 条件。
- speaker embedding。
- prompt mel。
- neutral condition。
- emotion condition。
- `emotion_delta_cond = emotion_cond - neutral_cond`。

建议 `.pt` 记录包含：

```python
{
    "utt_id": str,
    "text": str,
    "emotion": str,
    "emotion_id": int,
    "neutral_feat": torch.Tensor,       # [80, K]
    "emotion_feat": torch.Tensor,       # [80, K]
    "prompt_feat": torch.Tensor,        # [K_prompt, 80]
    "mu": torch.Tensor,                 # [80, K_total]
    "mask": torch.Tensor,               # [1, K_total]
    "spks": torch.Tensor,               # [80]
    "neutral_cond": torch.Tensor,       # [80, K_total]
    "delta_cond": torch.Tensor,         # [80, K_total]
    "generated_start": int,
}
```

长度问题：

- neutral 和 emotional wav 可能长度不同。
- 第一版直接截断到较短生成区间，避免复杂对齐影响主实验。
- 记录原始长度，并在评估中报告 duration drift。
- 如果截断导致结果不稳定，再引入 DTW 或 soft alignment。

## 7. Adapter 结构

### 7.1 最小模块

第一版 adapter 应该足够小，便于 overfit 和定位问题：

```text
EmotionVelocityAdapter
  输入通道：
    x_s                   80
    mu                    80
    neutral_cond          80
    emotion_delta_cond    80
    flow time embedding   16
    emotion embedding     16
  主体：
    Conv1d(352, 256, kernel=3)
    SiLU
    Conv1d(256, 256, kernel=3, dilation=2)
    SiLU
    Conv1d(256, 80, kernel=1)
  输出：
    delta_v [B, 80, K]
```

最后一层必须 zero-init。这样 adapter 未训练时，`delta_v = 0`，不会改变 base path。

### 7.2 为什么 adapter 不只看 `emotion_ref - neutral_ref`

用户公式中写的是：

```text
delta_v = adapter(emotion_ref - neutral_ref)
```

工程落地时建议改成：

```text
emotion_delta_cond = emotion_cond - neutral_cond
delta_v = adapter(x_s, mu, s, speaker, neutral_cond, emotion_delta_cond, emotion_label)
```

原因：

- velocity residual 与当前 flow state `x_s` 有关。
- residual 在不同 flow time `s` 的意义不同。
- residual 必须对齐到当前 mel/token 帧 `k`。
- `emotion_delta_cond` 仍然是主控制信号，但不是唯一输入。

### 7.3 只作用于生成区间

必须使用：

```text
delta_v = delta_v * mask * generated_region_mask
```

这样 prompt/reference 区间不会被 adapter 污染。

## 8. 训练目标

对 emotional target `x_1^emo`，采样 CosyVoice 原本的 flow path：

```text
s ~ Uniform(0, 1)
z ~ Normal(0, I)
x_s^emo = (1 - (1 - sigma_min) * s) * z + s * x_1^emo
u_s^emo = x_1^emo - (1 - sigma_min) * z
```

冻结 base estimator，用 neutral condition 计算：

```text
v_base = stopgrad(flow_estimator(x_s^emo, mask, mu, s, speaker, neutral_cond))
delta_v = adapter(x_s^emo, mask, mu, s, speaker, neutral_cond, emotion_delta_cond)
v_new = v_base + alpha * delta_v
```

第一版建议直接训练 residual target：

```text
delta_target = stopgrad(u_s^emo - v_base)
L_residual = MSE(mask * generated_region_mask * delta_v,
                 mask * generated_region_mask * delta_target)
```

中性约束：

```text
L_neutral_zero = mean(|| adapter(..., emotion_delta_cond=0) ||_2)
```

情绪结构辅助约束：

```text
pooled_delta = masked_mean(delta_v, k)
L_emotion_cls = CE(linear(pooled_delta), emotion_label)
```

残差幅度约束：

```text
L_norm = mean(||delta_v||_2 / (||v_base||_2 + eps))
```

第一版总 loss：

```text
L = L_residual
  + 0.1 * L_neutral_zero
  + 0.05 * L_emotion_cls
  + 0.01 * L_norm
```

训练时先固定 `alpha=1.0`。`alpha` 连续可控性放到 evaluation 阶段验证。

## 9. 训练阶段

### Stage 0：instrumentation

目标：

- 插入 adapter hook。
- 记录 `v_base`、`delta_v`、shape、norm ratio、cosine similarity。
- 验证 zero-init adapter 不改变 base 输出。

通过标准：

- `delta_v` shape 为 `[B,80,K]`。
- generated-region mask 正确排除 prompt frames。
- adapter disabled 或 `alpha=0` 时 base 输出不变。

### Stage 1：synthetic feature cache

目标：

- 生成或登记 neutral/emotional wav pair。
- 写入 JSONL manifest。
- 缓存 feature `.pt`。
- 写 emotion2vec score。

通过标准：

- 每个 emotion 至少有可用 pair。
- emotion2vec 显示 target emotion 强于 neutral。
- 不合格 pair 被过滤，而不是静默混入训练。

### Stage 2：tiny overfit

用 12 条 emotional tuple 训练。

通过标准：

- `L_residual` 在 200-1000 steps 内下降。
- adapter norm 不爆炸。
- `alpha=0` 保持 neutral/base 行为。
- `alpha=1` 至少在一个情绪上产生可听变化。

### Stage 3：alpha ladder

生成：

```text
alpha = 0.00, 0.25, 0.50, 0.75, 1.00, 1.25
```

每个输出记录：

- emotion2vec label 和 score。
- ASR transcript 或 token-level 内容代理指标。
- speaker similarity。
- duration / energy drift。
- residual norm ratio。
- `delta_v` 与 `v_base` 的 cosine similarity。

通过标准：

- 至少一个 emotion 的 target score 随 alpha 上升。
- transcript 基本稳定。
- speaker similarity 不明显崩溃。
- 没有明显 clipping 或 duration blow-up。

### Stage 4：ablation

必须做：

- random emotion reference。
- mismatched text reference。
- wrong emotion label。
- `emotion_delta_cond=0`。
- negative `alpha`。
- direct mel-delta baseline。

这些 ablation 用来判断 residual 是否真的在学情绪方向，而不是简单放大能量、噪声或时长。

## 10. 建议代码布局

新增：

- `cosyvoice/flow/emotion_adapter.py`
  - `EmotionVelocityAdapter`
  - `make_generated_region_mask`
  - residual stats helper

- `cosyvoice/flow/emotion_guided_flow.py`
  - 对 frozen `CausalConditionalCFM` 的非侵入式 wrapper
  - `solve_euler_emotion(...)`
  - `compute_adapter_residual_loss(...)`

- `examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py`
  - 生成或登记 same-text same-speaker neutral/emotional wav

- `examples/ctm_emotion_flow/prepare_parallel_features.py`
  - 从 manifest 提取并缓存 mel/token/speaker 条件

- `examples/ctm_emotion_flow/train_flow_emotion_adapter.py`
  - 只训练 adapter
  - checkpoint 保存到 `/test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/`

- `examples/ctm_emotion_flow/eval_emotion_adapter.py`
  - 生成 alpha ladder wav
  - 跑 emotion2vec / ASR / speaker metrics

- `examples/ctm_emotion_flow/config/adapter_tiny.yaml`
  - 路径、模型参数、训练参数、评估参数

测试：

- `tests/ctm_emotion_flow/test_emotion_adapter.py`
  - shape、dtype、zero-init、masking。
- `tests/ctm_emotion_flow/test_adapter_loss.py`
  - fake tensor 下验证 residual loss。
- `tests/ctm_emotion_flow/test_alpha_injection.py`
  - `alpha=0` 等于 base velocity，`alpha=1` 只加在生成区间。

文档：

- `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md`
- `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md`
- `docs/superpowers/plans/2026-05-31-flow-emotion-adapter-training.md`

## 11. 未来训练命令草案

进入服务器：

```bash
ssh sai-gpu160
bash
cd /test1208/zw/ctm_emotion_tts/repos/CosyVoice
nvidia-smi -i 0
export CUDA_VISIBLE_DEVICES=0
```

生成 synthetic manifest：

```bash
python examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py \
  --project-root /test1208/zw/ctm_emotion_tts \
  --model-dir /home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512 \
  --out-manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.jsonl \
  --num-texts 20 \
  --emotions angry happy sad
```

准备 feature：

```bash
python examples/ctm_emotion_flow/prepare_parallel_features.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.jsonl \
  --out-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.jsonl
```

tiny overfit：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ctm_emotion_flow/train_flow_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.jsonl \
  --output-dir /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit \
  --max-steps 1000 \
  --batch-size 1
```

评估 alpha ladder：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ctm_emotion_flow/eval_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --checkpoint /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit/adapter.pt \
  --out-dir /test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder \
  --alphas 0 0.25 0.5 0.75 1.0 1.25
```

## 12. 预期 blocker

可能问题：

- synthetic pair 的时长和韵律不完全平行。
- instruction 生成的 angry / sad 等标签不稳定。
- emotion2vec 对生成音频可能过度自信。
- residual 可能只学到 loudness/F0，而不是有效情绪方向。
- 过早 unroll Euler 训练会显存高。
- CosyVoice 当前 CFG 用两条 batch 分支，adapter 不能污染 unconditional 分支。

缓解策略：

- 第一版训练 residual target，不做完整 unrolled audio loss。
- 所有 base 模块冻结。
- 先 tiny overfit，再 alpha ladder，再 ablation。
- 失败也记录为路线结论。
- synthetic 通过后再换 ESD。

## 13. 最终路线判断

推荐下一步：

1. 实现 zero-init `EmotionVelocityAdapter`。
2. 增加 adapter-aware solver wrapper，不改原始 solver 行为。
3. 构建 synthetic parallel manifest 和 feature cache。
4. 用 residual target 训练 tiny adapter。
5. 做 alpha ladder 和 emotion2vec/content/speaker 评估。
6. 通过后再进入 ESD 真实平行数据验证。

这条路线有实际意义，因为它直接在 flow velocity 层验证“情绪速度残差”是否可学、可控、可听。它的结论应保持克制：成功只能说明存在一个可用的 emotion-biased residual channel，不能说明语义和情绪已经被完美因子化。
