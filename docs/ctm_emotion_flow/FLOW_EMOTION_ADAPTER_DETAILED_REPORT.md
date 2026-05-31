# Flow Emotion Adapter Training Design - Detailed Report

Date: 2026-05-31
Repo: `/test1208/zw/ctm_emotion_tts/repos/CosyVoice`
Branch: `exp/emotion-flow-separation-probe`

## 0. Boundary

This document plans the training code. It does not report trained results.

Hard constraints:

- Do not train or fine-tune the whole CosyVoice model in the first pass.
- Freeze CosyVoice LLM, flow encoder, DiT estimator, and vocoder.
- Train only a small module that injects an emotion-biased velocity residual during flow generation.
- Treat the proposed split as empirical residual guidance, not a proven semantic/emotion disentanglement.
- Use flow time as `s`; use speech/token/mel frame time as `k`.
- Use `/test1208/zw/ctm_emotion_tts` paths only.

The research question is narrow:

> Given parallel neutral/emotional speech for the same text and speaker, can a small module learn a velocity residual channel that controls perceived emotion intensity while preserving the frozen model's semantic/speaker channel?

## 1. Literature Grounding

### 1.1 Flow matching

Flow Matching trains a continuous vector field by regressing the velocity of a probability path from noise to data. The original paper describes FM as a simulation-free objective for continuous normalizing flows and explicitly frames it as vector-field regression over fixed conditional probability paths:

- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747

For this experiment, the useful part is not full CNF likelihood training. The useful part is the local supervised target:

```text
x_s = (1 - (1 - sigma_min) * s) * z + s * x_1
u_s = x_1 - (1 - sigma_min) * z
loss = || v_theta(x_s, s, condition) - u_s ||^2
```

CosyVoice's flow code uses the same shape of target in `compute_loss`, so an adapter can be trained without changing the overall flow formulation.

### 1.2 Conditional flow matching in TTS

Matcha-TTS uses optimal-transport conditional flow matching for non-autoregressive mel generation. It is relevant because it demonstrates that an ODE decoder can generate high-quality mel spectrograms in a small number of synthesis steps:

- Matcha-TTS paper: https://arxiv.org/abs/2309.03199
- Matcha-TTS code: https://github.com/shivammehta25/Matcha-TTS

F5-TTS uses flow matching with a DiT-style backbone and shows that flow-matching TTS can support expressive zero-shot synthesis:

- F5-TTS paper: https://arxiv.org/abs/2410.06885
- F5-TTS code: https://github.com/SWivid/F5-TTS

Voicebox is another important reference because it uses a non-autoregressive flow-matching model conditioned on audio context and text:

- Voicebox paper: https://arxiv.org/abs/2306.15687

These systems support the engineering premise that the vector field during speech generation is a meaningful intervention point. They do not prove that emotion and semantics are linearly separable.

### 1.3 Guidance and controllability

Classifier-free guidance combines conditional and unconditional model predictions to control fidelity/diversity without a separate classifier:

- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598

CosyVoice's CFM solver already uses a CFG-like combination in inference. The proposed adapter is similar in spirit: it changes the velocity before the Euler update, but it uses a learned residual from neutral/emotional reference contrast rather than unconditional guidance.

### 1.4 Emotion/style references

Global Style Tokens showed that a reference-derived style representation can control speaking style independently of text to some degree:

- GST paper: https://arxiv.org/abs/1803.09017

StyleTTS 2 models style as a latent variable using diffusion and speech-language-model-based losses:

- StyleTTS 2 paper: https://arxiv.org/abs/2306.07691

For parallel data, ESD is the most directly relevant public benchmark: same utterances are recorded by the same speakers under neutral, happy, angry, sad, and surprise emotions:

- ESD paper: https://arxiv.org/abs/2105.14762

The first pass can use self-generated parallel data from CosyVoice because it is cheap and matches the current model distribution. ESD or another real parallel emotional dataset should be used before making any strong claim.

## 2. Current CosyVoice Code Grounding

Observed current branch:

```text
## exp/emotion-flow-separation-probe...origin/exp/emotion-flow-separation-probe
```

Model config from the local CosyVoice3 checkpoint:

- `flow` is `cosyvoice.flow.flow.CausalMaskedDiffWithDiT`.
- The decoder is `cosyvoice.flow.flow_matching.CausalConditionalCFM`.
- The estimator is `cosyvoice.flow.DiT.dit.DiT`.
- The mel/velocity channel dimension is 80.
- The DiT estimator has `dim=1024`, `depth=22`, `heads=16`, `out_channels=80`.
- Inference uses Euler solver and `inference_cfg_rate=0.7`.

Relevant code locations:

- `cosyvoice/cli/model.py:425-448`
  - `CosyVoice3Model.token2wav()` calls `self.flow.inference(...)`, then passes generated mel to the HiFT vocoder.
- `cosyvoice/flow/flow.py:369-414`
  - `CausalMaskedDiffWithDiT.inference()` concatenates prompt and target speech tokens, produces text/semantic condition `h`, creates `cond` from prompt mel frames, calls `self.decoder(...)`, and removes prompt mel frames.
- `cosyvoice/flow/flow_matching.py:71-124`
  - `solve_euler()` repeatedly calls `forward_estimator(...)`, applies CFG, and performs `x = x + dt * dphi_dt`.
- `cosyvoice/flow/flow_matching.py:155-193`
  - `compute_loss()` samples flow time `s`, samples noise `z`, builds noised mel `y`, target velocity `u`, then trains the estimator by MSE.
- `cosyvoice/flow/DiT/dit.py:145-170`
  - `DiT.forward()` accepts `(x, mask, mu, t, spks, cond, streaming)` and returns a tensor in mel/velocity space.

The clean intervention point is after the base DiT velocity has been computed and after CFG combination has produced the final base velocity. At that point, the adapter can add a generated-region-only residual before the Euler update:

```text
v_base = guided_base_velocity(x_s, mask, mu, s, speaker, neutral_cond)
delta_v = adapter(x_s, mask, mu, s, speaker, neutral_cond, emotion_delta_cond)
v_new = v_base + alpha * generated_region_mask * delta_v
x_next = x_s + ds * v_new
```

## 3. Hypothesis

The testable hypothesis is:

> For a frozen CosyVoice3 flow, a small adapter trained on same-text same-speaker neutral/emotional pairs can learn a residual velocity component that increases target emotion classifier scores across an `alpha` ladder while mostly preserving transcript and speaker identity.

The stronger claim is explicitly not made:

> The model has perfectly separated semantic and emotional flow factors.

The adapter channel is expected to be entangled with prosody, duration, loudness, and local spectral shape. The experiment is useful only if evaluation shows controllable emotion changes without unacceptable semantic/speaker drift.

## 4. Approaches Considered

### Approach A: Inference-only residual probing

Use existing source/target condition swaps, hard switch, and crossfade demos. This is already useful for diagnosis but cannot learn an emotion direction. It should remain a smoke test.

Pros:

- No training risk.
- Fast and directly audible.

Cons:

- Does not answer whether a residual velocity component is learnable.
- Current first-round demos showed weak emotion separation except for the extreme prompt pair.

### Approach B: Adapter-only residual flow training

Freeze CosyVoice and train a small adapter that predicts `delta_v` from neutral/emotional reference contrast and current flow state.

Pros:

- Matches the user's formula.
- Keeps risk bounded.
- Can be overfit-tested on tiny synthetic parallel data.
- Produces measurable `alpha` control.

Cons:

- May learn acoustic shortcuts.
- May be sensitive to noisy synthetic labels.
- Needs careful evaluation to avoid false disentanglement claims.

Recommendation: start here.

### Approach C: Fine-tune the full flow

Train or LoRA-tune CosyVoice's DiT flow directly.

Pros:

- More capacity.

Cons:

- Violates the current boundary.
- Higher risk of content/speaker degradation.
- Harder to attribute effects to a residual channel.

Do not use in this phase.

## 5. Proposed Training Data

### 5.1 Synthetic parallel data first

Each training tuple:

```json
{
  "utt_id": "synth_000001_happy",
  "speaker_id": "spk_0001",
  "text": "same text for all emotions",
  "neutral_wav": "/test1208/zw/ctm_emotion_tts/data/parallel_synth/spk_0001/text_0001/neutral.wav",
  "emotion_wav": "/test1208/zw/ctm_emotion_tts/data/parallel_synth/spk_0001/text_0001/happy.wav",
  "emotion": "happy",
  "emotion_intensity": 1.0,
  "prompt_wav": "/test1208/zw/ctm_emotion_tts/data/references/spk_0001.wav",
  "generator": "CosyVoice3.inference_instruct2",
  "sample_rate": 24000
}
```

Generation policy:

- Same text, same speaker reference.
- Neutral instruction, then angry/happy/sad instruction.
- Keep prompt reference constant across neutral/emotional versions.
- Save generator instruction text and random seed when possible.
- Store all wavs and a JSONL manifest.
- Run emotion2vec on every generated wav and keep only pairs where the target label is stronger than neutral.

Recommended first scale:

- 3 speakers or references.
- 20 texts.
- 3 emotions: angry, happy, sad.
- 180 emotional tuples plus 60 neutral controls.

Minimum smoke scale:

- 1 speaker.
- 4 texts.
- 3 emotions.
- 12 emotional tuples plus 4 neutral controls.

### 5.2 Real parallel data after synthetic smoke

Use ESD as the first real benchmark because it contains parallel utterances across neutral, happy, angry, sad, and surprise for Chinese and English speakers. The real-data phase should not start until the adapter overfits a tiny synthetic set and the evaluation script is stable.

## 6. Feature Preparation

For each wav:

- Resample or load according to the CosyVoice frontend requirements.
- Extract mel features using the same CosyVoice feature path as inference/training.
- Extract speech tokens where needed for `mu`.
- Extract speaker embedding from the same reference as the base generation.
- Cache feature tensors to avoid recomputing during adapter training.

Feature record:

```text
neutral_feat:  [80, K]
emotion_feat:  [80, K]
prompt_feat:   [K_prompt, 80]
mu:            [80, K_total]
mask:          [1, K_total]
speaker:       [80]
emotion_label: int
emotion_delta_condition: [80, K_total]
```

Length issue:

- Neutral and emotional generated wavs may have different lengths.
- For the first adapter loss, align mel length by truncating to the shorter generated region after removing prompt frames.
- Store the original lengths and track duration drift in evaluation.
- Later, add DTW/soft alignment if truncation hides important prosodic changes.

## 7. Adapter Architecture

### 7.1 Minimal module

The first module should be small enough to overfit quickly:

```text
EmotionVelocityAdapter
  input channels:
    x_s                   80
    mu                    80
    neutral_cond          80
    emotion_delta_cond    80
    time embedding        16 broadcast channels
    emotion embedding     16 broadcast channels
  bottleneck:
    Conv1d(352, 256, kernel=3)
    SiLU
    Conv1d(256, 256, kernel=3, dilation=2)
    SiLU
    Conv1d(256, 80, kernel=1)
  output:
    delta_v [B, 80, K]
```

Use zero initialization for the final projection. This makes `alpha=0` and untrained adapter behavior exactly preserve the frozen base path.

### 7.2 Conditioning choice

The user formula says:

```text
delta_v = adapter(emotion_ref - neutral_ref)
```

In code, the safest first implementation is:

```text
emotion_delta_cond = emotion_cond - neutral_cond
delta_v = adapter(x_s, mu, s, speaker, neutral_cond, emotion_delta_cond, emotion_label)
```

Reason:

- If the adapter sees only `emotion_ref - neutral_ref`, it cannot adapt the residual to the current flow state `x_s` or local token/mel frame.
- If it sees `x_s`, `mu`, and `s`, it can predict a velocity residual with the right shape and stage-specific behavior.
- The emotion delta still remains the main controllable input.

### 7.3 Generated-region mask

Do not apply the residual to prompt frames. The residual must be multiplied by a generated-region mask:

```text
delta_v = delta_v * mask * generated_region_mask
```

This prevents prompt/reference mel corruption and keeps the experiment aligned with online replanning.

## 8. Training Objective

For an emotional target `x_1^emo`, sample the same flow path used by CosyVoice:

```text
s ~ Uniform(0, 1)
z ~ Normal(0, I)
x_s^emo = (1 - (1 - sigma_min) * s) * z + s * x_1^emo
u_s^emo = x_1^emo - (1 - sigma_min) * z
```

Run frozen base with neutral condition:

```text
v_base = stopgrad(flow_estimator(x_s^emo, mask, mu, s, speaker, neutral_cond))
delta_v = adapter(x_s^emo, mask, mu, s, speaker, neutral_cond, emotion_delta_cond)
v_new = v_base + alpha * delta_v
```

Primary loss:

```text
L_velocity = MSE(mask * generated_region_mask * v_new,
                 mask * generated_region_mask * u_s^emo)
```

Equivalent residual target view:

```text
delta_target = stopgrad(u_s^emo - v_base)
L_residual = MSE(mask * generated_region_mask * delta_v,
                 mask * generated_region_mask * delta_target)
```

Use `L_residual` first because it isolates the adapter's target and avoids accidental gradients through frozen modules.

Neutral invariance:

```text
L_neutral_zero = mean(|| adapter(..., emotion_delta_cond=0) ||_2)
```

Contrastive emotion structure:

```text
pooled_delta = masked_mean(delta_v, k)
L_emotion_cls = CE(linear(pooled_delta), emotion_label)
```

This auxiliary classifier is only for shaping the residual; it should not be interpreted as proof of disentanglement.

Regularization:

```text
L_norm = mean(||delta_v||_2 / (||v_base||_2 + eps))
```

Total first-pass loss:

```text
L = L_residual
  + 0.1 * L_neutral_zero
  + 0.05 * L_emotion_cls
  + 0.01 * L_norm
```

Start with `alpha=1.0` in training. Test the continuous alpha ladder only at evaluation.

## 9. Training Stages

### Stage 0: instrumentation only

Deliverables:

- A hook that logs `v_base`, `delta_v`, norm ratio, cosine similarity, and tensor shapes at each Euler step.
- A zero-initialized adapter that produces exactly the base audio when `alpha=0` and nearly base audio when untrained with `alpha>0`.

Pass criteria:

- `delta_v` shape is `[B,80,K]`.
- Generated-region mask excludes prompt frames.
- Base output unchanged when adapter is disabled.

### Stage 1: synthetic parallel feature cache

Deliverables:

- JSONL manifest for neutral/emotional pairs.
- Cached `.pt` feature records.
- emotion2vec scores for all wavs.

Pass criteria:

- At least one tuple per emotion has target emotion2vec score above neutral.
- Bad pairs are filtered rather than silently used.

### Stage 2: tiny overfit

Train on 12 emotional tuples.

Pass criteria:

- `L_residual` decreases over 200-1000 steps.
- Adapter norm does not explode.
- `alpha=0` reproduces neutral/base behavior.
- `alpha=1` audibly moves at least one emotion in the target direction.

### Stage 3: alpha ladder evaluation

Generate:

```text
alpha = 0.00, 0.25, 0.50, 0.75, 1.00, 1.25
```

For each output:

- emotion2vec label and score.
- ASR transcript or token-level content proxy.
- speaker similarity if available.
- duration and energy drift.
- residual norm ratio and cosine against `v_base`.

Pass criteria:

- Target emotion score trends upward for at least one emotion.
- Transcript remains stable.
- Speaker similarity does not collapse.
- No strong clipping or duration blow-up.

### Stage 4: ablations

Required ablations:

- Random emotion reference.
- Mismatched text reference.
- Wrong emotion label.
- `emotion_delta_cond=0`.
- Negative `alpha`.
- Direct mel-delta baseline without velocity adapter.

These ablations determine whether the residual is learning a meaningful emotion path or merely amplifying energy/noise.

## 10. Proposed Code Layout

Create:

- `cosyvoice/flow/emotion_adapter.py`
  - `EmotionVelocityAdapter`
  - `make_generated_region_mask`
  - small pooled auxiliary emotion head

- `cosyvoice/flow/emotion_guided_flow.py`
  - wrapper around frozen `CausalConditionalCFM`
  - adapter-aware `solve_euler_emotion(...)`
  - adapter training helper `compute_adapter_residual_loss(...)`

- `examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py`
  - generate or register same-text same-speaker neutral/emotional wavs
  - save manifest JSONL

- `examples/ctm_emotion_flow/prepare_parallel_features.py`
  - load manifest
  - extract/cache mel/token/speaker features
  - write feature index

- `examples/ctm_emotion_flow/train_flow_emotion_adapter.py`
  - load frozen CosyVoice3
  - train adapter only
  - save checkpoints under `/test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/`

- `examples/ctm_emotion_flow/eval_emotion_adapter.py`
  - generate alpha ladder wavs
  - run emotion2vec/ASR/speaker metrics where available
  - write report JSON and markdown

- `examples/ctm_emotion_flow/config/adapter_tiny.yaml`
  - model/data/training/eval defaults

Tests:

- `tests/ctm_emotion_flow/test_emotion_adapter.py`
  - shape, dtype, zero-init, masking.
- `tests/ctm_emotion_flow/test_adapter_loss.py`
  - residual loss with fake tensors and frozen fake estimator.
- `tests/ctm_emotion_flow/test_alpha_injection.py`
  - `alpha=0` equals base velocity; `alpha=1` adds residual only in generated region.

Docs:

- `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md`
- `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md`
- `docs/superpowers/plans/2026-05-31-flow-emotion-adapter-training.md`

## 11. Minimal Commands for the Future Training Pass

Use interactive bash on the server:

```bash
ssh sai-gpu160
bash
cd /test1208/zw/ctm_emotion_tts/repos/CosyVoice
nvidia-smi -i 0
export CUDA_VISIBLE_DEVICES=0
```

Build synthetic manifest:

```bash
python examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py \
  --project-root /test1208/zw/ctm_emotion_tts \
  --model-dir /home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512 \
  --out-manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.jsonl \
  --num-texts 20 \
  --emotions angry happy sad
```

Prepare features:

```bash
python examples/ctm_emotion_flow/prepare_parallel_features.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.jsonl \
  --out-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.jsonl
```

Tiny overfit:

```bash
python examples/ctm_emotion_flow/train_flow_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.jsonl \
  --output-dir /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit \
  --max-steps 1000 \
  --batch-size 2
```

Evaluate alpha ladder:

```bash
python examples/ctm_emotion_flow/eval_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --checkpoint /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit/adapter.pt \
  --out-dir /test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder \
  --alphas 0 0.25 0.5 0.75 1.0 1.25
```

## 12. Expected Blockers

Likely blockers:

- Synthetic generated pairs may not be truly parallel in duration/prosody.
- CosyVoice instruction emotion may be inconsistent, especially for angry.
- Emotion2vec may saturate on generated audio and overstate success.
- Residual might mostly change energy/F0 while damaging intelligibility.
- Unrolled Euler training may be too memory-heavy if implemented too early.
- Flow CFG path currently uses a two-item batch; adapter injection must be careful not to corrupt the unconditional branch.

Mitigations:

- Start with residual-target loss, not full unrolled audio loss.
- Keep all base modules frozen.
- Evaluate alpha monotonicity and ablations before scaling data.
- Report failure explicitly if residual is non-meaningful.
- Move to ESD real parallel data after synthetic smoke, not before.

## 13. Decision

The recommended next engineering route is:

1. Implement zero-initialized `EmotionVelocityAdapter`.
2. Add adapter-aware solver wrapper while preserving the original solver.
3. Build synthetic parallel manifest and feature cache.
4. Train tiny adapter on residual target.
5. Evaluate alpha ladder with emotion2vec plus content/speaker checks.
6. Only if this passes, move to ESD and stronger validation.

The route is meaningful because it tests a specific mechanistic question at the exact velocity-field level. It remains scientifically modest: success would show a controllable emotion-biased residual channel, not a complete semantic/emotion factorization.
