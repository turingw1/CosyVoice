# Flow Emotion Adapter - Brief Report

Date: 2026-05-31
Repo: `/test1208/zw/ctm_emotion_tts/repos/CosyVoice`
Branch: `exp/emotion-flow-separation-probe`

## Goal

Plan a small adapter-only experiment for CosyVoice3 flow generation:

```text
v_base = frozen_flow_velocity(x_s, text, speaker, neutral_ref)
delta_v = adapter(emotion_ref - neutral_ref, x_s, s, text, speaker)
v_new = v_base + alpha * delta_v
```

The goal is to test whether this residual velocity has practical emotional control value. It is not a claim that semantic and emotion flow factors are perfectly decomposed.

## Why this is the right insertion point

CosyVoice3 currently uses:

```text
CausalMaskedDiffWithDiT -> CausalConditionalCFM -> DiT -> HiFT vocoder
```

The flow solver calls the DiT estimator at every Euler step and receives a velocity-like tensor with mel dimension `[B,80,K]`. The adapter should be inserted after the frozen base velocity and CFG combination, before:

```text
x_next = x_s + ds * v_new
```

Apply the adapter only to generated frames, not prompt/reference frames.

## Recommended first design

Train only:

- `EmotionVelocityAdapter`: small Conv1d/FiLM residual module.
- Optional pooled emotion classifier head for residual shaping.

Freeze:

- LLM/token generator.
- flow encoder/pre-lookahead.
- DiT estimator.
- vocoder.

First loss:

```text
x_s^emo = (1 - (1 - sigma_min) * s) * z + s * x_1^emo
u_s^emo = x_1^emo - (1 - sigma_min) * z
v_base = stopgrad(frozen_estimator(x_s^emo, neutral_ref))
delta_target = stopgrad(u_s^emo - v_base)
L = MSE(delta_v, delta_target)
```

Add small penalties for neutral zero residual, residual norm, and emotion-label consistency.

## Data

Start with synthetic same-text same-speaker pairs:

- neutral audio
- angry audio
- happy audio
- sad audio

Keep only pairs where emotion2vec confirms the target emotion is stronger than neutral. Later move to ESD real parallel data.

## Evaluation

Generate an alpha ladder:

```text
alpha = 0, 0.25, 0.5, 0.75, 1.0, 1.25
```

Measure:

- emotion2vec target score trend.
- transcript/content stability.
- speaker similarity if available.
- duration/energy drift.
- residual norm ratio and cosine vs base velocity.
- ablations: random emotion ref, wrong emotion label, zero delta, negative alpha.

## Deliverables for the next implementation pass

Planned files:

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

## Decision

Proceed with adapter-only residual training. The first target is not quality, but falsifiability:

- If loss does not decrease on tiny overfit, the velocity residual idea is likely weak.
- If alpha does not produce monotonic emotion change, the residual is not a useful control channel.
- If content/speaker drift dominates, the apparent emotion channel is not practically meaningful.
