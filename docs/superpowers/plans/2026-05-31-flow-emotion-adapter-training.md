# Flow Emotion Adapter Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small adapter-only training path that tests whether CosyVoice3 flow velocity can accept a learnable emotion residual channel.

**Architecture:** Keep CosyVoice3 frozen and add a zero-initialized `EmotionVelocityAdapter` around the flow solver. Train the adapter on parallel neutral/emotional mel pairs by regressing the frozen base velocity residual, then evaluate alpha-controlled generation.

**Tech Stack:** Python, PyTorch, CosyVoice3, CosyVoice CFM/DiT flow, FunASR emotion2vec, pytest.

---

## File Structure

- Create `cosyvoice/flow/emotion_adapter.py`
  - Small adapter module, generated-frame mask helper, residual stats helper.
- Create `cosyvoice/flow/emotion_guided_flow.py`
  - Non-invasive wrapper for frozen `CausalConditionalCFM`; original CosyVoice solver remains unchanged.
- Create `examples/ctm_emotion_flow/config/adapter_tiny.yaml`
  - Model paths, data paths, adapter hyperparameters, training/eval defaults.
- Create `examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py`
  - Generate or register synthetic parallel neutral/emotional wav pairs.
- Create `examples/ctm_emotion_flow/prepare_parallel_features.py`
  - Convert manifest wavs into cached mel/token/speaker feature records.
- Create `examples/ctm_emotion_flow/train_flow_emotion_adapter.py`
  - Train adapter only; save `adapter.pt`, logs, and config copy.
- Create `examples/ctm_emotion_flow/eval_emotion_adapter.py`
  - Generate alpha ladder wavs and score emotion/content/speaker metrics.
- Create `tests/ctm_emotion_flow/test_emotion_adapter.py`
  - Shape, zero-init, masking tests.
- Create `tests/ctm_emotion_flow/test_adapter_loss.py`
  - Residual target loss with fake tensors.
- Create `tests/ctm_emotion_flow/test_alpha_injection.py`
  - Verify `alpha=0` and generated-region-only injection.
- Create `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md`
  - Long research/design report.
- Create `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md`
  - Short route summary.

## Task 1: Adapter Module

**Files:**
- Create: `cosyvoice/flow/emotion_adapter.py`
- Test: `tests/ctm_emotion_flow/test_emotion_adapter.py`

- [ ] **Step 1: Write the adapter shape and zero-init test**

```python
import torch

from cosyvoice.flow.emotion_adapter import EmotionVelocityAdapter, make_generated_region_mask


def test_adapter_zero_init_outputs_zero_residual():
    adapter = EmotionVelocityAdapter(mel_dim=80, hidden_dim=32, time_dim=8, emotion_dim=8, num_emotions=4)
    batch, frames = 2, 17
    x_s = torch.randn(batch, 80, frames)
    mu = torch.randn(batch, 80, frames)
    neutral_cond = torch.randn(batch, 80, frames)
    delta_cond = torch.randn(batch, 80, frames)
    flow_s = torch.tensor([0.2, 0.8])
    emotion = torch.tensor([1, 2])
    out = adapter(x_s=x_s, mu=mu, neutral_cond=neutral_cond, delta_cond=delta_cond, flow_s=flow_s, emotion_id=emotion)
    assert out.shape == (batch, 80, frames)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_generated_region_mask_excludes_prompt_frames():
    mask = make_generated_region_mask(total_frames=10, prompt_frames=3, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (1, 1, 10)
    assert mask[0, 0, :3].sum().item() == 0
    assert mask[0, 0, 3:].sum().item() == 7
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest tests/ctm_emotion_flow/test_emotion_adapter.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'cosyvoice.flow.emotion_adapter'
```

- [ ] **Step 3: Implement `EmotionVelocityAdapter`**

```python
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EmotionAdapterStats:
    residual_norm: torch.Tensor
    base_norm: torch.Tensor
    norm_ratio: torch.Tensor
    cosine_to_base: torch.Tensor


def make_generated_region_mask(total_frames: int, prompt_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.ones(1, 1, total_frames, device=device, dtype=dtype)
    if prompt_frames > 0:
        mask[:, :, :prompt_frames] = 0
    return mask


class EmotionVelocityAdapter(nn.Module):
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 256, time_dim: int = 16, emotion_dim: int = 16, num_emotions: int = 4):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.emotion_embed = nn.Embedding(num_emotions, emotion_dim)
        in_channels = mel_dim * 4 + time_dim + emotion_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, mel_dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        *,
        x_s: torch.Tensor,
        mu: torch.Tensor,
        neutral_cond: torch.Tensor,
        delta_cond: torch.Tensor,
        flow_s: torch.Tensor,
        emotion_id: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, frames = x_s.shape
        t = flow_s.reshape(batch, 1).to(dtype=x_s.dtype, device=x_s.device)
        t_feat = self.time_mlp(t).unsqueeze(-1).expand(batch, -1, frames)
        e_feat = self.emotion_embed(emotion_id.to(x_s.device)).to(dtype=x_s.dtype).unsqueeze(-1).expand(batch, -1, frames)
        stacked = torch.cat([x_s, mu, neutral_cond, delta_cond, t_feat, e_feat], dim=1)
        return self.net(stacked)


def residual_stats(delta_v: torch.Tensor, v_base: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> EmotionAdapterStats:
    masked_delta = delta_v * mask
    masked_base = v_base * mask
    residual_norm = torch.linalg.vector_norm(masked_delta.flatten(1), dim=1)
    base_norm = torch.linalg.vector_norm(masked_base.flatten(1), dim=1)
    cosine = torch.nn.functional.cosine_similarity(masked_delta.flatten(1), masked_base.flatten(1), dim=1, eps=eps)
    return EmotionAdapterStats(
        residual_norm=residual_norm,
        base_norm=base_norm,
        norm_ratio=residual_norm / (base_norm + eps),
        cosine_to_base=cosine,
    )
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:

```bash
pytest tests/ctm_emotion_flow/test_emotion_adapter.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add cosyvoice/flow/emotion_adapter.py tests/ctm_emotion_flow/test_emotion_adapter.py
git commit -m "feat: add emotion velocity adapter"
```

## Task 2: Residual Loss Helper

**Files:**
- Create: `cosyvoice/flow/emotion_guided_flow.py`
- Test: `tests/ctm_emotion_flow/test_adapter_loss.py`

- [ ] **Step 1: Write the residual loss test**

```python
import torch

from cosyvoice.flow.emotion_guided_flow import compute_adapter_residual_loss


class FakeEstimator(torch.nn.Module):
    def forward(self, x, mask, mu, t, spks, cond, streaming=False):
        return torch.ones_like(x) * 0.25


class FakeAdapter(torch.nn.Module):
    def forward(self, *, x_s, mu, neutral_cond, delta_cond, flow_s, emotion_id):
        return torch.ones_like(x_s) * 0.75


def test_residual_loss_matches_target_residual():
    batch, frames = 1, 5
    x1_emo = torch.ones(batch, 80, frames)
    z = torch.zeros_like(x1_emo)
    mask = torch.ones(batch, 1, frames)
    mu = torch.zeros_like(x1_emo)
    spks = torch.zeros(batch, 80)
    neutral_cond = torch.zeros_like(x1_emo)
    delta_cond = torch.ones_like(x1_emo)
    flow_s = torch.tensor([0.5])
    emotion_id = torch.tensor([1])
    loss, stats = compute_adapter_residual_loss(
        estimator=FakeEstimator(),
        adapter=FakeAdapter(),
        x1_emo=x1_emo,
        z=z,
        mask=mask,
        mu=mu,
        spks=spks,
        neutral_cond=neutral_cond,
        delta_cond=delta_cond,
        flow_s=flow_s,
        emotion_id=emotion_id,
        sigma_min=1e-6,
    )
    assert loss.item() < 1e-5
    assert stats["v_base_shape"] == [1, 80, 5]
    assert stats["delta_v_shape"] == [1, 80, 5]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest tests/ctm_emotion_flow/test_adapter_loss.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'cosyvoice.flow.emotion_guided_flow'
```

- [ ] **Step 3: Implement the residual loss helper**

```python
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def sample_flow_state(x1: torch.Tensor, z: torch.Tensor, flow_s: torch.Tensor, sigma_min: float) -> Tuple[torch.Tensor, torch.Tensor]:
    s = flow_s.reshape(-1, 1, 1).to(device=x1.device, dtype=x1.dtype)
    x_s = (1 - (1 - sigma_min) * s) * z + s * x1
    u_s = x1 - (1 - sigma_min) * z
    return x_s, u_s


def compute_adapter_residual_loss(
    *,
    estimator: torch.nn.Module,
    adapter: torch.nn.Module,
    x1_emo: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    neutral_cond: torch.Tensor,
    delta_cond: torch.Tensor,
    flow_s: torch.Tensor,
    emotion_id: torch.Tensor,
    sigma_min: float,
    generated_region_mask: torch.Tensor | None = None,
    streaming: bool = False,
) -> tuple[torch.Tensor, Dict[str, object]]:
    x_s, u_s = sample_flow_state(x1_emo, z, flow_s, sigma_min)
    with torch.no_grad():
        v_base = estimator(x_s, mask, mu, flow_s, spks, neutral_cond, streaming=streaming)
    delta_target = (u_s - v_base).detach()
    delta_v = adapter(
        x_s=x_s,
        mu=mu,
        neutral_cond=neutral_cond,
        delta_cond=delta_cond,
        flow_s=flow_s,
        emotion_id=emotion_id,
    )
    active_mask = mask if generated_region_mask is None else mask * generated_region_mask
    loss = F.mse_loss(delta_v * active_mask, delta_target * active_mask, reduction="sum") / (
        torch.sum(active_mask) * delta_v.shape[1]
    )
    stats = {
        "v_base_shape": list(v_base.shape),
        "delta_v_shape": list(delta_v.shape),
        "delta_target_shape": list(delta_target.shape),
        "loss": float(loss.detach().cpu()),
    }
    return loss, stats
```

- [ ] **Step 4: Run the test and confirm it passes**

Run:

```bash
pytest tests/ctm_emotion_flow/test_adapter_loss.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add cosyvoice/flow/emotion_guided_flow.py tests/ctm_emotion_flow/test_adapter_loss.py
git commit -m "feat: add adapter residual loss"
```

## Task 3: Alpha Injection Contract

**Files:**
- Modify: `cosyvoice/flow/emotion_guided_flow.py`
- Test: `tests/ctm_emotion_flow/test_alpha_injection.py`

- [ ] **Step 1: Write alpha injection tests**

```python
import torch

from cosyvoice.flow.emotion_guided_flow import apply_emotion_residual


def test_alpha_zero_returns_base_velocity():
    v_base = torch.randn(1, 80, 6)
    delta_v = torch.randn(1, 80, 6)
    mask = torch.ones(1, 1, 6)
    out = apply_emotion_residual(v_base=v_base, delta_v=delta_v, alpha=0.0, mask=mask)
    assert torch.allclose(out, v_base)


def test_residual_applies_only_to_generated_region():
    v_base = torch.zeros(1, 80, 6)
    delta_v = torch.ones(1, 80, 6)
    mask = torch.tensor([[[0, 0, 1, 1, 1, 1]]], dtype=torch.float32)
    out = apply_emotion_residual(v_base=v_base, delta_v=delta_v, alpha=2.0, mask=mask)
    assert out[:, :, :2].sum().item() == 0
    assert torch.allclose(out[:, :, 2:], torch.ones_like(out[:, :, 2:]) * 2.0)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest tests/ctm_emotion_flow/test_alpha_injection.py -q
```

Expected:

```text
ImportError: cannot import name 'apply_emotion_residual'
```

- [ ] **Step 3: Implement `apply_emotion_residual`**

```python
def apply_emotion_residual(*, v_base: torch.Tensor, delta_v: torch.Tensor, alpha: float, mask: torch.Tensor) -> torch.Tensor:
    return v_base + float(alpha) * delta_v * mask
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/ctm_emotion_flow/test_alpha_injection.py tests/ctm_emotion_flow/test_adapter_loss.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add cosyvoice/flow/emotion_guided_flow.py tests/ctm_emotion_flow/test_alpha_injection.py
git commit -m "feat: define emotion residual alpha contract"
```

## Task 4: Tiny Config

**Files:**
- Create: `examples/ctm_emotion_flow/config/adapter_tiny.yaml`

- [ ] **Step 1: Create the config**

```yaml
project_root: /test1208/zw/ctm_emotion_tts
cosyvoice_repo: /test1208/zw/ctm_emotion_tts/repos/CosyVoice
model_dir: /home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512

data:
  manifest: /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.jsonl
  feature_index: /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.jsonl
  emotions:
    neutral: 0
    angry: 1
    happy: 2
    sad: 3

adapter:
  mel_dim: 80
  hidden_dim: 256
  time_dim: 16
  emotion_dim: 16
  num_emotions: 4
  zero_init: true

training:
  seed: 20260531
  batch_size: 2
  max_steps: 1000
  learning_rate: 0.0001
  sigma_min: 0.000001
  alpha_train: 1.0
  log_every: 20
  save_every: 200
  output_dir: /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit

eval:
  alphas: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
  output_dir: /test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder
```

- [ ] **Step 2: Validate the config file parses**

Run:

```bash
python - <<'PY'
from pathlib import Path
import yaml
path = Path("examples/ctm_emotion_flow/config/adapter_tiny.yaml")
cfg = yaml.safe_load(path.read_text())
assert cfg["adapter"]["mel_dim"] == 80
assert cfg["data"]["emotions"]["happy"] == 2
print("config ok")
PY
```

Expected:

```text
config ok
```

- [ ] **Step 3: Commit**

```bash
git add examples/ctm_emotion_flow/config/adapter_tiny.yaml
git commit -m "chore: add tiny emotion adapter config"
```

## Task 5: Synthetic Parallel Manifest Builder

**Files:**
- Create: `examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py`

- [ ] **Step 1: Implement manifest schema validation before generation**

```python
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class ParallelItem:
    utt_id: str
    speaker_id: str
    text: str
    neutral_wav: str
    emotion_wav: str
    emotion: str
    emotion_intensity: float
    prompt_wav: str
    generator: str
    sample_rate: int


def write_manifest(items: list[ParallelItem], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
```

- [ ] **Step 2: Add CLI dry-run mode**

```python
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    item = ParallelItem(
        utt_id="dryrun_0001_happy",
        speaker_id="spk_0001",
        text="今天的系统测试用于验证情绪强度控制。",
        neutral_wav="/test1208/zw/ctm_emotion_tts/data/parallel_synth/spk_0001/text_0001/neutral.wav",
        emotion_wav="/test1208/zw/ctm_emotion_tts/data/parallel_synth/spk_0001/text_0001/happy.wav",
        emotion="happy",
        emotion_intensity=1.0,
        prompt_wav="/test1208/zw/ctm_emotion_tts/data/references/spk_0001.wav",
        generator="dry-run",
        sample_rate=24000,
    )
    write_manifest([item], Path(args.out_manifest))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run dry-run manifest**

Run:

```bash
python examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py \
  --dry-run \
  --out-manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.dryrun.jsonl
```

Expected:

```text
file exists: /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.dryrun.jsonl
```

- [ ] **Step 4: Add real CosyVoice generation after dry-run passes**

Use the existing first-round inference pattern from `/test1208/zw/ctm_emotion_tts/generate_extreme_emotion_pair.py`:

```python
from cosyvoice.cli.cosyvoice import CosyVoice3
```

The generation function must save neutral first, then angry/happy/sad for the same `text` and `prompt_wav`. It must write every wav path to the manifest and skip no errors silently.

- [ ] **Step 5: Commit**

```bash
git add examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py
git commit -m "feat: add synthetic parallel manifest builder"
```

## Task 6: Feature Cache Builder

**Files:**
- Create: `examples/ctm_emotion_flow/prepare_parallel_features.py`

- [ ] **Step 1: Implement cached feature record writer**

```python
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import torch


@dataclass
class FeatureItem:
    utt_id: str
    feature_path: str
    emotion: str
    emotion_id: int
    text: str


def save_feature_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, out_path)


def write_index(items: list[FeatureItem], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
```

- [ ] **Step 2: Store tensors with exact keys**

Each `.pt` record must contain:

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

- [ ] **Step 3: Add fake-tensor smoke mode**

Run:

```bash
python examples/ctm_emotion_flow/prepare_parallel_features.py \
  --fake-smoke \
  --out-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.fake.jsonl
```

Expected:

```text
wrote 1 fake feature record
```

- [ ] **Step 4: Commit**

```bash
git add examples/ctm_emotion_flow/prepare_parallel_features.py
git commit -m "feat: add parallel feature cache builder"
```

## Task 7: Adapter Training Script

**Files:**
- Create: `examples/ctm_emotion_flow/train_flow_emotion_adapter.py`

- [ ] **Step 1: Implement dataset loader**

```python
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ParallelFeatureDataset(Dataset):
    def __init__(self, index_path: str):
        self.items = [json.loads(line) for line in Path(index_path).read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return torch.load(item["feature_path"], map_location="cpu")
```

- [ ] **Step 2: Implement collate for batch size 1 first**

```python
def collate_one(batch):
    if len(batch) != 1:
        raise ValueError("first training pass uses batch_size=1 until padding is implemented")
    item = batch[0]
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}
```

- [ ] **Step 3: Freeze CosyVoice estimator and train adapter only**

```python
for parameter in estimator.parameters():
    parameter.requires_grad = False
adapter.train()
optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["training"]["learning_rate"])
```

- [ ] **Step 4: Log residual stats every step**

Write JSONL rows to:

```text
/test1208/zw/ctm_emotion_tts/logs/emotion_flow_adapter_train.jsonl
```

Each row:

```json
{"step": 1, "loss": 0.123, "emotion": "happy", "residual_norm": 4.2, "base_norm": 80.1, "norm_ratio": 0.052}
```

- [ ] **Step 5: Save adapter checkpoint**

Save:

```python
torch.save(
    {
        "adapter": adapter.state_dict(),
        "config": cfg,
        "step": step,
    },
    output_dir / "adapter.pt",
)
```

- [ ] **Step 6: Run fake smoke training**

Run:

```bash
python examples/ctm_emotion_flow/train_flow_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.fake.jsonl \
  --output-dir /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/fake_smoke \
  --max-steps 2
```

Expected:

```text
step=1 loss=
step=2 loss=
saved adapter.pt
```

- [ ] **Step 7: Commit**

```bash
git add examples/ctm_emotion_flow/train_flow_emotion_adapter.py
git commit -m "feat: add emotion adapter training script"
```

## Task 8: Alpha Ladder Evaluation

**Files:**
- Create: `examples/ctm_emotion_flow/eval_emotion_adapter.py`

- [ ] **Step 1: Implement alpha loop contract**

```python
def parse_alphas(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")
```

- [ ] **Step 2: Save per-alpha output manifest**

Each generated row:

```json
{
  "utt_id": "synth_000001_happy",
  "emotion": "happy",
  "alpha": 0.75,
  "wav": "/test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder/synth_000001_happy/alpha_0p75.wav",
  "emotion2vec_label": "happy",
  "emotion2vec_score": 0.81,
  "duration_sec": 4.2
}
```

- [ ] **Step 3: Add fake eval mode**

Run:

```bash
python examples/ctm_emotion_flow/eval_emotion_adapter.py \
  --fake-smoke \
  --alphas 0 0.25 0.5 1.0 \
  --out-dir /test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder_fake
```

Expected:

```text
wrote evaluation manifest
```

- [ ] **Step 4: Commit**

```bash
git add examples/ctm_emotion_flow/eval_emotion_adapter.py
git commit -m "feat: add alpha ladder evaluation script"
```

## Task 9: Real Tiny Run Gate

**Files:**
- No new files.
- Uses scripts from Tasks 5-8.

- [ ] **Step 1: Check GPU0**

Run:

```bash
nvidia-smi -i 0
```

Expected:

```text
GPU 0 visible; no unexpected high-memory process started by this experiment
```

- [ ] **Step 2: Build a 12-pair synthetic manifest**

Run:

```bash
python examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py \
  --project-root /test1208/zw/ctm_emotion_tts \
  --model-dir /home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512 \
  --out-manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.tiny.jsonl \
  --num-texts 4 \
  --emotions angry happy sad
```

Expected:

```text
12 emotional pairs plus neutral wavs
```

- [ ] **Step 3: Prepare real feature cache**

Run:

```bash
python examples/ctm_emotion_flow/prepare_parallel_features.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.tiny.jsonl \
  --out-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.tiny.jsonl
```

Expected:

```text
wrote 12 feature records
```

- [ ] **Step 4: Train tiny overfit**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ctm_emotion_flow/train_flow_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.tiny.jsonl \
  --output-dir /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit \
  --max-steps 1000 \
  --batch-size 1
```

Expected:

```text
loss decreases relative to first 20-step moving average
adapter.pt saved
```

- [ ] **Step 5: Evaluate alpha ladder**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ctm_emotion_flow/eval_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --checkpoint /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit/adapter.pt \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.tiny.jsonl \
  --out-dir /test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder \
  --alphas 0 0.25 0.5 0.75 1.0 1.25
```

Expected:

```text
evaluation manifest and wavs saved
```

## Task 10: Report Update Gate

**Files:**
- Modify: `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md`
- Modify: `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md`

- [ ] **Step 1: Add actual run evidence**

Record:

```text
GPU0 status before run:
commands:
manifest path:
feature index path:
adapter checkpoint:
alpha ladder output:
loss trend:
emotion2vec trend:
content/speaker checks:
blockers:
route decision:
```

- [ ] **Step 2: Run privacy scan**

Run:

```bash
python - <<'PY'
from pathlib import Path

terms_path = Path("/test1208/zw/ctm_emotion_tts/private/privacy_terms.txt")
terms = [line.strip() for line in terms_path.read_text().splitlines() if line.strip()] if terms_path.exists() else []
roots = [Path("docs/ctm_emotion_flow"), Path("docs/superpowers/plans")]
hits = []
for root in roots:
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in terms:
            if term in text:
                hits.append(f"{path}: contains private term")
if hits:
    raise SystemExit("\n".join(hits))
print("privacy scan ok")
PY
```

Expected:

```text
privacy scan ok
```

- [ ] **Step 3: Commit**

```bash
git add docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md
git commit -m "docs: report emotion adapter training probe"
```

## Self-Review

Spec coverage:

- Adapter-only training: covered by Tasks 1-3 and 7.
- Same-text same-speaker neutral/emotional data: covered by Task 5.
- Parallel feature preparation: covered by Task 6.
- Alpha-controlled flow generation: covered by Tasks 3 and 8.
- No full CosyVoice training: all tasks freeze the base model.
- Detailed and brief documentation: covered by Task 10.

Placeholder scan:

- No `TBD` or unresolved implementation placeholders are left.
- The real generation function in Task 5 is intentionally gated after dry-run and points to the existing local inference pattern to avoid duplicating unstable inference code in the plan.

Type consistency:

- `flow_s` is used for flow time.
- Mel/speech frame axis is consistently `K`.
- Adapter output is always `[B, 80, K]`.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-05-31-flow-emotion-adapter-training.md`.

Two execution options:

1. Subagent-Driven: dispatch a fresh worker per task and review after each task.
2. Inline Execution: execute this plan in the current session with checkpoints.

Recommended first execution target: Task 1 through Task 4 only, then stop for review before touching real GPU generation or training.
