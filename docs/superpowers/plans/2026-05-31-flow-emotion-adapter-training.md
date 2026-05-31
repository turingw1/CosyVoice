# 情绪 Flow Adapter 训练实现计划

> **给后续 agent 的要求：** 实现本计划时，必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步执行。所有步骤使用 checkbox 追踪。

**目标：** 在不训练完整 CosyVoice 的前提下，增加一个小型 adapter-only 训练流程，验证 CosyVoice3 flow velocity 是否能接受一个可学习的情绪残差通道。

**架构：** CosyVoice3 主体全部冻结，只在 flow solver 外侧增加 zero-init `EmotionVelocityAdapter`。训练时用 neutral/emotional 平行 mel pair 回归 frozen base velocity 的 residual；评估时用 `alpha` 控制情绪强度。

**技术栈：** Python、PyTorch、CosyVoice3、CosyVoice CFM/DiT flow、FunASR emotion2vec、pytest。

---

## 文件结构

- 新增 `cosyvoice/flow/emotion_adapter.py`
  - 小型 adapter 模块、生成区间 mask、residual 统计工具。
- 新增 `cosyvoice/flow/emotion_guided_flow.py`
  - 对 frozen `CausalConditionalCFM` 的非侵入式 wrapper；原始 CosyVoice solver 不改。
- 新增 `examples/ctm_emotion_flow/config/adapter_tiny.yaml`
  - 模型路径、数据路径、adapter 超参、训练和评估默认值。
- 新增 `examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py`
  - 生成或登记 synthetic neutral/emotional 平行 wav。
- 新增 `examples/ctm_emotion_flow/prepare_parallel_features.py`
  - 把 manifest 中的 wav 转成缓存的 mel/token/speaker feature record。
- 新增 `examples/ctm_emotion_flow/train_flow_emotion_adapter.py`
  - 只训练 adapter，保存 `adapter.pt`、日志和 config 副本。
- 新增 `examples/ctm_emotion_flow/eval_emotion_adapter.py`
  - 生成 alpha ladder wav，并记录 emotion/content/speaker 指标。
- 新增 `tests/ctm_emotion_flow/test_emotion_adapter.py`
  - shape、zero-init、masking 测试。
- 新增 `tests/ctm_emotion_flow/test_adapter_loss.py`
  - fake tensor 下的 residual target loss 测试。
- 新增 `tests/ctm_emotion_flow/test_alpha_injection.py`
  - 验证 `alpha=0` 和 generated-region-only injection。
- 新增 `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md`
  - 中文详细设计报告。
- 新增 `docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md`
  - 中文简要路线报告。

## Task 1: Adapter 模块

**文件：**
- 新增：`cosyvoice/flow/emotion_adapter.py`
- 测试：`tests/ctm_emotion_flow/test_emotion_adapter.py`

- [ ] **Step 1: 先写 shape 和 zero-init 测试**

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

- [ ] **Step 2: 运行测试，确认失败原因正确**

运行：

```bash
pytest tests/ctm_emotion_flow/test_emotion_adapter.py -q
```

预期：

```text
ModuleNotFoundError: No module named 'cosyvoice.flow.emotion_adapter'
```

- [ ] **Step 3: 实现 `EmotionVelocityAdapter`**

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

- [ ] **Step 4: 运行测试，确认通过**

运行：

```bash
pytest tests/ctm_emotion_flow/test_emotion_adapter.py -q
```

预期：

```text
2 passed
```

- [ ] **Step 5: 提交**

```bash
git add cosyvoice/flow/emotion_adapter.py tests/ctm_emotion_flow/test_emotion_adapter.py
git commit -m "feat: add emotion velocity adapter"
```

## Task 2: Residual Loss Helper

**文件：**
- 新增：`cosyvoice/flow/emotion_guided_flow.py`
- 测试：`tests/ctm_emotion_flow/test_adapter_loss.py`

- [ ] **Step 1: 写 residual loss 测试**

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

- [ ] **Step 2: 运行测试，确认失败原因正确**

运行：

```bash
pytest tests/ctm_emotion_flow/test_adapter_loss.py -q
```

预期：

```text
ModuleNotFoundError: No module named 'cosyvoice.flow.emotion_guided_flow'
```

- [ ] **Step 3: 实现 residual loss helper**

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

- [ ] **Step 4: 运行测试，确认通过**

运行：

```bash
pytest tests/ctm_emotion_flow/test_adapter_loss.py -q
```

预期：

```text
1 passed
```

- [ ] **Step 5: 提交**

```bash
git add cosyvoice/flow/emotion_guided_flow.py tests/ctm_emotion_flow/test_adapter_loss.py
git commit -m "feat: add adapter residual loss"
```

## Task 3: Alpha 注入契约

**文件：**
- 修改：`cosyvoice/flow/emotion_guided_flow.py`
- 测试：`tests/ctm_emotion_flow/test_alpha_injection.py`

- [ ] **Step 1: 写 alpha injection 测试**

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

- [ ] **Step 2: 运行测试，确认失败原因正确**

运行：

```bash
pytest tests/ctm_emotion_flow/test_alpha_injection.py -q
```

预期：

```text
ImportError: cannot import name 'apply_emotion_residual'
```

- [ ] **Step 3: 实现 `apply_emotion_residual`**

```python
def apply_emotion_residual(*, v_base: torch.Tensor, delta_v: torch.Tensor, alpha: float, mask: torch.Tensor) -> torch.Tensor:
    return v_base + float(alpha) * delta_v * mask
```

- [ ] **Step 4: 运行测试**

运行：

```bash
pytest tests/ctm_emotion_flow/test_alpha_injection.py tests/ctm_emotion_flow/test_adapter_loss.py -q
```

预期：

```text
3 passed
```

- [ ] **Step 5: 提交**

```bash
git add cosyvoice/flow/emotion_guided_flow.py tests/ctm_emotion_flow/test_alpha_injection.py
git commit -m "feat: define emotion residual alpha contract"
```

## Task 4: Tiny Config

**文件：**
- 新增：`examples/ctm_emotion_flow/config/adapter_tiny.yaml`

- [ ] **Step 1: 创建 config**

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
  batch_size: 1
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

- [ ] **Step 2: 验证 config 可以解析**

运行：

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

预期：

```text
config ok
```

- [ ] **Step 3: 提交**

```bash
git add examples/ctm_emotion_flow/config/adapter_tiny.yaml
git commit -m "chore: add tiny emotion adapter config"
```

## Task 5: Synthetic Parallel Manifest Builder

**文件：**
- 新增：`examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py`

- [ ] **Step 1: 先实现 manifest schema 和写入函数**

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

- [ ] **Step 2: 增加 dry-run CLI**

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

- [ ] **Step 3: 运行 dry-run manifest**

运行：

```bash
python examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py \
  --dry-run \
  --out-manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.dryrun.jsonl
```

预期：

```text
生成 /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.dryrun.jsonl
```

- [ ] **Step 4: dry-run 通过后再接入真实 CosyVoice 生成**

参考已有脚本：

```text
/test1208/zw/ctm_emotion_tts/generate_extreme_emotion_pair.py
```

真实生成函数必须：

- 对同一 text 和同一 prompt_wav 先生成 neutral。
- 再生成 angry / happy / sad。
- 每个 wav 保存路径写入 manifest。
- 每个错误显式记录，不允许静默跳过。

- [ ] **Step 5: 提交**

```bash
git add examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py
git commit -m "feat: add synthetic parallel manifest builder"
```

## Task 6: Feature Cache Builder

**文件：**
- 新增：`examples/ctm_emotion_flow/prepare_parallel_features.py`

- [ ] **Step 1: 实现 feature record 和 index 写入**

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

- [ ] **Step 2: 固定 `.pt` record key**

每个 `.pt` record 必须包含：

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

- [ ] **Step 3: 增加 fake-tensor smoke mode**

运行：

```bash
python examples/ctm_emotion_flow/prepare_parallel_features.py \
  --fake-smoke \
  --out-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.fake.jsonl
```

预期：

```text
wrote 1 fake feature record
```

- [ ] **Step 4: 提交**

```bash
git add examples/ctm_emotion_flow/prepare_parallel_features.py
git commit -m "feat: add parallel feature cache builder"
```

## Task 7: Adapter 训练脚本

**文件：**
- 新增：`examples/ctm_emotion_flow/train_flow_emotion_adapter.py`

- [ ] **Step 1: 实现 dataset loader**

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

- [ ] **Step 2: 第一版只支持 batch size 1**

```python
def collate_one(batch):
    if len(batch) != 1:
        raise ValueError("first training pass uses batch_size=1 until padding is implemented")
    item = batch[0]
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}
```

- [ ] **Step 3: 冻结 CosyVoice estimator，只训练 adapter**

```python
for parameter in estimator.parameters():
    parameter.requires_grad = False
adapter.train()
optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["training"]["learning_rate"])
```

- [ ] **Step 4: 每步记录 residual stats**

日志路径：

```text
/test1208/zw/ctm_emotion_tts/logs/emotion_flow_adapter_train.jsonl
```

每行格式：

```json
{"step": 1, "loss": 0.123, "emotion": "happy", "residual_norm": 4.2, "base_norm": 80.1, "norm_ratio": 0.052}
```

- [ ] **Step 5: 保存 adapter checkpoint**

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

- [ ] **Step 6: 运行 fake smoke training**

运行：

```bash
python examples/ctm_emotion_flow/train_flow_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.fake.jsonl \
  --output-dir /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/fake_smoke \
  --max-steps 2
```

预期：

```text
step=1 loss=
step=2 loss=
saved adapter.pt
```

- [ ] **Step 7: 提交**

```bash
git add examples/ctm_emotion_flow/train_flow_emotion_adapter.py
git commit -m "feat: add emotion adapter training script"
```

## Task 8: Alpha Ladder 评估

**文件：**
- 新增：`examples/ctm_emotion_flow/eval_emotion_adapter.py`

- [ ] **Step 1: 实现 alpha 解析**

```python
def parse_alphas(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")
```

- [ ] **Step 2: 保存 per-alpha output manifest**

每条生成记录：

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

- [ ] **Step 3: 增加 fake eval mode**

运行：

```bash
python examples/ctm_emotion_flow/eval_emotion_adapter.py \
  --fake-smoke \
  --alphas 0 0.25 0.5 1.0 \
  --out-dir /test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder_fake
```

预期：

```text
wrote evaluation manifest
```

- [ ] **Step 4: 提交**

```bash
git add examples/ctm_emotion_flow/eval_emotion_adapter.py
git commit -m "feat: add alpha ladder evaluation script"
```

## Task 9: 真实 Tiny Run Gate

**文件：**
- 不新增文件。
- 使用 Task 5-8 的脚本。

- [ ] **Step 1: 检查 GPU0**

运行：

```bash
nvidia-smi -i 0
```

预期：

```text
GPU0 可见，且没有本实验之外的异常高显存占用需要处理。
```

- [ ] **Step 2: 构建 12-pair synthetic manifest**

运行：

```bash
python examples/ctm_emotion_flow/build_synthetic_parallel_manifest.py \
  --project-root /test1208/zw/ctm_emotion_tts \
  --model-dir /home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512 \
  --out-manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.tiny.jsonl \
  --num-texts 4 \
  --emotions angry happy sad
```

预期：

```text
生成 12 条 emotional pair 和对应 neutral wav。
```

- [ ] **Step 3: 准备真实 feature cache**

运行：

```bash
python examples/ctm_emotion_flow/prepare_parallel_features.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --manifest /test1208/zw/ctm_emotion_tts/data/parallel_synth/manifest.tiny.jsonl \
  --out-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.tiny.jsonl
```

预期：

```text
wrote 12 feature records
```

- [ ] **Step 4: tiny overfit**

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ctm_emotion_flow/train_flow_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.tiny.jsonl \
  --output-dir /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit \
  --max-steps 1000 \
  --batch-size 1
```

预期：

```text
loss 相比前 20 step moving average 有下降，adapter.pt 已保存。
```

- [ ] **Step 5: alpha ladder 评估**

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/ctm_emotion_flow/eval_emotion_adapter.py \
  --config examples/ctm_emotion_flow/config/adapter_tiny.yaml \
  --checkpoint /test1208/zw/ctm_emotion_tts/models/emotion_flow_adapter/tiny_overfit/adapter.pt \
  --feature-index /test1208/zw/ctm_emotion_tts/data/parallel_synth/features/index.tiny.jsonl \
  --out-dir /test1208/zw/ctm_emotion_tts/outputs/emotion_flow_adapter_alpha_ladder \
  --alphas 0 0.25 0.5 0.75 1.0 1.25
```

预期：

```text
生成 evaluation manifest 和 wav。
```

## Task 10: 报告更新 Gate

**文件：**
- 修改：`docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md`
- 修改：`docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md`

- [ ] **Step 1: 写入真实运行证据**

必须记录：

```text
GPU0 状态：
运行命令：
manifest 路径：
feature index 路径：
adapter checkpoint：
alpha ladder 输出：
loss 趋势：
emotion2vec 趋势：
content/speaker 检查：
blocker：
路线判断：
```

- [ ] **Step 2: 运行隐私扫描**

运行：

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

预期：

```text
privacy scan ok
```

- [ ] **Step 3: 提交**

```bash
git add docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_DETAILED_REPORT.md docs/ctm_emotion_flow/FLOW_EMOTION_ADAPTER_BRIEF.md
git commit -m "docs: report emotion adapter training probe"
```

## 自检

覆盖情况：

- adapter-only 训练：Task 1-3 和 Task 7 覆盖。
- 同文本同说话人 neutral/emotional 数据：Task 5 覆盖。
- 平行 feature 准备：Task 6 覆盖。
- alpha-controlled flow generation：Task 3 和 Task 8 覆盖。
- 不训练完整 CosyVoice：所有任务都冻结 base model。
- 中文详细和简要文档：Task 10 覆盖。

占位检查：

- 没有未解决的占位内容。
- Task 5 的真实生成函数故意放在 dry-run 之后接入，避免先写不稳定推理逻辑。

类型一致性：

- flow 时间统一写作 `flow_s` 或 `s`。
- mel/speech frame 轴统一写作 `K`。
- adapter 输出始终为 `[B,80,K]`。

## 执行选择

计划已保存到：

```text
docs/superpowers/plans/2026-05-31-flow-emotion-adapter-training.md
```

建议后续执行顺序：

1. 先执行 Task 1-4，只做 adapter、loss helper、alpha contract 和 config。
2. 停下来 review。
3. 再执行 Task 5-8。
4. 最后在用户确认后执行 Task 9 的真实 GPU tiny run。
