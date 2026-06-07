"""Paired (emo, neu) ESD dataset for SVD flow experiments.

Each batch element provides:
  - speech_token_emo, speech_token_neu:   (N_tok_emo,), (N_tok_neu,)   int32
  - speech_feat_emo,  speech_feat_neu:    (K_emo, 80), (K_neu, 80)     float32 mel
  - embedding_emo,    embedding_neu:      (192,), (192,)               speaker x-vec
  - emotion_label:                         int (0..4)
  - text:                                  str
  - key:                                   str

For training we use the emo side as the supervised CFM target. The neu side is
needed only for the L_sem / L_emo proxies (time-averaged mel and velocity
direction).

We pre-extract speech_token / speech_feat / embedding once into a feature
cache directory to avoid repeating the ONNX speech tokenizer and Whisper-mel
extraction every epoch. See `precompute_features.py` for the cache builder.

This loader reads from the cache; if a key's cache is missing it falls back to
on-the-fly extraction (slower; logged warning).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)

EMOTION_TO_ID = {"Neutral": 0, "Happy": 1, "Angry": 2, "Sad": 3, "Surprise": 4}


class ESDParallelDataset(Dataset):
    """Reads ESD parallel manifest and serves (emo, neu) pairs.

    Args:
        manifest:   path to JSONL produced by build_esd_manifest.py
        feature_dir: directory of pre-extracted .pt files (per key, per emotion)
        target_emotions: list of emotions to use as "emo" side. Default: all 4
                         non-neutral.
        split: 'train' | 'evaluation' | 'test' | 'all'
        max_feat_len: if set, drop records whose either side exceeds this
                      (in mel frames). Avoids OOM on long utterances.
        require_cache: if True, raise on missing cache; else fall back to on-the-fly.
    """

    def __init__(
        self,
        manifest: str,
        feature_dir: str,
        target_emotions: Optional[List[str]] = None,
        split: str = "train",
        max_feat_len: int = 1200,
        min_feat_len: int = 40,
        require_cache: bool = True,
    ):
        super().__init__()
        self.manifest = Path(manifest)
        self.feature_dir = Path(feature_dir)
        self.target_emotions = target_emotions or ["Happy", "Angry", "Sad", "Surprise"]
        self.split = split
        self.max_feat_len = max_feat_len
        self.min_feat_len = min_feat_len
        self.require_cache = require_cache

        self.records: List[Dict[str, Any]] = []
        with open(self.manifest, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if split != "all" and r.get("split") != split:
                    continue
                if "Neutral" not in r["emotions"]:
                    continue
                # one (emo, neu) pair per target emotion present
                for emo in self.target_emotions:
                    if emo not in r["emotions"]:
                        continue
                    self.records.append({**r, "emotion_pair": (emo, "Neutral")})
        LOG.info(f"ESDParallelDataset({split}): {len(self.records)} pairs across "
                 f"{len(self.target_emotions)} target emotions")

    def __len__(self) -> int:
        return len(self.records)

    def _load_cache(self, key: str, emotion: str) -> Dict[str, torch.Tensor]:
        """Load cached features for one (key, emotion). Returns dict of tensors."""
        path = self.feature_dir / f"{key}_{emotion}.pt"
        if not path.exists():
            if self.require_cache:
                raise FileNotFoundError(f"Feature cache missing: {path}")
            LOG.warning(f"Cache miss: {path}; on-the-fly fallback not implemented")
            raise NotImplementedError("On-the-fly extraction not wired yet")
        return torch.load(path, map_location="cpu", weights_only=True)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        rec = self.records[idx]
        emo_name, neu_name = rec["emotion_pair"]
        try:
            f_emo = self._load_cache(rec["key"], emo_name)
            f_neu = self._load_cache(rec["key"], neu_name)
        except FileNotFoundError as e:
            LOG.warning(f"Skip {rec['key']} ({emo_name}): {e}")
            return None

        # Length filter
        K_emo = f_emo["speech_feat"].shape[0]
        K_neu = f_neu["speech_feat"].shape[0]
        if max(K_emo, K_neu) > self.max_feat_len or min(K_emo, K_neu) < self.min_feat_len:
            return None

        return {
            "key": rec["key"],
            "speaker": rec["speaker"],
            "text": rec["text"],
            "lang": rec["lang"],
            "emotion": emo_name,
            "emotion_id": EMOTION_TO_ID[emo_name],
            "speech_token_emo": f_emo["speech_token"],   # (N_tok,)
            "speech_token_neu": f_neu["speech_token"],
            "speech_feat_emo": f_emo["speech_feat"],     # (K, 80)
            "speech_feat_neu": f_neu["speech_feat"],
            "embedding_emo": f_emo["embedding"],         # (192,)
            "embedding_neu": f_neu["embedding"],
        }


def collate_parallel(batch: List[Optional[dict]]) -> Optional[dict]:
    """Pad-collate a list of (emo, neu) pair samples.

    Returns dict with batched tensors. The (B,80,K) mels and (B,N_tok) tokens
    are padded with zeros to max length within the batch. Lengths are recorded
    so downstream code can build masks.

    None entries (filtered out by __getitem__) are dropped; if the whole batch
    is empty the function returns None.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    def stack_pad(tensors: List[torch.Tensor], pad_value: float = 0.0,
                  time_dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        lens = torch.tensor([t.shape[time_dim] for t in tensors], dtype=torch.int32)
        max_len = int(lens.max().item())
        padded = []
        for t in tensors:
            pad_amt = max_len - t.shape[time_dim]
            if pad_amt > 0:
                if t.dim() == 1:
                    p = torch.nn.functional.pad(t, (0, pad_amt), value=pad_value)
                else:
                    # pad last dim before time_dim assumed -- for our (K, 80) mel, time is dim 0
                    pad_spec = [0, 0] * (t.dim() - 1 - time_dim) + [0, pad_amt]
                    p = torch.nn.functional.pad(t, tuple(pad_spec), value=pad_value)
            else:
                p = t
            padded.append(p)
        return torch.stack(padded, dim=0), lens

    out: Dict[str, Any] = {
        "keys": [b["key"] for b in batch],
        "speakers": [b["speaker"] for b in batch],
        "texts": [b["text"] for b in batch],
        "langs": [b["lang"] for b in batch],
        "emotions": [b["emotion"] for b in batch],
        "emotion_ids": torch.tensor([b["emotion_id"] for b in batch], dtype=torch.long),
    }

    out["speech_token_emo"], out["speech_token_emo_len"] = stack_pad(
        [b["speech_token_emo"] for b in batch])
    out["speech_token_neu"], out["speech_token_neu_len"] = stack_pad(
        [b["speech_token_neu"] for b in batch])

    # mels are (K, 80) -> stack to (B, K_max, 80), len dim = 0
    feat_emo, feat_emo_len = stack_pad([b["speech_feat_emo"] for b in batch], time_dim=0)
    feat_neu, feat_neu_len = stack_pad([b["speech_feat_neu"] for b in batch], time_dim=0)
    out["speech_feat_emo"] = feat_emo      # (B, K_emo, 80)
    out["speech_feat_emo_len"] = feat_emo_len
    out["speech_feat_neu"] = feat_neu
    out["speech_feat_neu_len"] = feat_neu_len

    out["embedding_emo"] = torch.stack([b["embedding_emo"] for b in batch], dim=0)
    out["embedding_neu"] = torch.stack([b["embedding_neu"] for b in batch], dim=0)

    return out
