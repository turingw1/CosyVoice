"""Train CosyVoice3 DiT to produce velocities that SVD-decompose meaningfully.

Loss (see plan.md v2):
  L = lambda_fm  * L_fm                        # CFM anchor — keeps TTS alive
    + lambda_sem * L_sem                       # time-avg of B ≈ time-avg of neu mel
    + lambda_emo * L_emo                       # A direction ≈ (emo - neu) direction

SVD convention (do NOT flip): B = top-k (semantic), A = residual (emotion).

Training surface:
  - Frozen: LLM, vocoder, flow.{input_embedding, pre_lookahead, encoder_proj,
            spk_embed_affine_layer, length_regulator}, flow.decoder (the CFM
            solver wrapper — has no parameters)
  - Trained: flow.decoder.estimator (the DiT, 22 layers)

Optimizer: AdamW lr=1e-5 (small), gradient accumulation if memory tight.

CFG handling: training_cfg_rate is forced to 0 during this experiment to keep
the loss formulation simple (no random condition dropout). Inference CFG can
be re-enabled later.

Run example:
  CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/train_svd_flow.py \\
      --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \\
      --feature_dir /test1208/zw/ctm_emotion_tts/data/esd_features \\
      --out_dir /test1208/zw/ctm_emotion_tts/outputs/svd_v2/train \\
      --epochs 5 --batch_size 4 --k_svd 16
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path("/test1208/zw/ctm_emotion_tts")
COSYVOICE_REPO = ROOT / "repos" / "CosyVoice"
MODEL_DIR = Path("/home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512")

sys.path.insert(0, str(COSYVOICE_REPO))
sys.path.insert(0, str(COSYVOICE_REPO / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel                                # noqa: E402
from cosyvoice.utils.mask import make_pad_mask                                # noqa: E402

from experiments.svd_emotion.dataset import ESDParallelDataset, collate_parallel  # noqa: E402
from experiments.svd_emotion.svd_decompose import svd_decompose               # noqa: E402

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger("train_svd_flow")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def freeze_all_but_dit(cosyvoice) -> int:
    """Freeze every parameter except the DiT estimator. Returns trainable count."""
    for p in cosyvoice.model.parameters():
        p.requires_grad_(False)
    estimator = cosyvoice.model.flow.decoder.estimator
    for p in estimator.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    return n_train


def build_flow_inputs(flow, batch: dict, device: torch.device) -> dict:
    """Run the L1 encoder pipeline to build (mu, spks, cond, x_1, K, mask).

    For training we use the EMO side as the supervised CFM target. mu / cond /
    spks all come from the emo wav's cached features.

    Returns:
      x_1:   (B, 80, K_emo) target mel for CFM
      mu:    (B, 80, K_emo) encoded semantic content
      spks:  (B, 80)        speaker embedding (post affine)
      cond:  (B, 80, K_emo) zero tensor (no prompt prefix in this training mode)
      mask:  (B, 1, K_emo)  valid-frame mask
      K_eff: (B,) effective length per sample
    """
    token = batch["speech_token_emo"].to(device).long()             # (B, N_tok_emo) — pad with 0 (clamped below)
    token_len = batch["speech_token_emo_len"].to(device)
    feat = batch["speech_feat_emo"].to(device)                      # (B, K_emo, 80)
    feat_len = batch["speech_feat_emo_len"].to(device)
    embedding = batch["embedding_emo"].to(device)                    # (B, 192)

    # speaker projection (matches inference path)
    embedding = F.normalize(embedding, dim=1)
    spks = flow.spk_embed_affine_layer(embedding)                   # (B, 80)

    # token mask + embedding lookup + pre_lookahead + repeat_interleave
    tok_pad_mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(spks)  # (B, N_tok, 1)
    token_emb = flow.input_embedding(torch.clamp(token, min=0)) * tok_pad_mask
    h = flow.pre_lookahead_layer(token_emb)                          # (B, N_tok, 80)
    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)             # (B, ~K, 80)

    # Align to feat length (the pre_lookahead -> repeat path can differ by 1-2 frames)
    K = min(h.shape[1], feat.shape[1])
    h = h[:, :K]                                                     # (B, K, 80)
    x_1 = feat[:, :K].transpose(1, 2).contiguous()                   # (B, 80, K)
    mu = h.transpose(1, 2).contiguous()                              # (B, 80, K)
    cond = torch.zeros_like(x_1)

    # Effective length per sample = min(2*N_tok, feat_len, K)
    K_eff = torch.minimum(token_len * flow.token_mel_ratio, feat_len).clamp(max=K)
    mask = (~make_pad_mask(K_eff)).to(spks).unsqueeze(1)              # (B, 1, K)

    return {"x_1": x_1, "mu": mu, "spks": spks, "cond": cond, "mask": mask, "K_eff": K_eff}


def cfm_perturb(x_1: torch.Tensor, sigma_min: float = 1e-6,
                t_min: float = 0.05, t_max: float = 0.95):
    """Sample CFM (y, t, u_true). Clamp t into [t_min, t_max] for stability."""
    B = x_1.shape[0]
    t = torch.rand(B, device=x_1.device, dtype=x_1.dtype) * (t_max - t_min) + t_min
    z = torch.randn_like(x_1)
    t_b = t.view(B, 1, 1)
    y = (1.0 - (1.0 - sigma_min) * t_b) * z + t_b * x_1
    u_true = x_1 - (1.0 - sigma_min) * z
    return y, t, z, u_true


def masked_mean_time(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Time-average a (B, C, K) tensor using a (B, 1, K) mask. Returns (B, C)."""
    num = (x * mask).sum(dim=-1)
    den = mask.sum(dim=-1).clamp(min=1e-6)
    return num / den


def compute_svd_loss(v_hat: torch.Tensor, u_true: torch.Tensor,
                     emo_feat: torch.Tensor, neu_feat: torch.Tensor,
                     emo_mask: torch.Tensor, neu_mask: torch.Tensor,
                     k_svd: int, lambdas=(1.0, 0.1, 0.1)):
    """Compute L = λ_fm·L_fm + λ_sem·L_sem + λ_emo·L_emo.

    Args:
      v_hat:    (B, 80, K_emo) model output
      u_true:   (B, 80, K_emo) CFM target velocity
      emo_feat: (B, 80, K_emo) emo mel target
      neu_feat: (B, 80, K_neu) parallel neu mel
      emo_mask: (B, 1, K_emo)
      neu_mask: (B, 1, K_neu)
      k_svd:    int, SVD top-k
      lambdas:  (λ_fm, λ_sem, λ_emo)

    Returns:
      total_loss, dict(L_fm, L_sem, L_emo, |A|/|v|, cos(A,target_dir))
    """
    # SVD decomposition (batched). With our convention: B = top-k = semantic,
    # A = residual = emotion.
    A, B_top, _ = svd_decompose(v_hat, k_svd)

    # ---- L_fm (masked MSE against true CFM velocity) ----
    sq = (v_hat - u_true).pow(2) * emo_mask
    denom = emo_mask.sum() * v_hat.shape[1] + 1e-6
    L_fm = sq.sum() / denom

    # ---- Time-averaged means (proxies for direction in mel/velocity space) ----
    B_mean = masked_mean_time(B_top, emo_mask)                       # (B, 80)
    A_mean = masked_mean_time(A, emo_mask)
    emo_mel_mean = masked_mean_time(emo_feat, emo_mask)
    neu_mel_mean = masked_mean_time(neu_feat, neu_mask)

    # ---- L_sem: B (semantic baseline) should match neutral mel direction ----
    L_sem = F.mse_loss(B_mean, neu_mel_mean)

    # ---- L_emo: A (emotion residual) should point along (emo - neu) ----
    target_dir = emo_mel_mean - neu_mel_mean
    # Normalize for cosine
    cos_A_target = F.cosine_similarity(A_mean, target_dir, dim=-1)   # (B,)
    L_emo = (1.0 - cos_A_target).mean()

    # ---- Total ----
    total = lambdas[0] * L_fm + lambdas[1] * L_sem + lambdas[2] * L_emo

    diag = {
        "L_fm": float(L_fm.detach()),
        "L_sem": float(L_sem.detach()),
        "L_emo": float(L_emo.detach()),
        "cos_A_target_mean": float(cos_A_target.mean().detach()),
        "A_over_v_ratio": float(
            (torch.linalg.vector_norm(A) /
             torch.linalg.vector_norm(v_hat).clamp(min=1e-6)).detach()
        ),
        "B_over_v_ratio": float(
            (torch.linalg.vector_norm(B_top) /
             torch.linalg.vector_norm(v_hat).clamp(min=1e-6)).detach()
        ),
    }
    return total, diag


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--k_svd", type=int, default=16)
    ap.add_argument("--lambda_fm", type=float, default=1.0)
    ap.add_argument("--lambda_sem", type=float, default=0.1)
    ap.add_argument("--lambda_emo", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every_epoch", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--smoke_test", action="store_true",
                    help="Tiny overfit: 100 samples, 200 steps, no ckpt")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    log_path = out_dir / "train.log.jsonl"

    LOG.info(f"loading CosyVoice3 from {MODEL_DIR}")
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR), fp16=False)
    flow = cosyvoice.model.flow
    device = cosyvoice.model.device
    LOG.info(f"device={device}")

    # Force training_cfg_rate to 0 (we want clean (mu, spks, cond) without dropout)
    flow.decoder.training_cfg_rate = 0.0
    LOG.info(f"training_cfg_rate forced to 0")

    n_train = freeze_all_but_dit(cosyvoice)
    LOG.info(f"trainable params (DiT only): {n_train/1e6:.2f} M")

    estimator = flow.decoder.estimator
    estimator.train()
    optimizer = torch.optim.AdamW(
        [p for p in estimator.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0
    )

    LOG.info(f"loading dataset from {args.manifest}")
    full_ds = ESDParallelDataset(
        manifest=args.manifest, feature_dir=args.feature_dir,
        split="train", max_feat_len=1200, min_feat_len=40,
    )
    if args.smoke_test:
        # Subsample 100 records for fast overfit verification
        from torch.utils.data import Subset
        idx = list(range(min(100, len(full_ds))))
        full_ds = Subset(full_ds, idx)
        LOG.warning(f"SMOKE TEST: using {len(full_ds)} samples")

    loader = DataLoader(
        full_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_parallel,
        pin_memory=True, drop_last=True,
    )

    step = 0
    t0 = time.time()
    log_f = open(log_path, "a", encoding="utf-8")

    for epoch in range(args.epochs):
        for batch in loader:
            if batch is None:
                continue

            # Build flow inputs from emo side
            with torch.no_grad():
                inp = build_flow_inputs(flow, batch, device)
            x_1 = inp["x_1"]
            mu = inp["mu"]
            spks = inp["spks"]
            cond = inp["cond"]
            mask = inp["mask"]

            # CFM perturbation
            y, t_samp, z, u_true = cfm_perturb(x_1)

            # Forward DiT (this is where gradients flow)
            v_hat = flow.decoder.estimator(
                y, mask, mu, t_samp, spks, cond, streaming=False
            )

            # Build neu-side feat tensor + mask (for L_sem / L_emo proxies)
            neu_feat = batch["speech_feat_neu"].to(device).transpose(1, 2)  # (B, 80, K_neu)
            neu_len = batch["speech_feat_neu_len"].to(device)
            neu_mask = (~make_pad_mask(neu_len, max_len=neu_feat.shape[-1])).to(spks).unsqueeze(1)
            emo_feat = x_1                                                   # already (B, 80, K_emo)
            emo_mask = mask

            loss, diag = compute_svd_loss(
                v_hat=v_hat, u_true=u_true,
                emo_feat=emo_feat, neu_feat=neu_feat,
                emo_mask=emo_mask, neu_mask=neu_mask,
                k_svd=args.k_svd,
                lambdas=(args.lambda_fm, args.lambda_sem, args.lambda_emo),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(estimator.parameters(), args.grad_clip)
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                rec = {
                    "step": step, "epoch": epoch,
                    "loss": float(loss.detach()),
                    "grad_norm": float(gn.detach()),
                    "lr": args.lr,
                    "elapsed_s": round(time.time() - t0, 1),
                    **diag,
                }
                LOG.info(json.dumps(rec))
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()

            if args.max_steps and step >= args.max_steps:
                LOG.info(f"max_steps={args.max_steps} reached")
                break

        if args.max_steps and step >= args.max_steps:
            break

        # Checkpoint at end of epoch
        if (epoch + 1) % args.ckpt_every_epoch == 0 and not args.smoke_test:
            ck_path = out_dir / "ckpts" / f"dit_epoch{epoch+1}_step{step}.pt"
            torch.save(estimator.state_dict(), ck_path)
            LOG.info(f"checkpoint saved: {ck_path}")

    log_f.close()
    LOG.info(f"training complete; step={step}, elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
