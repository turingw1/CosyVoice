"""Train CosyVoice3 DiT to produce velocities that SVD-decompose meaningfully.

Loss (see plan.md v2):
  L = lambda_fm  * L_fm                        # CFM anchor — keeps TTS alive
    + lambda_sem * L_sem                       # time-avg of B ≈ time-avg of neu mel
    + lambda_emo * L_emo                       # A direction ≈ (emo - neu) direction

SVD convention (do NOT flip): B = top-k (semantic), A = residual (emotion).

Memory notes (GPU5 has 39.6 GB free):
  - 331.14M trainable params (full DiT) needs ~6 GB for weights+grads+Adam states
  - 22 transformer blocks × dim=1024 × seq~600 activations in fp32 OOM even at bs=1
  - Default fix: bf16 autocast around forward + loss. SVD stays in fp32 (cast back).
  - Optional --grad_ckpt wraps each DiT block in torch.utils.checkpoint.checkpoint.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn
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
    datefmt="%H:%M:%S", level=logging.INFO,
)
LOG = logging.getLogger("train_svd_flow")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def freeze_all_but_dit(cosyvoice) -> int:
    """Freeze every parameter except the DiT estimator.

    CosyVoice3Model is NOT an nn.Module — it holds .llm / .flow / .hift
    sub-modules. Iterate over them individually.
    """
    model = cosyvoice.model
    for sub_name in ("llm", "flow", "hift"):
        sub = getattr(model, sub_name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad_(False)
    estimator = model.flow.decoder.estimator
    for p in estimator.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    n_total = sum(p.numel() for sn in ("llm", "flow", "hift")
                  for p in getattr(model, sn).parameters())
    LOG.info(f"freeze: trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M")
    return n_train


def enable_grad_checkpointing(estimator) -> int:
    """Wrap each transformer block forward in torch.utils.checkpoint.

    Returns the number of blocks wrapped. The wrapping is functionally
    transparent: forward returns the same tensor, but activations are not
    saved (they are recomputed during backward, trading compute for memory).
    """
    if not hasattr(estimator, "transformer_blocks"):
        LOG.warning("estimator has no .transformer_blocks; skipping grad-ckpt")
        return 0
    blocks = estimator.transformer_blocks
    for blk in blocks:
        orig_forward = blk.forward
        @functools.wraps(orig_forward)
        def _ckpt_forward(*args, _orig=orig_forward, **kwargs):
            # use_reentrant=False is the recommended new API
            def run(*a):
                return _orig(*a, **kwargs)
            return ckpt_fn(run, *args, use_reentrant=False)
        blk.forward = _ckpt_forward
    LOG.info(f"grad checkpointing enabled on {len(blocks)} DiT blocks")
    return len(blocks)


def build_flow_inputs(flow, batch: dict, device: torch.device) -> dict:
    """L1 encoder pipeline: tokens/feat -> (mu, spks, cond, x_1, mask)."""
    token = batch["speech_token_emo"].to(device).long()
    token_len = batch["speech_token_emo_len"].to(device)
    feat = batch["speech_feat_emo"].to(device)
    feat_len = batch["speech_feat_emo_len"].to(device)
    embedding = batch["embedding_emo"].to(device)

    embedding = F.normalize(embedding, dim=1)
    spks = flow.spk_embed_affine_layer(embedding)

    tok_pad_mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(spks)
    token_emb = flow.input_embedding(torch.clamp(token, min=0)) * tok_pad_mask
    h = flow.pre_lookahead_layer(token_emb)
    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)

    K = min(h.shape[1], feat.shape[1])
    h = h[:, :K]
    x_1 = feat[:, :K].transpose(1, 2).contiguous()
    mu = h.transpose(1, 2).contiguous()
    cond = torch.zeros_like(x_1)
    K_eff = torch.minimum(token_len * flow.token_mel_ratio, feat_len).clamp(max=K)
    mask = (~make_pad_mask(K_eff, max_len=K)).to(spks).unsqueeze(1)
    return {"x_1": x_1, "mu": mu, "spks": spks, "cond": cond, "mask": mask, "K_eff": K_eff}


def cfm_perturb(x_1: torch.Tensor, sigma_min: float = 1e-6,
                t_min: float = 0.05, t_max: float = 0.95):
    B = x_1.shape[0]
    t = torch.rand(B, device=x_1.device, dtype=x_1.dtype) * (t_max - t_min) + t_min
    z = torch.randn_like(x_1)
    t_b = t.view(B, 1, 1)
    y = (1.0 - (1.0 - sigma_min) * t_b) * z + t_b * x_1
    u_true = x_1 - (1.0 - sigma_min) * z
    return y, t, z, u_true


def masked_mean_time(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num = (x * mask).sum(dim=-1)
    den = mask.sum(dim=-1).clamp(min=1e-6)
    return num / den


def compute_svd_loss(v_hat: torch.Tensor, u_true: torch.Tensor,
                     emo_feat: torch.Tensor, neu_feat: torch.Tensor,
                     emo_mask: torch.Tensor, neu_mask: torch.Tensor,
                     k_svd: int, lambdas=(1.0, 0.1, 0.1)):
    """Compute the 3-term loss. SVD is done in fp32 for numerical stability."""
    # Promote to fp32 around SVD (bf16 SVD can blow up).
    v32 = v_hat.float()
    A, B_top, _ = svd_decompose(v32, k_svd)

    sq = (v32 - u_true.float()).pow(2) * emo_mask
    denom = emo_mask.sum() * v32.shape[1] + 1e-6
    L_fm = sq.sum() / denom

    B_mean = masked_mean_time(B_top, emo_mask)
    A_mean = masked_mean_time(A, emo_mask)
    emo_mel_mean = masked_mean_time(emo_feat.float(), emo_mask)
    neu_mel_mean = masked_mean_time(neu_feat.float(), neu_mask)

    L_sem = F.mse_loss(B_mean, neu_mel_mean)

    target_dir = emo_mel_mean - neu_mel_mean
    cos_A_target = F.cosine_similarity(A_mean, target_dir, dim=-1)
    L_emo = (1.0 - cos_A_target).mean()

    total = lambdas[0] * L_fm + lambdas[1] * L_sem + lambdas[2] * L_emo

    diag = {
        "L_fm": float(L_fm.detach()),
        "L_sem": float(L_sem.detach()),
        "L_emo": float(L_emo.detach()),
        "cos_A_target_mean": float(cos_A_target.mean().detach()),
        "A_over_v_ratio": float(
            (torch.linalg.vector_norm(A) /
             torch.linalg.vector_norm(v32).clamp(min=1e-6)).detach()),
        "B_over_v_ratio": float(
            (torch.linalg.vector_norm(B_top) /
             torch.linalg.vector_norm(v32).clamp(min=1e-6)).detach()),
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
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--k_svd", type=int, default=16)
    ap.add_argument("--lambda_fm", type=float, default=1.0)
    ap.add_argument("--lambda_sem", type=float, default=0.1)
    ap.add_argument("--lambda_emo", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every_epoch", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--amp_dtype", default="bf16",
                    choices=["fp32", "bf16", "fp16"],
                    help="Autocast dtype for forward (default bf16). "
                         "fp32 OOMs DiT even at bs=1 on a 39GB-free H100.")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="Wrap each DiT block in torch.utils.checkpoint "
                         "(slower, big activation savings).")
    ap.add_argument("--max_feat_len", type=int, default=800,
                    help="Drop samples whose mel exceeds this many frames "
                         "(default 800 = ~16s, keeps memory bounded).")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    log_path = out_dir / "train.log.jsonl"

    LOG.info(f"loading CosyVoice3 from {MODEL_DIR}")
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR), fp16=False)
    flow = cosyvoice.model.flow
    device = cosyvoice.model.device
    LOG.info(f"device={device}")

    flow.decoder.training_cfg_rate = 0.0
    LOG.info("training_cfg_rate forced to 0")

    n_train = freeze_all_but_dit(cosyvoice)

    estimator = flow.decoder.estimator
    estimator.train()
    if args.grad_ckpt:
        enable_grad_checkpointing(estimator)

    optimizer = torch.optim.AdamW(
        [p for p in estimator.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = dtype_map[args.amp_dtype]
    use_amp = (args.amp_dtype != "fp32")
    LOG.info(f"amp_dtype={args.amp_dtype}, use_amp={use_amp}, grad_ckpt={args.grad_ckpt}")

    LOG.info(f"loading dataset from {args.manifest}")
    full_ds = ESDParallelDataset(
        manifest=args.manifest, feature_dir=args.feature_dir,
        split="train", max_feat_len=args.max_feat_len, min_feat_len=40,
    )
    if args.smoke_test:
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

            with torch.no_grad():
                inp = build_flow_inputs(flow, batch, device)
            x_1 = inp["x_1"]
            mu = inp["mu"]; spks = inp["spks"]; cond = inp["cond"]; mask = inp["mask"]

            y, t_samp, z, u_true = cfm_perturb(x_1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                v_hat = flow.decoder.estimator(
                    y, mask, mu, t_samp, spks, cond, streaming=False
                )

            # Build neu-side feat tensor + mask
            neu_feat = batch["speech_feat_neu"].to(device).transpose(1, 2)
            neu_len = batch["speech_feat_neu_len"].to(device)
            neu_mask = (~make_pad_mask(neu_len, max_len=neu_feat.shape[-1])).to(spks).unsqueeze(1)
            emo_feat = x_1; emo_mask = mask

            # Loss is computed in fp32 (svd_decompose casts internally)
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
                    "K_emo": int(x_1.shape[-1]),
                    "K_neu": int(neu_feat.shape[-1]),
                    **diag,
                }
                LOG.info(json.dumps(rec))
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()

            if args.max_steps and step >= args.max_steps:
                LOG.info(f"max_steps={args.max_steps} reached")
                break

        if args.max_steps and step >= args.max_steps:
            break

        if (epoch + 1) % args.ckpt_every_epoch == 0 and not args.smoke_test:
            ck_path = out_dir / "ckpts" / f"dit_epoch{epoch+1}_step{step}.pt"
            torch.save(estimator.state_dict(), ck_path)
            LOG.info(f"checkpoint saved: {ck_path}")

    log_f.close()
    LOG.info(f"training complete; step={step}, elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
