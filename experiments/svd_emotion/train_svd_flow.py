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


def build_flow_inputs(flow, batch: dict, device: torch.device,
                      side: str = "emo") -> dict:
    """L1 encoder pipeline: tokens/feat -> (mu, spks, cond, x_1, mask, K_eff).

    Args:
      side: "emo" or "neu" -- selects which side of the parallel pair to use.
    """
    suffix = side
    token = batch[f"speech_token_{suffix}"].to(device).long()
    token_len = batch[f"speech_token_{suffix}_len"].to(device)
    feat = batch[f"speech_feat_{suffix}"].to(device)
    feat_len = batch[f"speech_feat_{suffix}_len"].to(device)
    embedding = batch[f"embedding_{suffix}"].to(device)

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
                     v_hat_neu: torch.Tensor,
                     emo_feat: torch.Tensor, neu_feat: torch.Tensor,
                     emo_mask: torch.Tensor, neu_mask: torch.Tensor,
                     common_mask: torch.Tensor,
                     k_svd: int, lambdas=(1.0, 0.1, 0.1)):
    """Three-term loss with VELOCITY-space emotion target (Step-0 setup).

    L_fm:  masked MSE(v_hat, u_true)                                -- TTS anchor
    L_sem: per-channel time-avg MSE(B_top, neu_feat (time-avg))     -- soft pull
    L_emo: 1 - cos(vec(A_crop_masked), vec(v_hat - v_hat_neu)_crop_masked)
           target_dir = v_hat[:,:,:K_c] - v_hat_neu[:,:,:K_c] in velocity space.
           This matches the Step 0 diagnostic where cos(A, r) measured 0.29
           naturally pre-training. (The earlier mel-diff target was hurt by
           linear interpolation destroying phone-level correspondence.)

    Args:
      v_hat:      (B, 80, K_emo)  emo forward, with grad
      u_true:     (B, 80, K_emo)  true CFM velocity for emo
      v_hat_neu:  (B, 80, K_neu)  neu forward, no_grad
      emo_feat:   (B, 80, K_emo)
      neu_feat:   (B, 80, K_neu)
      emo_mask:   (B, 1, K_emo)
      neu_mask:   (B, 1, K_neu)
      common_mask:(B, 1, K_common) where K_common = min(K_emo, K_neu)
      k_svd:      int
      lambdas:    (lambda_fm, lambda_sem, lambda_emo)
    """
    v32 = v_hat.float()
    v32_neu = v_hat_neu.float()
    A, B_top, _ = svd_decompose(v32, k_svd)
    emo_feat_f = emo_feat.float()
    neu_feat_f = neu_feat.float()

    # --- L_fm ---
    sq = (v32 - u_true.float()).pow(2) * emo_mask
    denom = emo_mask.sum() * v32.shape[1] + 1e-6
    L_fm = sq.sum() / denom

    # --- L_sem: time-avg per channel ---
    B_mean = masked_mean_time(B_top, emo_mask)
    neu_mean = masked_mean_time(neu_feat_f, neu_mask)
    L_sem = F.mse_loss(B_mean, neu_mean)

    # --- L_emo: velocity-space cosine on common crop ---
    K_c = common_mask.shape[-1]
    v_emo_c = v32[..., :K_c]
    v_neu_c = v32_neu[..., :K_c]
    target_dir = v_emo_c - v_neu_c                  # (B, 80, K_c)
    A_c = A[..., :K_c]
    mask_full = common_mask.expand_as(A_c)
    A_flat = (A_c * mask_full).flatten(start_dim=1)
    tgt_flat = (target_dir * mask_full).flatten(start_dim=1)
    cos_A_target = F.cosine_similarity(A_flat, tgt_flat, dim=-1)
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
        "K_common": int(K_c),
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
                inp_emo = build_flow_inputs(flow, batch, device, side="emo")
                inp_neu = build_flow_inputs(flow, batch, device, side="neu")
            x_1 = inp_emo["x_1"]
            mu = inp_emo["mu"]; spks = inp_emo["spks"]
            cond = inp_emo["cond"]; mask = inp_emo["mask"]

            y, t_samp, z, u_true = cfm_perturb(x_1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                v_hat = flow.decoder.estimator(
                    y, mask, mu, t_samp, spks, cond, streaming=False
                )

            # ---- Second forward on neu side (no_grad to save memory) ----
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                y_neu, _, _, _ = cfm_perturb(inp_neu["x_1"])
                v_hat_neu = flow.decoder.estimator(
                    y_neu, inp_neu["mask"], inp_neu["mu"], t_samp,
                    inp_neu["spks"], inp_neu["cond"], streaming=False
                )

            neu_feat = inp_neu["x_1"]       # (B, 80, K_neu)
            neu_mask = inp_neu["mask"]
            emo_feat = x_1; emo_mask = mask

            # Build common mask: min length across both sides per sample
            K_emo_t = x_1.shape[-1]; K_neu_t = inp_neu["x_1"].shape[-1]
            K_c = min(K_emo_t, K_neu_t)
            emo_len = inp_emo["K_eff"]; neu_len = inp_neu["K_eff"]
            common_len = torch.minimum(emo_len, neu_len).clamp(max=K_c)
            common_mask = (~make_pad_mask(common_len, max_len=K_c)).to(spks).unsqueeze(1)

            loss, diag = compute_svd_loss(
                v_hat=v_hat, u_true=u_true, v_hat_neu=v_hat_neu,
                emo_feat=emo_feat, neu_feat=neu_feat,
                emo_mask=emo_mask, neu_mask=neu_mask,
                common_mask=common_mask,
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
