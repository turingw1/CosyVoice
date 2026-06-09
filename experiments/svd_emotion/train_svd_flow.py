"""Train CosyVoice3 DiT with v3 framework: contrastive learning on emotion residual.

v3 design (vs v2):
  - A2 architecture: SVD residual is used ONLY as initialization for an
    EmotionResidualHead. Then A = SVD_residual_init (detached) + head(v).
    SVD is no longer recomputed in the loss target — it just gives a
    starting point.
  - L_emo: Standard InfoNCE contrastive (explicit positive + negative)
        pos = u_true_emo  (ground truth CFM velocity for the emo wav)
        neg = u_true_neu  (ground truth CFM velocity for the parallel neu wav)
        Temperature T controls how sharply the softmax separates pos vs neg.
            T -> 0   : sharp; gradients amplified, training harder
            T -> inf : smooth; loss approaches log(2), no learning signal
  - L_ortho: cross-channel Gram matrix of A and B, |A @ B^T|_F^2.
        Prevents A from drifting into v's column space (i.e. A = v cheating).
  - L_sem: REMOVED. B = v - A is not directly supervised.

Total loss:  L = lambda_fm * L_fm + lambda_emo * L_emo + lambda_ortho * L_ortho
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
import torch.nn as nn
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
# v3 EmotionResidualHead (A2 architecture)
# ---------------------------------------------------------------------------


class EmotionResidualHead(nn.Module):
    """Learnable refinement of the SVD residual.

    Forward:  A = A_svd_init.detach() + delta(v)

    The final conv is initialized to zero, so at training step 0:
      delta(v) == 0  =>  A == A_svd_init  (pure SVD residual)

    As training progresses, the head learns to bend A away from the pure SVD
    residual toward the emotion direction (driven by InfoNCE L_emo).
    """

    def __init__(self, mel_dim: int = 80, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(mel_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, mel_dim, kernel_size=3, padding=1),
        )
        # Zero-init the last conv so initial delta = 0 (A starts as SVD residual)
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B, 80, K)  ->  delta: (B, 80, K)"""
        return self.proj(v)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def freeze_all_but_dit(cosyvoice) -> int:
    """Freeze every parameter except the DiT estimator."""
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
    LOG.info(f"freeze: trainable_dit={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M")
    return n_train


def enable_grad_checkpointing(estimator) -> int:
    if not hasattr(estimator, "transformer_blocks"):
        LOG.warning("estimator has no .transformer_blocks; skipping grad-ckpt")
        return 0
    blocks = estimator.transformer_blocks
    for blk in blocks:
        orig_forward = blk.forward
        @functools.wraps(orig_forward)
        def _ckpt_forward(*args, _orig=orig_forward, **kwargs):
            def run(*a):
                return _orig(*a, **kwargs)
            return ckpt_fn(run, *args, use_reentrant=False)
        blk.forward = _ckpt_forward
    LOG.info(f"grad checkpointing enabled on {len(blocks)} DiT blocks")
    return len(blocks)


def build_flow_inputs(flow, batch: dict, device: torch.device,
                      side: str = "emo") -> dict:
    """L1 encoder pipeline: tokens/feat -> (mu, spks, cond, x_1, mask, K_eff)."""
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
                t_min: float = 0.05, t_max: float = 0.95,
                t_explicit: torch.Tensor | None = None):
    """Sample (y, t, z, u_true) for CFM. If t_explicit given, use it (shared with neu side)."""
    B = x_1.shape[0]
    if t_explicit is None:
        t = torch.rand(B, device=x_1.device, dtype=x_1.dtype) * (t_max - t_min) + t_min
    else:
        t = t_explicit
    z = torch.randn_like(x_1)
    t_b = t.view(B, 1, 1)
    y = (1.0 - (1.0 - sigma_min) * t_b) * z + t_b * x_1
    u_true = x_1 - (1.0 - sigma_min) * z
    return y, t, z, u_true


def masked_mean_time(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num = (x * mask).sum(dim=-1)
    den = mask.sum(dim=-1).clamp(min=1e-6)
    return num / den


def compute_svd_loss_v3(v_hat: torch.Tensor, u_true_emo: torch.Tensor,
                       u_true_neu_c: torch.Tensor,
                       emotion_head: EmotionResidualHead,
                       emo_mask: torch.Tensor, common_mask: torch.Tensor,
                       k_svd: int, T: float = 0.1,
                       lambdas=(1.0, 0.5, 0.01)):
    """v3 three-term loss: L_fm + lambda_emo * L_emo (InfoNCE) + lambda_ortho * L_ortho.

    Args:
      v_hat:        (B, 80, K_emo) emo DiT output, with grad
      u_true_emo:   (B, 80, K_emo) ground truth CFM velocity for emo wav
      u_true_neu_c: (B, 80, K_c)   ground truth CFM velocity for neu wav, CROPPED to K_c
      emotion_head: EmotionResidualHead (learnable A refiner)
      emo_mask:     (B, 1, K_emo)
      common_mask:  (B, 1, K_c)    K_c = min(K_emo, K_neu)
      k_svd:        int, SVD top-k for init
      T:            float, InfoNCE temperature
      lambdas:      (lambda_fm, lambda_emo, lambda_ortho)
    """
    v = v_hat.float()
    Bsz = v.shape[0]

    # --- A2 architecture: SVD provides init, head learns delta ---
    with torch.no_grad():
        # SVD residual as initialization (no gradient through SVD)
        A_init, B_init, _ = svd_decompose(v, k_svd)
        # A_init is the residual; B_init is the top-k reconstruction.

    delta = emotion_head(v)               # (B, 80, K), starts at 0
    A = A_init + delta                    # learnable A
    B_top = v - A                         # semantic part = whatever's left

    # --- L_fm: anchor v to ground truth emo velocity ---
    sq = (v - u_true_emo.float()).pow(2) * emo_mask
    denom = emo_mask.sum() * v.shape[1] + 1e-6
    L_fm = sq.sum() / denom

    # --- L_emo: InfoNCE on common-crop region ---
    K_c = common_mask.shape[-1]
    A_c = A[..., :K_c]
    u_emo_c = u_true_emo[..., :K_c].float()
    u_neu_c = u_true_neu_c.float()
    mask_full = common_mask.expand_as(A_c)

    A_flat = (A_c * mask_full).flatten(start_dim=1)        # (B, 80*K_c)
    u_emo_flat = (u_emo_c * mask_full).flatten(start_dim=1)
    u_neu_flat = (u_neu_c * mask_full).flatten(start_dim=1)

    sim_pos = F.cosine_similarity(A_flat, u_emo_flat, dim=-1)   # (B,)
    sim_neg = F.cosine_similarity(A_flat, u_neu_flat, dim=-1)   # (B,)

    # InfoNCE with 1 positive + 1 negative per sample
    logits = torch.stack([sim_pos / T, sim_neg / T], dim=-1)    # (B, 2)
    labels = torch.zeros(Bsz, dtype=torch.long, device=v.device)
    L_emo = F.cross_entropy(logits, labels)

    # --- L_ortho: cross-channel Gram of A and B ---
    # A: (B, 80, K), B_top: (B, 80, K)
    A_mat = A.reshape(Bsz, 80, -1)
    B_mat = B_top.reshape(Bsz, 80, -1)
    prod = torch.matmul(A_mat, B_mat.transpose(1, 2))    # (B, 80, 80)
    L_ortho = (prod.pow(2)).mean()

    total = lambdas[0] * L_fm + lambdas[1] * L_emo + lambdas[2] * L_ortho

    # Diagnostics
    with torch.no_grad():
        v_norm = torch.linalg.vector_norm(v).clamp(min=1e-6)
        cos_B_v_neu_proxy = float("nan")  # filled by caller if needed
    diag = {
        "L_fm": float(L_fm.detach()),
        "L_emo": float(L_emo.detach()),
        "L_ortho": float(L_ortho.detach()),
        "sim_pos": float(sim_pos.mean().detach()),
        "sim_neg": float(sim_neg.mean().detach()),
        "sim_gap": float((sim_pos - sim_neg).mean().detach()),
        "A_over_v_ratio": float(torch.linalg.vector_norm(A).detach() / v_norm),
        "B_over_v_ratio": float(torch.linalg.vector_norm(B_top).detach() / v_norm),
        "delta_norm_ratio": float(torch.linalg.vector_norm(delta).detach() /
                                  torch.linalg.vector_norm(A_init).clamp(min=1e-6).detach()),
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
    ap.add_argument("--lr_head", type=float, default=1e-4,
                    help="Higher lr for emotion_head (small fresh net)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--k_svd", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="InfoNCE temperature: smaller=sharper, harder")
    ap.add_argument("--lambda_fm", type=float, default=1.0)
    ap.add_argument("--lambda_emo", type=float, default=0.5)
    ap.add_argument("--lambda_ortho", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every_epoch", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--amp_dtype", default="fp32",
                    choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--max_feat_len", type=int, default=800)
    ap.add_argument("--load_ckpt", default="",
                    help="Optional path to DiT ckpt to resume from (v2 ckpt for continuation experiments)")
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

    # Optionally load a v2 ckpt
    if args.load_ckpt:
        sd = torch.load(args.load_ckpt, map_location=device, weights_only=True)
        estimator.load_state_dict(sd, strict=True)
        LOG.info(f"resumed DiT from {args.load_ckpt}")

    # v3 EmotionResidualHead (A2 architecture)
    emotion_head = EmotionResidualHead(mel_dim=80, hidden=128).to(device)
    emotion_head.train()
    n_head = sum(p.numel() for p in emotion_head.parameters())
    LOG.info(f"emotion_head params: {n_head/1e3:.1f}K")

    # Two-group optimizer: DiT (small lr) + head (larger lr because fresh)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in estimator.parameters() if p.requires_grad], "lr": args.lr},
        {"params": emotion_head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.0)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = dtype_map[args.amp_dtype]
    use_amp = (args.amp_dtype != "fp32")
    LOG.info(f"amp_dtype={args.amp_dtype}, T={args.temperature}, "
             f"lambdas=({args.lambda_fm}, {args.lambda_emo}, {args.lambda_ortho})")

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

            # Build emo + neu inputs
            with torch.no_grad():
                inp_emo = build_flow_inputs(flow, batch, device, side="emo")
                inp_neu = build_flow_inputs(flow, batch, device, side="neu")
            x_1 = inp_emo["x_1"]
            mu = inp_emo["mu"]; spks = inp_emo["spks"]
            cond = inp_emo["cond"]; mask = inp_emo["mask"]

            # CFM perturb on emo with shared t
            y_emo, t_samp, z_emo, u_true_emo = cfm_perturb(x_1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                v_hat = flow.decoder.estimator(
                    y_emo, mask, mu, t_samp, spks, cond, streaming=False
                )

            # Compute ground truth u_true_neu (no DiT forward needed, just CFM)
            with torch.no_grad():
                x_1_neu = inp_neu["x_1"]
                z_neu = torch.randn_like(x_1_neu)
                # share the same t with emo side for fair velocity-space comparison
                t_neu = t_samp
                t_b = t_neu.view(-1, 1, 1)
                sigma_min = 1e-6
                u_true_neu_full = x_1_neu - (1.0 - sigma_min) * z_neu
                # crop to K_c
                K_c = min(x_1.shape[-1], x_1_neu.shape[-1])
                u_true_neu_c = u_true_neu_full[..., :K_c]

            # Common mask
            emo_len = inp_emo["K_eff"]; neu_len = inp_neu["K_eff"]
            common_len = torch.minimum(emo_len, neu_len).clamp(max=K_c)
            common_mask = (~make_pad_mask(common_len, max_len=K_c)).to(spks).unsqueeze(1)

            loss, diag = compute_svd_loss_v3(
                v_hat=v_hat,
                u_true_emo=u_true_emo,
                u_true_neu_c=u_true_neu_c,
                emotion_head=emotion_head,
                emo_mask=mask,
                common_mask=common_mask,
                k_svd=args.k_svd,
                T=args.temperature,
                lambdas=(args.lambda_fm, args.lambda_emo, args.lambda_ortho),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                list(estimator.parameters()) + list(emotion_head.parameters()),
                args.grad_clip
            )
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                rec = {
                    "step": step, "epoch": epoch,
                    "loss": float(loss.detach()),
                    "grad_norm": float(gn.detach()),
                    "lr": args.lr, "lr_head": args.lr_head,
                    "elapsed_s": round(time.time() - t0, 1),
                    "K_emo": int(x_1.shape[-1]),
                    "K_neu": int(x_1_neu.shape[-1]),
                    "K_c": int(K_c),
                    "T": args.temperature,
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
            head_path = out_dir / "ckpts" / f"head_epoch{epoch+1}_step{step}.pt"
            torch.save(emotion_head.state_dict(), head_path)
            LOG.info(f"checkpoint saved: {ck_path}, {head_path}")

    log_f.close()
    LOG.info(f"training complete; step={step}, elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
