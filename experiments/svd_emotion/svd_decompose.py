"""SVD decomposition utilities for flow velocity tensors.

v2 convention (see plan.md section 3):
- Input: v_hat of shape (B, F, K) where F=80 mel channels, K=mel frames.
- B = top-k SVD reconstruction  ->  semantic baseline (high energy)
- A = v_hat - B                  ->  emotion residual    (low energy)

This direction is forced by physical energy distribution:
- ||u_neu||_F  >>  ||u_emo - u_neu||_F  (emotion is a small perturbation)
- SVD top-k captures high-energy directions, so they correspond to u_neu structure.

Do NOT use top-k as emotion. Doing so would force training into a permanent
loss-tradeoff between L_fm and L_emo (see plan.md proof).
"""
from __future__ import annotations

from typing import Tuple

import torch


def svd_decompose(v_hat: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample SVD decomposition with top-k semantic / residual split.

    Args:
        v_hat: (..., F, K) velocity tensor; supports any leading batch dims.
        k:     number of top singular components to keep as semantic.

    Returns:
        A:     (..., F, K) emotion residual = v_hat - B
        B:     (..., F, K) semantic baseline = top-k reconstruction
        S:     (..., min(F,K)) all singular values (for diagnostics)
    """
    if v_hat.dim() < 2:
        raise ValueError(f"v_hat must have at least 2 dims (F, K), got {v_hat.shape}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    # Batched SVD on last two dims (PyTorch supports this natively)
    U, S, Vh = torch.linalg.svd(v_hat, full_matrices=False)
    # U:  (..., F, r)
    # S:  (..., r)        r = min(F, K)
    # Vh: (..., r, K)
    r = S.shape[-1]
    k_eff = min(k, r)

    Uk = U[..., :, :k_eff]                   # (..., F, k)
    Sk = S[..., :k_eff]                      # (..., k)
    Vk = Vh[..., :k_eff, :]                  # (..., k, K)

    # Reconstruct: (U_k * S_k) @ V_k
    B = torch.matmul(Uk * Sk.unsqueeze(-2), Vk)  # (..., F, K)
    A = v_hat - B
    return A, B, S


def reconstruct_x1_cfm(y: torch.Tensor, t: torch.Tensor | float, v: torch.Tensor) -> torch.Tensor:
    """Rectified-flow estimate of x_1 from (y, t, v).

    Formula:  x_1_pred = y + (1 - t) * v

    Args:
        y: (B, F, K) noisy state at time t
        t: scalar, (B,), or (B, 1, 1). If scalar/(B,), reshapes to (B,1,1).
        v: (B, F, K) velocity prediction
    """
    if isinstance(t, (int, float)):
        t_b = torch.tensor(float(t), device=y.device, dtype=y.dtype)
    else:
        t_b = t
    if t_b.dim() == 0:
        coeff = 1.0 - t_b
    elif t_b.dim() == 1:
        coeff = (1.0 - t_b).view(-1, 1, 1)
    else:
        coeff = 1.0 - t_b
    return y + coeff * v


def cfm_forward_perturb(x_1: torch.Tensor, sigma_min: float = 1e-6,
                        t: torch.Tensor | None = None,
                        z: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the CFM training inputs (y, t, z, u_target) given the target x_1.

    y     = (1 - (1 - sigma_min) * t) * z + t * x_1
    u_tgt = x_1 - (1 - sigma_min) * z

    Args:
        x_1: (B, F, K) target mel
        sigma_min: small CFM hyperparameter (default 1e-6)
        t: optional (B,) sample times in [0, 1]. If None, uniform sampled.
        z: optional (B, F, K) Gaussian noise. If None, sampled.

    Returns:
        y: (B, F, K) noisy state
        t: (B,) sampling times
        z: (B, F, K) noise
        u_tgt: (B, F, K) target velocity
    """
    B = x_1.shape[0]
    device, dtype = x_1.device, x_1.dtype
    if t is None:
        t = torch.rand(B, device=device, dtype=dtype)
    if z is None:
        z = torch.randn_like(x_1)
    t_b = t.view(B, 1, 1)
    y = (1.0 - (1.0 - sigma_min) * t_b) * z + t_b * x_1
    u_tgt = x_1 - (1.0 - sigma_min) * z
    return y, t, z, u_tgt


# ---------------------------------------------------------------------------
# Diagnostic helpers used by Step 0 and training-time monitors.
# ---------------------------------------------------------------------------


def energy_ratio(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-8) -> float:
    """||numer||_F / ||denom||_F (Frobenius norms over all dims).

    Returns a Python float for logging.
    """
    n = torch.linalg.vector_norm(numer.float()).item()
    d = max(torch.linalg.vector_norm(denom.float()).item(), eps)
    return n / d


def cosine_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Cosine similarity of two tensors, flattened."""
    af = a.float().flatten()
    bf = b.float().flatten()
    num = (af * bf).sum().item()
    denom = max((torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item()), eps)
    return num / denom


def singular_spectrum_summary(v: torch.Tensor) -> dict:
    """Returns summary stats of singular values of v (last two dims).

    Useful to diagnose how low-rank the velocity field is.
    """
    _, S, _ = torch.linalg.svd(v.float(), full_matrices=False)
    S_np = S.detach().cpu().numpy().tolist()
    total = sum(s * s for s in S_np) or 1e-12
    cum = 0.0
    energy_at = {}
    for ki in (1, 2, 4, 8, 16, 32):
        if ki <= len(S_np):
            cum_ki = sum(s * s for s in S_np[:ki])
            energy_at[f"top_{ki}_energy_frac"] = cum_ki / total
    return {
        "num_singular_values": len(S_np),
        "sigma_1": S_np[0] if S_np else 0.0,
        "sigma_max_to_min_ratio": (S_np[0] / max(S_np[-1], 1e-12)) if S_np else 0.0,
        **energy_at,
    }
