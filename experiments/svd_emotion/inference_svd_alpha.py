"""SVD-aware inference with alpha control over the emotion residual.

Replicates `CausalConditionalCFM.solve_euler` step-by-step but, at each Euler
step, decomposes the predicted velocity v via SVD and rescales the residual
component A:

    v_ctrl = alpha * A + B
    x_{s+ds} = x_s + ds * v_ctrl

where B is the top-k SVD reconstruction (semantic baseline) and A = v - B is
the residual (emotion component, per plan.md v2 convention).

alpha values:
  alpha = 0   -> emotion-free (drops residual entirely; should approach neutral)
  alpha = 1   -> identical to original CosyVoice3 generation
  alpha > 1   -> stronger emotion (extrapolation)
  alpha < 0   -> inverted emotion direction (exploratory)

Usage:
  CUDA_VISIBLE_DEVICES=5 python experiments/svd_emotion/inference_svd_alpha.py \\
    --ckpt /test1208/zw/ctm_emotion_tts/outputs/svd_v2/train/ckpts/dit_epochN.pt \\
    --manifest /test1208/zw/ctm_emotion_tts/data/manifests/esd_parallel.jsonl \\
    --out_dir /test1208/zw/ctm_emotion_tts/outputs/svd_v2/eval/alpha_ladder \\
    --emotions Happy,Sad,Angry,Surprise --n_samples 10 \\
    --alphas 0,0.25,0.5,0.75,1.0,1.25,1.5 --k_svd 16

If --ckpt is omitted, the original (untrained) CosyVoice3 weights are used —
useful for sanity-checking SVD control before any training.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path("/test1208/zw/ctm_emotion_tts")
COSYVOICE_REPO = ROOT / "repos" / "CosyVoice"
MODEL_DIR = Path("/home/saiadmin/modelscope_cache/FunAudioLLM/Fun-CosyVoice3-0___5B-2512")

sys.path.insert(0, str(COSYVOICE_REPO))
sys.path.insert(0, str(COSYVOICE_REPO / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel                                # noqa: E402
from cosyvoice.utils.mask import make_pad_mask                                # noqa: E402

from experiments.svd_emotion.svd_decompose import svd_decompose               # noqa: E402

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
LOG = logging.getLogger("inference_svd_alpha")


@torch.inference_mode()
def svd_alpha_solve_euler(decoder, x_init, mu, mask, spks, cond,
                          n_timesteps: int = 10, k_svd: int = 16,
                          alpha: float = 1.0,
                          use_cfg: bool = False,
                          collect_diag: bool = True):
    """Custom Euler loop replacing CausalConditionalCFM.solve_euler.

    Mirrors CFG / non-CFG behaviour of the original; at each step the velocity
    is SVD-decomposed (per-sample) and the residual A is scaled by `alpha`
    before the Euler update.

    Args:
      decoder:    flow.decoder (CausalConditionalCFM)
      x_init:     (B, 80, K) initial noise (B is usually 1 for inference)
      mu, mask, spks, cond: standard CFM conditions
      n_timesteps: number of Euler steps (default 10, matches inference)
      k_svd:      SVD top-k cutoff
      alpha:      scale on residual component
      use_cfg:    if True, run condition + uncondition each step (2x forward)
      collect_diag: if True, also return per-step SVD diagnostics

    Returns:
      x_final:    (B, 80, K) generated mel
      diag:       list of per-step diagnostics if collect_diag else None
    """
    device = x_init.device
    dtype = x_init.dtype
    x = x_init

    t_span = torch.linspace(0.0, 1.0, n_timesteps + 1, device=device, dtype=dtype)
    if getattr(decoder, "t_scheduler", "cosine") == "cosine":
        t_span = 1.0 - torch.cos(t_span * 0.5 * 3.141592653589793)

    diag = [] if collect_diag else None

    for step in range(1, len(t_span)):
        s_now = t_span[step - 1].unsqueeze(0)
        ds = (t_span[step] - t_span[step - 1])

        if use_cfg:
            # Stack [cond, uncond]
            x_in = torch.cat([x, x], dim=0)
            mask_in = torch.cat([mask, mask], dim=0)
            mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
            cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
            t_in = s_now.repeat(2)
            v_both = decoder.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in,
                                       streaming=False)
            v_cond, v_uncond = v_both.split(x.shape[0], dim=0)
            w = getattr(decoder, "inference_cfg_rate", 0.7)
            v_raw = (1.0 + w) * v_cond - w * v_uncond
        else:
            v_raw = decoder.estimator(x, mask, mu, s_now, spks, cond, streaming=False)

        # SVD decompose: B = top-k = semantic, A = residual = emotion
        A, B, _ = svd_decompose(v_raw, k_svd)

        # Scale residual by alpha
        v_ctrl = alpha * A + B

        x = x + ds * v_ctrl

        if collect_diag:
            diag.append({
                "step": step,
                "s": float(s_now.item()),
                "ds": float(ds.item()),
                "v_raw_norm": float(torch.linalg.vector_norm(v_raw).item()),
                "A_norm": float(torch.linalg.vector_norm(A).item()),
                "B_norm": float(torch.linalg.vector_norm(B).item()),
                "v_ctrl_norm": float(torch.linalg.vector_norm(v_ctrl).item()),
            })

    return x, diag


@torch.inference_mode()
def synthesize_with_alpha(cosyvoice, ref_wav_path: str, ref_text: str,
                          target_text: str,
                          alpha: float, k_svd: int = 16, use_cfg: bool = False,
                          n_timesteps: int = 10):
    """Run end-to-end synthesis with SVD-alpha control over the flow's velocity.

    Pipeline:
      1. Use ref_wav as zero-shot prompt to extract speaker / mel / token features
      2. Call LLM to generate speech tokens for target_text
      3. Build flow inputs (mu, spks, cond)
      4. Run custom svd_alpha_solve_euler with the requested alpha
      5. Vocoder to waveform
    """
    device = cosyvoice.model.device
    sr = cosyvoice.sample_rate
    flow = cosyvoice.model.flow
    hift = cosyvoice.model.hift

    # 1. Frontend -- CosyVoice3 requires `<|endofprompt|>` token somewhere in
    #    text or prompt_text; prepend the system prefix to the reference text.
    PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
    prompt_text = PROMPT_PREFIX + ref_text
    mi = cosyvoice.frontend.frontend_zero_shot(target_text, prompt_text, ref_wav_path, sr, "")

    # 2. LLM tokens
    llm_tokens = []
    for tok in cosyvoice.model.llm.inference(
        text=mi["text"].to(device),
        text_len=torch.tensor([mi["text"].shape[1]], dtype=torch.int32, device=device),
        prompt_text=mi["prompt_text"].to(device),
        prompt_text_len=torch.tensor([mi["prompt_text"].shape[1]], dtype=torch.int32, device=device),
        prompt_speech_token=mi["llm_prompt_speech_token"].to(device),
        prompt_speech_token_len=torch.tensor([mi["llm_prompt_speech_token"].shape[1]], dtype=torch.int32, device=device),
        embedding=mi["llm_embedding"].to(device),
        uuid="svd_alpha_infer",
    ):
        llm_tokens.append(int(tok))
    if not llm_tokens:
        raise RuntimeError("LLM produced no tokens")
    gen_token = torch.tensor(llm_tokens, dtype=torch.int32, device=device).unsqueeze(0)

    # 3. Flow encoder pipeline (matches inference path)
    prompt_token = mi["flow_prompt_speech_token"].to(device)
    prompt_feat = mi["prompt_speech_feat"].to(device)
    embedding = mi["flow_embedding"].to(device)

    embedding = F.normalize(embedding, dim=1)
    spks = flow.spk_embed_affine_layer(embedding)

    token = torch.cat([prompt_token, gen_token], dim=1)
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=device)
    mask_tok = (~make_pad_mask(token_len)).unsqueeze(-1).to(spks)
    token_emb = flow.input_embedding(torch.clamp(token, min=0)) * mask_tok
    h = flow.pre_lookahead_layer(token_emb)
    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
    mel_len1 = prompt_feat.shape[1]
    mel_total = h.shape[1]
    if mel_total <= mel_len1:
        raise RuntimeError(f"bad mel lengths total={mel_total} prompt={mel_len1}")
    cond = torch.zeros([1, mel_total, flow.output_size], device=device, dtype=h.dtype)
    cond[:, :mel_len1] = prompt_feat
    cond = cond.transpose(1, 2).contiguous()
    mu = h.transpose(1, 2).contiguous()
    mask = (~make_pad_mask(torch.tensor([mel_total], device=device))).to(h).unsqueeze(1)

    # 4. Custom Euler with SVD-alpha
    z = torch.randn([1, 80, mel_total], device=device, dtype=spks.dtype)
    mel_full, diag = svd_alpha_solve_euler(
        flow.decoder, z, mu, mask, spks, cond,
        n_timesteps=n_timesteps, k_svd=k_svd, alpha=alpha,
        use_cfg=use_cfg, collect_diag=True,
    )
    mel = mel_full[:, :, mel_len1:].float()                                 # drop prompt section

    # 5. Vocoder
    hift_in = {"speech_feat": mel}
    waveform, _ = hift.inference(speech_feat=mel, finalize=True)
    return waveform.cpu(), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="DiT state_dict to load (optional)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--emotions", default="Happy,Sad,Angry,Surprise")
    ap.add_argument("--n_samples", type=int, default=10, help="Pairs per emotion")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--k_svd", type=int, default=16)
    ap.add_argument("--use_cfg", action="store_true")
    ap.add_argument("--n_timesteps", type=int, default=10)
    ap.add_argument("--ref_emotion", default="Neutral",
                    help="Which wav to use as reference (default Neutral)")
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--filter_lang", default="")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("loading model...")
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR), fp16=False)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=cosyvoice.model.device, weights_only=True)
        cosyvoice.model.flow.decoder.estimator.load_state_dict(sd, strict=True)
        LOG.info(f"loaded ckpt {args.ckpt}")

    target_emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    alphas = [float(a) for a in args.alphas.split(",")]

    records = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if args.filter_lang and r["lang"] != args.filter_lang:
                continue
            records.append(r)
    LOG.info(f"manifest: {len(records)} records (filter_lang={args.filter_lang or 'all'})")

    selected = []
    for emo in target_emotions:
        req_ref = emo if args.ref_emotion == "target" else args.ref_emotion
        cands = [r for r in records if req_ref in r["emotions"] and emo in r["emotions"]]
        random.shuffle(cands)
        for r in cands[:args.n_samples]:
            selected.append((emo, r))
    LOG.info(f"selected {len(selected)} (emo, sample) pairs")

    manifest_out = out_dir / "alpha_ladder_manifest.jsonl"
    diag_out = out_dir / "alpha_ladder_diag.jsonl"
    mf = open(manifest_out, "w", encoding="utf-8")
    df = open(diag_out, "w", encoding="utf-8")

    for emo, rec in selected:
        # ref_emotion="target" means use the target-emo wav as reference
        # (so LLM produces emo-conditioned speech tokens).
        actual_ref = emo if args.ref_emotion == "target" else args.ref_emotion
        ref_path = rec["emotions"][actual_ref]
        text = rec["text"]
        for alpha in alphas:
            tag = f"{rec['key']}__refEmo_{actual_ref}__alpha_{alpha:+.2f}"
            try:
                wav, diag = synthesize_with_alpha(
                    cosyvoice, ref_path, ref_text=text, target_text=text,
                    alpha=alpha, k_svd=args.k_svd,
                    use_cfg=args.use_cfg, n_timesteps=args.n_timesteps,
                )
                wav_path = out_dir / f"{tag}.wav"
                torchaudio.save(str(wav_path), wav, cosyvoice.sample_rate)
                mf.write(json.dumps({
                    "tag": tag, "key": rec["key"], "speaker": rec["speaker"],
                    "lang": rec["lang"], "text": text,
                    "target_emotion": emo, "ref_emotion": args.ref_emotion,
                    "alpha": alpha, "wav": str(wav_path),
                }, ensure_ascii=False) + "\n")
                df.write(json.dumps({"tag": tag, "diag": diag}) + "\n")
                LOG.info(f"[{tag}] OK")
            except Exception as e:
                LOG.error(f"[{tag}] {type(e).__name__}: {e}")

    mf.close(); df.close()
    LOG.info(f"done. results in {out_dir}")


if __name__ == "__main__":
    main()
