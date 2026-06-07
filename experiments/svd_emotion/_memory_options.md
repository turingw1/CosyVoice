# Memory-reduction options for DiT fine-tuning

CosyVoice3 DiT estimator has 331.14M trainable params (full DiT). Memory budget:
  weights (fp32):      1.3 GB
  grads:               1.3 GB
  AdamW m, v:          2.6 GB
  activations (~22L × dim=1024 × seq~600 × bs=2): ~15-20 GB
                                                  ----
                                       total:    ~20-25 GB peak

GPU5 free is 39.6 GB (CyberVerse takes 41.5 GB). Tight but should fit at bs=1
without further tricks.

If bs=1 still OOMs, options in order of preference:

1. **bf16 autocast** (cleanest, no architecture change):
   ```python
   from torch.cuda.amp import autocast
   with autocast(dtype=torch.bfloat16):
       v_hat = flow.decoder.estimator(...)
       loss, diag = compute_svd_loss(...)
   ```
   Note: SVD in bf16 may be slightly less stable; might need to cast v_hat to fp32
   before svd_decompose. Try with cast first.

2. **Gradient checkpointing on DiT blocks** (slower but big memory savings):
   ```python
   for blk in flow.decoder.estimator.transformer_blocks:
       blk.gradient_checkpointing = True  # if supported; else wrap with
                                          # torch.utils.checkpoint.checkpoint
   ```

3. **LoRA-only fine-tuning** (smallest memory, but limited capacity):
   Use peft library; train rank-16 adapters on attention QKV/out projections only.
   ~5M trainable params instead of 331M.

4. **Reduce sequence length further**: `max_feat_len 400` drops more samples but
   keeps memory bounded.

5. **Disable optimizer state-doubling**: switch AdamW -> SGD or Adafactor (Adafactor
   uses ~1/3 the memory of Adam).
