# H100 Port & 6 Competition Runs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port all proven MLX innovations to `train_gpt.py` (PyTorch H100 script), then run 6 differentiated competition runs targeting ≤1.12 BPB on the leaderboard.

**Architecture:** Single `train_gpt.py` with env-var toggles for all features. Each run is a shell script setting different env vars. Common foundation (warmdown WD, freq skip gating, LeakyReLU(0.5)², etc.) shared by all runs. Per-run features (deep supervision, calibrated quant, progressive growing) gated by env vars.

**Tech Stack:** PyTorch, DDP on 8xH100, zstd compression, int8 quantization

---

## Phase 1: Port Common Foundation

All changes in `train_gpt.py`. These are proven innovations from our MLX experiments.

### Task 1: Add new env vars to Hyperparameters class

**Files:**
- Modify: `train_gpt.py:39-94` (Hyperparameters class)

- [ ] **Step 1: Add all new hyperparameters**

After line 87 (`grad_clip_norm`), before TTT section (line 89), add:

```python
    # --- Our proven innovations (from MLX experiments) ---
    # Muon weight decay with warmdown-aware scheduling: wd = base_wd * (2 - lr_mul)
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    # Per-layer LR scaling: deeper layers get higher LR. lr_i = base * (1 + scale * i/(L-1))
    layer_lr_scale = float(os.environ.get("LAYER_LR_SCALE", 0.0))
    # Asymmetric MLP: "encoder_mult,decoder_mult" e.g. "2,4". Overrides mlp_mult.
    mlp_mult_asymmetric = os.environ.get("MLP_MULT_ASYMMETRIC", "")
    # Frequency-decomposed skip gating: 2-band lo/hi on skip connections
    freq_skip_gating = bool(int(os.environ.get("FREQ_SKIP_GATING", "0")))
    freq_skip_window = int(os.environ.get("FREQ_SKIP_WINDOW", 32))
    # Pre-warmdown QAT: quantization-aware training as regularizer
    qat_prewarmdown = bool(int(os.environ.get("QAT_PREWARMDOWN", "0")))
    qat_strength = float(os.environ.get("QAT_STRENGTH", 0.1))
    qat_stop_lr_mul = float(os.environ.get("QAT_STOP_LR_MUL", 0.8))
    qat_every = int(os.environ.get("QAT_EVERY", 10))
    # EMA shadow model
    ema_decay = float(os.environ.get("EMA_DECAY", 0.0))  # 0 = disabled, 0.997 = standard
    # FP16 embedding passthrough during quantization
    int8_keep_float_fp16_name_patterns = os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS", "")
    # Deep supervision: auxiliary prediction losses at intermediate layers
    deep_supervision = bool(int(os.environ.get("DEEP_SUPERVISION", "0")))
    deep_supervision_alpha = float(os.environ.get("DEEP_SUPERVISION_ALPHA", 0.1))
    deep_supervision_layers = os.environ.get("DEEP_SUPERVISION_LAYERS", "")
    # Calibrated quantization: per-row MSE-optimal clipping
    calibrated_quant = bool(int(os.environ.get("CALIBRATED_QUANT", "0")))
    # Progressive layer growing: start shallow, grow to full depth
    grow_layers_from = int(os.environ.get("GROW_LAYERS_FROM", 0))  # 0 = disabled
    grow_at_wallclock_frac = float(os.environ.get("GROW_AT_WALLCLOCK_FRAC", 0.35))
    # Compression: zstd instead of zlib
    use_zstd = bool(int(os.environ.get("USE_ZSTD", "0")))
    zstd_level = int(os.environ.get("ZSTD_LEVEL", 22))
```

- [ ] **Step 2: Add zstd import at top of file**

After the existing imports (around line 18), add:

```python
try:
    import zstandard as zstd
except ImportError:
    zstd = None
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('train_gpt.py').read()); print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add env vars for all H100 innovations (WD, freq skip, QAT, EMA, deep supervision, etc.)"
```

---

### Task 2: LeakyReLU(0.5)² activation in MLP

**Files:**
- Modify: `train_gpt.py:652-665` (MLP class)

- [ ] **Step 1: Change ReLU² to LeakyReLU(0.5)²**

Replace the MLP forward method (lines 663-665):

```python
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = torch.where(x > 0, x, 0.5 * x)  # LeakyReLU(0.5)
        return self.proj(x.square())
```

- [ ] **Step 2: Update class docstring**

Replace line 653:
```python
    # LeakyReLU(0.5)² activation: preserves 50% gradient for negatives (no dead neurons),
    # squaring provides sparsity. -0.0084 BPB vs ReLU² on Mac.
```

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: LeakyReLU(0.5)² activation — eliminates dead neurons, -0.008 BPB"
```

---

### Task 3: Asymmetric MLP width (encoder=2x, decoder=4x)

**Files:**
- Modify: `train_gpt.py:703-744` (GPT.__init__, block construction)

- [ ] **Step 1: Add mlp_mult_asymmetric parameter to GPT**

Add `mlp_mult_asymmetric: str = ""` to GPT.__init__ signature (after `qk_gain_init` on line 716):

```python
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mlp_mult_asymmetric: str = "",
    ):
```

- [ ] **Step 2: Use per-layer MLP widths in block construction**

Replace the block construction (lines 732-743):

```python
        # Per-layer MLP width: asymmetric allows different encoder/decoder widths
        if mlp_mult_asymmetric:
            enc_m, dec_m = (int(x) for x in mlp_mult_asymmetric.split(","))
            layer_mlp_mults = [enc_m] * self.num_encoder_layers + [dec_m] * self.num_decoder_layers
        else:
            layer_mlp_mults = [mlp_mult] * num_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    layer_mlp_mults[i],
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
```

- [ ] **Step 3: Pass mlp_mult_asymmetric from main()**

In model construction (line 1124-1136), add the parameter:

```python
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mlp_mult_asymmetric=args.mlp_mult_asymmetric,
    ).to(device).bfloat16()
```

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: asymmetric MLP width (encoder=2x, decoder=4x) — -0.001 BPB"
```

---

### Task 4: Warmdown-aware weight decay in Muon

**Files:**
- Modify: `train_gpt.py:122-182` (Muon class)

- [ ] **Step 1: Add weight_decay parameter to Muon**

Update `__init__` (lines 123-127):

```python
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )
```

- [ ] **Step 2: Apply warmdown-aware WD in step()**

After the parameter update loop (line 176-180), replace with:

```python
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                wd = group["weight_decay"]
                if wd > 0:
                    # Warmdown-aware WD: wd_eff = base_wd * (2 - lr_mul)
                    # lr_mul = current_lr / base_lr. When lr_mul=1 → wd. When lr_mul→0 → 2*wd.
                    base_lr = group.get("base_lr", group["lr"])
                    lr_mul = group["lr"] / base_lr if base_lr > 0 else 1.0
                    wd_eff = wd * (2.0 - lr_mul)
                    p.data.mul_(1.0 - group["lr"] * wd_eff)
                p.add_(g, alpha=-group["lr"])
                curr += p.numel()
```

- [ ] **Step 3: Pass weight_decay when constructing Muon in main()**

Update line 1172-1177:

```python
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
```

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: warmdown-aware WD in Muon — wd*(2-lr_mul), our original, -0.010 BPB"
```

---

### Task 5: Frequency-decomposed skip gating

**Files:**
- Modify: `train_gpt.py:703-798` (GPT class)

- [ ] **Step 1: Add freq_skip_gating to GPT.__init__**

Add parameters to signature (after `mlp_mult_asymmetric`):

```python
        freq_skip_gating: bool = False,
        freq_skip_window: int = 32,
```

Store them and create per-band skip weights (after `self.skip_weights` at line 731):

```python
        self.freq_skip_gating = freq_skip_gating
        self.freq_skip_window = freq_skip_window
        if freq_skip_gating:
            self.skip_lo_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
            self.skip_hi_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        else:
            pass  # keep existing skip_weights
```

Note: `skip_weights` is always created (line 731). When `freq_skip_gating=True`, we add lo/hi weights and ignore `skip_weights` in forward. Both lo/hi and skip_weights are always in the state dict to avoid shape mismatches.

- [ ] **Step 2: Modify forward() decoder skip logic**

Replace lines 778-779 in forward():

```python
            if skips:
                skip = skips.pop()
                if self.freq_skip_gating:
                    W = self.freq_skip_window
                    s = skip.reshape(*skip.shape[:-1], -1, W)
                    lo = s.mean(dim=-1, keepdim=True).expand_as(s).reshape(skip.shape)
                    hi = skip - lo
                    x = x + (self.skip_lo_weights[i].to(dtype=x.dtype)[None, None, :] * lo +
                             self.skip_hi_weights[i].to(dtype=x.dtype)[None, None, :] * hi)
                else:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
```

- [ ] **Step 3: Pass from main() and add to scalar_params**

In model construction, add the arguments. In the scalar_params list (line 1158-1164), also include skip_lo_weights and skip_hi_weights:

```python
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if hasattr(base_model, 'skip_lo_weights'):
        scalar_params.append(base_model.skip_lo_weights)
        scalar_params.append(base_model.skip_hi_weights)
```

Also update CONTROL_TENSOR_NAME_PATTERNS to include the new names:
Add `skip_lo_weight,skip_hi_weight` to the default pattern string.

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: frequency-decomposed skip gating (2-band lo/hi, W=32) — our original, -0.010 BPB"
```

---

### Task 6: Per-layer LR scaling in Muon

**Files:**
- Modify: `train_gpt.py:122-182` (Muon), `train_gpt.py:1147-1194` (optimizer setup in main)

- [ ] **Step 1: Store per-param LR scales in Muon**

After `__init__` (line 127), add a method:

```python
    def set_layer_lr_scales(self, param_to_scale: dict):
        """Set per-parameter LR multipliers for layer-dependent learning rates."""
        self._param_lr_scales = param_to_scale
```

In `step()`, when applying the update (in the parameter update loop), scale the LR:

```python
                lr_scale = self._param_lr_scales.get(id(p), 1.0) if hasattr(self, '_param_lr_scales') else 1.0
                effective_lr = group["lr"] * lr_scale
                p.add_(g, alpha=-effective_lr)
```

Also apply the same scale to WD if present.

- [ ] **Step 2: Compute and pass scales in main()**

After creating `optimizer_muon` (line 1177), add:

```python
    if args.layer_lr_scale != 0:
        param_to_scale = {}
        n = max(args.num_layers - 1, 1)
        for name, p in base_model.blocks.named_parameters():
            if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                layer_idx = int(name.split(".")[0])
                param_to_scale[id(p)] = 1.0 + args.layer_lr_scale * (layer_idx / n)
        optimizer_muon.set_layer_lr_scales(param_to_scale)
```

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: per-layer LR scaling in Muon — deeper layers get higher LR, -0.001 BPP"
```

---

### Task 7: EMA shadow model

**Files:**
- Modify: `train_gpt.py:1274-1454` (training loop and serialization in main)

- [ ] **Step 1: Create EMA state after model setup**

After optimizer setup (after line 1194), add:

```python
    # EMA shadow model: maintain exponential moving average of weights
    ema_state = None
    if args.ema_decay > 0:
        ema_state = {k: v.clone() for k, v in base_model.state_dict().items()}
        log0(f"ema:enabled decay={args.ema_decay}")
```

- [ ] **Step 2: Update EMA every step in training loop**

After `opt.step()` for all optimizers (after line 1344), add:

```python
        # EMA update: shadow = decay * shadow + (1 - decay) * current
        if ema_state is not None:
            with torch.no_grad():
                for k, v in base_model.state_dict().items():
                    ema_state[k].mul_(args.ema_decay).add_(v, alpha=1.0 - args.ema_decay)
```

- [ ] **Step 3: Load EMA weights before serialization**

Before serialization (before line 1382), add:

```python
    # Swap in EMA weights for final eval and serialization
    if ema_state is not None:
        log0("Loading EMA weights for final evaluation and serialization")
        base_model.load_state_dict(ema_state)
```

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: EMA shadow model (decay=0.997) — standard practice, essential for H100"
```

---

### Task 8: Pre-warmdown QAT and zstd compression

**Files:**
- Modify: `train_gpt.py` (training loop for QAT, serialization for zstd)

- [ ] **Step 1: Add sim_quant_roundtrip function**

Before the quantization section (~line 337), add:

```python
def sim_quant_roundtrip(w: Tensor) -> Tensor:
    """Simulate int8 quantize→dequantize in PyTorch ops (stays on GPU)."""
    f = w.float()
    qmax = 127.0
    if f.ndim == 2:
        row_max = f.abs().amax(dim=1, keepdim=True).clamp(min=1.0 / qmax)
        scale = row_max / qmax
        q = (f / scale).round().clamp(-qmax, qmax)
        return (q * scale).to(w.dtype)
    amax = f.abs().amax().clamp(min=1.0 / qmax)
    scale = amax / qmax
    q = (f / scale).round().clamp(-qmax, qmax)
    return (q * scale).to(w.dtype)
```

- [ ] **Step 2: Add QAT in training loop**

After the optimizer step block (after line 1344), add:

```python
        # Pre-warmdown QAT: nudge weights toward quantized form
        if args.qat_prewarmdown and scale >= args.qat_stop_lr_mul and step % args.qat_every == 0:
            with torch.no_grad():
                for name, p in base_model.named_parameters():
                    if not p.is_floating_point() or p.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
                        continue
                    if args.int8_keep_float_fp16_name_patterns and any(
                        pat in name for pat in args.int8_keep_float_fp16_name_patterns.split(",") if pat
                    ):
                        continue
                    p_q = sim_quant_roundtrip(p)
                    p.data.add_(p_q - p, alpha=args.qat_strength)
```

- [ ] **Step 3: Add zstd compression option in serialization**

Replace the zlib compression (lines 1393-1394):

```python
    if args.use_zstd and zstd is not None:
        compressor = zstd.ZstdCompressor(level=args.zstd_level)
        quant_blob = compressor.compress(quant_raw)
        compression_name = f"zstd-{args.zstd_level}"
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
        compression_name = "zlib-9"
```

Update the log message and filename accordingly (use `.ptz` for zlib, `.ptzst` for zstd, or just keep `.ptz`).

- [ ] **Step 4: Add FP16 embedding passthrough in quantization**

In `quantize_state_dict_int8` (around line 402-406), after the small-tensor passthrough check, add:

```python
        # Named patterns can force specific large tensors to stay as FP16
        fp16_patterns = tuple(p for p in os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS", "").split(",") if p)
        if fp16_patterns and any(p in name for p in fp16_patterns):
            kept = keep_float_tensor(name, tensor, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
```

- [ ] **Step 5: Commit**

```bash
git add train_gpt.py
git commit -m "feat: pre-warmdown QAT + zstd compression + FP16 embedding passthrough"
```

---

## Phase 2: Per-Run Features

### Task 9: Deep supervision (Runs 3, 5, 6)

**Files:**
- Modify: `train_gpt.py` (GPT class forward, loss computation)

- [ ] **Step 1: Add deep supervision to GPT forward**

Add parameters to GPT.__init__ signature:

```python
        deep_supervision: bool = False,
        deep_supervision_alpha: float = 0.1,
        deep_supervision_tap_layers: list[int] | None = None,
```

Store them:
```python
        self._deep_supervision = deep_supervision
        self._ds_alpha = deep_supervision_alpha
        self._ds_taps = set(deep_supervision_tap_layers or [])
```

- [ ] **Step 2: Modify forward to compute auxiliary losses**

In the GPT.forward method, collect hidden states at tap points and compute auxiliary CE losses. The forward already returns a loss scalar — add auxiliary losses to it:

```python
    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        aux_states: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[i](x, x0, qd, vd)
            skips.append(x)
            if self._deep_supervision and i in self._ds_taps:
                aux_states.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                skip = skips.pop()
                if self.freq_skip_gating:
                    W = self.freq_skip_window
                    s = skip.reshape(*skip.shape[:-1], -1, W)
                    lo = s.mean(dim=-1, keepdim=True).expand_as(s).reshape(skip.shape)
                    hi = skip - lo
                    x = x + (self.skip_lo_weights[i].to(dtype=x.dtype)[None, None, :] * lo +
                             self.skip_hi_weights[i].to(dtype=x.dtype)[None, None, :] * hi)
                else:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = self.blocks[bi](x, x0, qd, vd)
            if self._deep_supervision and bi in self._ds_taps:
                aux_states.append(x)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(
                logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none").reshape(bsz, sl)

        main_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")

        # Auxiliary losses from deep supervision taps
        if self._deep_supervision and aux_states:
            n_aux = len(aux_states)
            emb_w = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
            for idx, ax in enumerate(aux_states):
                ax_norm = F.rms_norm(ax, (ax.size(-1),))
                aux_logits = F.linear(ax_norm, emb_w)
                aux_logits = self.logit_softcap * torch.tanh(aux_logits / self.logit_softcap)
                aux_ce = F.cross_entropy(aux_logits.float().reshape(-1, aux_logits.size(-1)), target_ids.reshape(-1), reduction="mean")
                weight = self._ds_alpha ** (n_aux - idx)
                main_loss = main_loss + weight * aux_ce

        return main_loss
```

- [ ] **Step 3: Pass from main()**

Add deep supervision args to GPT construction. Compute default tap layers:

```python
        deep_supervision=args.deep_supervision,
        deep_supervision_alpha=args.deep_supervision_alpha,
        deep_supervision_tap_layers=(
            [int(x) for x in args.deep_supervision_layers.split(",") if x]
            if args.deep_supervision_layers
            else list(range(1, args.num_layers - 1, 2))
        ) if args.deep_supervision else None,
```

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: deep supervision with shared-embedding aux heads — our original, zero extra params"
```

---

### Task 10: Calibrated per-row quantization (Runs 4, 6)

**Files:**
- Modify: `train_gpt.py:350-428` (quantization pipeline)

- [ ] **Step 1: Add calibrated quantization function**

After `quantize_float_tensor` (line 369), add:

```python
def quantize_float_tensor_calibrated(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row int8 quantization with MSE-optimal clip percentile selection.
    Tries 5 candidate percentiles per row, picks the one minimizing reconstruction MSE."""
    if tensor.ndim != 2:
        return quantize_float_tensor(tensor)  # only calibrate 2D weights

    f32 = tensor.detach().float().cpu()
    candidates = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
    best_q = torch.zeros_like(f32, dtype=torch.int8)
    best_scale = torch.zeros(f32.size(0), dtype=torch.float32)

    for row_idx in range(f32.size(0)):
        row = f32[row_idx]
        best_mse = float('inf')
        for clip_q in candidates:
            clip_abs = float(row.abs().quantile(clip_q)) if row.numel() > 0 else 0.0
            scale = max(clip_abs / 127.0, 1.0 / 127.0)
            clipped = row.clamp(-clip_abs, clip_abs)
            q = (clipped / scale).round().clamp(-127, 127).to(torch.int8)
            recon = q.float() * scale
            mse = ((row - recon) ** 2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_q[row_idx] = q
                best_scale[row_idx] = scale

    return best_q.numpy(), best_scale.to(torch.float16).numpy()
```

- [ ] **Step 2: Use calibrated version when enabled**

In `quantize_state_dict_int8` (line ~408-415), replace the `quantize_float_tensor` call:

```python
        if calibrated and tensor.ndim == 2:
            q, s = quantize_float_tensor_calibrated(tensor)
        else:
            q, s = quantize_float_tensor(tensor)
```

Pass `calibrated` as a parameter to `quantize_state_dict_int8`.

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: calibrated per-row int8 quantization — MSE-optimal clipping, our own implementation"
```

---

### Task 11: Progressive layer growing (Run 5)

**Files:**
- Modify: `train_gpt.py` (main training loop)

- [ ] **Step 1: Add layer growing logic in training loop**

After the curriculum transition code location (inside the `while True` loop, before the training step), add:

```python
        # Progressive layer growing: reconstruct model at phase boundary
        if args.grow_layers_from > 0 and not layer_grown:
            elapsed_frac = approx_training_time_ms / max_wallclock_ms if max_wallclock_ms else 0
            if elapsed_frac >= args.grow_at_wallclock_frac:
                layer_grown = True
                log0(f"growing model: {args.grow_layers_from}L -> {args.num_layers}L at frac={elapsed_frac:.3f}")
                # Save current small model weights
                small_state = {k: v.clone() for k, v in base_model.state_dict().items()}
                # Rebuild model at full depth
                # ... (reconstruct model, copy matching weights, zero-init new layers)
                # ... (rebuild optimizers)
                # ... (recompile)
```

This is the most complex feature. The implementation needs to:
1. Save the small model's state dict
2. Create a new GPT with `args.num_layers`
3. Copy weights from matching layers (encoder layers map to encoder, decoder to decoder)
4. Zero-init new layer output projections
5. Rebuild all optimizers with new parameter lists
6. Re-wrap in DDP and torch.compile
7. If EMA is active, rebuild EMA state

**Note**: This is complex enough that it should be implemented as a standalone function `grow_model()` that returns the new model, optimizers, and EMA state. Full implementation TBD based on testing.

- [ ] **Step 2: Initialize tracking variables before loop**

```python
    layer_grown = False if args.grow_layers_from > 0 else True
```

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: progressive layer growing (8L->13L) — our original moonshot idea"
```

---

## Phase 3: Run Scripts and Code Review

### Task 12: Create run scripts for all 6 runs

**Files:**
- Create: `h100_run1_proven.sh`
- Create: `h100_run2_deep_capacity.sh`
- Create: `h100_run3_training_amplifier.sh`
- Create: `h100_run4_calibrated.sh`
- Create: `h100_run5_progressive.sh`
- Create: `h100_run6_everything.sh`

- [ ] **Step 1: Write all 6 run scripts**

Each script sets env vars and launches training. Example for Run 1:

```bash
#!/bin/bash
# Run 1: Proven Foundation — all validated innovations, zero experimental risk
export RUN_ID=h100_run1_proven
export NUM_LAYERS=11
export MLP_MULT_ASYMMETRIC=2,4
export MUON_WEIGHT_DECAY=0.10
export FREQ_SKIP_GATING=1
export GRAD_CLIP_NORM=0.5
export LAYER_LR_SCALE=0.5
export WARMUP_STEPS=50
export QAT_PREWARMDOWN=1
export QAT_STRENGTH=0.1
export QAT_STOP_LR_MUL=0.8
export QAT_EVERY=10
export EMA_DECAY=0.997
export USE_ZSTD=1
export ZSTD_LEVEL=22
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
export ITERATIONS=20000
export VAL_LOSS_EVERY=1000
torchrun --nproc_per_node=8 train_gpt.py
```

Run 6 (Everything):
```bash
#!/bin/bash
# Run 6: Everything, Everywhere, All At Once
export RUN_ID=h100_run6_everything
export NUM_LAYERS=13
export MLP_MULT_ASYMMETRIC=2,4
export MUON_WEIGHT_DECAY=0.12
export FREQ_SKIP_GATING=1
export GRAD_CLIP_NORM=0.5
export LAYER_LR_SCALE=0.5
export WARMUP_STEPS=50
export QAT_PREWARMDOWN=1
export QAT_STRENGTH=0.1
export QAT_STOP_LR_MUL=0.8
export QAT_EVERY=10
export EMA_DECAY=0.997
export USE_ZSTD=1
export ZSTD_LEVEL=22
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
export DEEP_SUPERVISION=1
export DEEP_SUPERVISION_ALPHA=0.05
export DEEP_SUPERVISION_LAYERS=3,5,9,11
export CALIBRATED_QUANT=1
export ITERATIONS=20000
export VAL_LOSS_EVERY=1000
torchrun --nproc_per_node=8 train_gpt.py
```

- [ ] **Step 2: Commit all scripts**

```bash
git add h100_run*.sh
git commit -m "feat: 6 H100 competition run scripts — differentiated approaches"
```

---

### Task 13: Code review — verify single-file cleanliness

- [ ] **Step 1: Check file length**

Run: `wc -l train_gpt.py`
Target: ≤1500 lines (hard stop from code header). If over, trim comments or consolidate.

- [ ] **Step 2: Verify all env vars have defaults that match baseline**

Every new env var must default to the baseline behavior (disabled). Run with zero env vars and confirm the script behaves identically to the original.

- [ ] **Step 3: Run syntax check**

```bash
python3 -c "import ast; ast.parse(open('train_gpt.py').read()); print('OK')"
```

- [ ] **Step 4: Verify torch.compile compatibility**

The deep supervision's variable-length `aux_states` list and conditional logic might break `torch.compile(dynamic=False, fullgraph=True)`. Test with:

```bash
DEEP_SUPERVISION=1 python3 -c "
import torch
from train_gpt import GPT
model = GPT(1024, 11, 512, 8, 4, 2, True, 0.005, 30.0, 10000.0, 1.5,
            deep_supervision=True, deep_supervision_alpha=0.1, deep_supervision_tap_layers=[3,7])
model = model.cuda().bfloat16()
compiled = torch.compile(model, dynamic=False, fullgraph=True)
x = torch.randint(0, 1024, (2, 64), device='cuda')
y = torch.randint(0, 1024, (2, 64), device='cuda')
with torch.autocast('cuda', torch.bfloat16):
    loss = compiled(x, y)
print(f'Loss: {loss.item():.4f}')
"
```

If `fullgraph=True` fails with deep supervision, fall back to `fullgraph=False` when `DEEP_SUPERVISION=1`.

- [ ] **Step 5: Final commit with all fixes**

```bash
git add train_gpt.py
git commit -m "fix: code review — verify cleanliness, torch.compile compat, env var defaults"
```

---

## Verification

1. **Baseline equivalence**: Run with zero new env vars → identical to original `train_gpt.py`
2. **Per-feature smoke test**: Enable one feature at a time on Mac/single-GPU, verify no crashes
3. **Artifact size check**: Run 2 and 4 (13L) must produce <16MB artifacts with zstd-22
4. **H100 full runs**: Execute all 6 run scripts on RunPod, collect results
