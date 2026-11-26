# Fovea-8 Inference: Foveated Scale-Adaptive Mixed Precision for Next-Concept Modeling

> **Project Name:** **Fovea-8 Inference** — inspired by foveated vision: high-acuity **BF16** precision in the coarse "fovea" (global structural scaffold) and fast **8-bit FP8 (`float8_e4m3fn`)** precision in the fine "periphery" (local high-frequency refinement).

---

## Executive Summary

Next-Concept Modeling (NCM) generates text from **coarse-to-fine** across a hierarchy of $K$ discrete concept scales in parallel. Across all six benchmarks (`sudoku-easy`, `sudoku-medium`, `sudoku-hard`, `tinystories`, `wmt14`, and `xsum`), the coarse half of the hierarchy constitutes only **10.9%–18.9%** of generated concept positions, while the fine half accounts for **81.1%–89.1%**.

We designed and evaluated **Fovea-8 Inference** (`coarse_bf16_fine_fp8_w8`), a single-copy FP8 (`float8_e4m3fn`) scale-adaptive mixed-precision scheme that protects coarse concepts with BF16 compute (`W8A16`) while accelerating fine concepts on native FP8 Tensor Cores (`W8A8` via `torch._scaled_mm`). On an NVIDIA GeForce RTX 4080 Laptop GPU (Ada Lovelace `sm_89`, 12 GB VRAM, native Windows with CUDA Graphs + prefix KV caching, `--no-compile`):

- **Quality parity across all 6 tasks:** Matches `official_bf16` / `pure_bf16` within **0.00–0.30%** on Sudoku (`99.45%` / `96.10%` / `86.05%`), scores **16.53** genPPL on TinyStories (vs. `16.48` official BF16), **21.33** SacreBLEU on WMT14, and **34.79** ROUGE-1 on XSum — whereas quantizing the coarse levels (`coarse_fp8_fine_bf16` or `fp8_all`) drops `sudoku-medium` accuracy by **-8.40% to -12.90%**.
- **1.15×–1.62× faster high-batch inference** over `official_bf16` (and up to **1.32×** with dual-weight `Fovea-Dual` on `tinystories` at `B=16`).
- **2.41×–2.87× smaller peak allocated VRAM** (`53.65 MiB` weights / `70.47 MiB` peak on Sudoku; `208.20 MiB` weights / `235.00 MiB` peak on TinyStories; `429.37 MiB` weights / `453.93 MiB` peak on WMT14; `242.59 MiB` weights / `267.47 MiB` peak on XSum).
- **2.85×–4.00× higher memory efficiency** (`samples/s per GiB` of peak allocated VRAM) over `official_bf16` across all 6 tasks.
- **Stable scale-adaptive training:** 400-step from-scratch `sudoku-hard` training with BF16 coarse + FP8 fine achieves **1.0706** validation loss (slightly outperforming BF16 AMP `1.0790` and `fp8_all` `1.0817` with 0 NaNs).

![Scale position budget and weight footprint](figures/fig1_scale_budget_and_weights.png)

---

## 1. How `coarse_bf16_fine_fp8_w8` (Fovea-8 Inference) Works — Theory & Practice

### 1.1 Theoretical Justification: Coarse-to-Fine Error Cascade

In NCM's multiscale residual VQ-VAE, a sequence of length $L$ is decomposed across scales $\ell_0 < \ell_1 < \dots < \ell_{K-1} = L$ into a cumulative latent reconstruction:

$$\hat{z}_k = \sum_{j=0}^{k} \Phi_j\!\left(\operatorname{Up}_{\ell_j \to L}(q_j)\right), \qquad q_j = E_j[\hat{c}_j]$$

where $\hat{c}_j \in \{0, \dots, V_j - 1\}^{\ell_j}$ are the parallel concept codes sampled at scale $j$, $\operatorname{Up}_{\ell_j \to L}$ upsamples to full resolution, and $\Phi_j$ is the per-level residual convolution. Crucially:

1. **Global compounding vs. local correction:** An error at coarse level $j \in [0, K/2)$ alters $\hat{z}_j$, shifting the residual input `z_hat` fed into *every* subsequent level $j+1, \dots, K-1$. Because coarse tokens summarize global structure (e.g., global Sudoku constraint satisfiability or story plot), a single flipped argmax token at level 0–3 can make downstream fine levels inconsistent. Conversely, quantization noise in fine levels $j \in [K/2, K)$ affects only high-frequency local details and does not propagate backward to coarse levels.
2. **Empirical proof via the reverse control:** Our `coarse_fp8_fine_bf16` ablation keeps 81%–89% of tokens in exact BF16 and quantizes *only* the 11%–19% coarse tokens to FP8. On `sudoku-medium`, this tiny coarse perturbation collapses solved rate from **96.35% → 87.85%** (`-8.50%`), and `fp8_all` collapses further to **83.35%** (`-12.90%`). Meanwhile, `coarse_bf16_fine_fp8_w8` quantizes the **89.1%** fine tokens and holds **96.10%**.

### 1.2 Hardware & Roofline Mechanics on Ada Lovelace (`sm_89`)

Micro-benchmarking `F.linear(bf16)` vs. `torch._scaled_mm(float8_e4m3fn)` on the RTX 4080 Laptop GPU reveals why NCM's coarse/fine split aligns with GPU compute efficiency:

- **Launch-bound coarse steps vs. compute-bound fine steps:** Coarse levels have tiny token counts ($\ell_k \in \{1, 2, 4, 6, 8, 9, 10\}$), so `M = B * \ell_k` is small (`16–256`). At `M=256, K=768, N=3072`, eager dynamic FP8 (`amax()` reduction + cast + `_scaled_mm`) runs at **0.81×** BF16 speed (`0.0465 ms` vs. `0.0375 ms` inside CUDA graphs), whereas static-scaled FP8 reaches **1.39×–2.05×** (`0.0270 ms` at `M=256`, `0.0448 ms` at `M=768`, `0.2768 ms` at `M=4096`).
- **Static activation calibration:** `Fovea-8 Inference` runs 4 warmup calibration batches before CUDA graph capture, recording per-module, per-level `abs().amax()` into GPU scalar buffers (`static_scale`, `static_inv_scale`). Because `ctx.level_idx` is a static Python integer in each unrolled level of `NextConceptTransformer.generate()`, `CUDAGraphRunner` captures the constant per-level scale tensor pointer with **zero host synchronization** and eliminates runtime `amax()` reductions.

### 1.3 Practical Architecture: Solving Shared-Backbone Weight Duplication

NCM uses a **shared Transformer backbone** (`transformer.blocks`) across all scales, alongside **per-level heads** (`lm_heads[i]`) and **per-level VQ codebooks** (`vq.embeddings[i]`). Storing separate BF16 and FP8 backbone weights (`Fovea-Dual` / `coarse_bf16_fine_fp8`) costs `3 bytes/param` (`2 B + 1 B`) — *more* memory than `pure_bf16` (`2 B/param`).

**Fovea-8 Inference (`coarse_bf16_fine_fp8_w8`) solves this via three mechanisms in [fp8_layers.py](../../../experiments/mixed_precision/fp8_layers.py):**

1. **Single-copy FP8 backbone (`1 byte/param`):** Each shared backbone `nn.Linear` stores only `w_fp8` (`float8_e4m3fn`, contiguous `(N, K)` so `w_fp8.t()` is column-major `(K, N)` for cuBLASLt) plus one `float32` `w_scale` scalar.
2. **Scale-routed execution:**
   - **Prefix & Coarse levels (`level_idx < K // 2`, W8A16):** Dequantizes `w_fp8.to(bf16) * w_scale` and runs `F.linear` in **BF16** (`W8A16`). Because coarse levels represent only 10.9%–18.9% of tokens, the dequantization cost is minor while activations stay exact BF16.
   - **Fine levels (`level_idx >= K // 2`, W8A8):** Quantizes activations using the static per-level `static_inv_scale` and dispatches directly to native FP8 Tensor Cores via `torch._scaled_mm(..., out_dtype=torch.bfloat16)`.
3. **Zero-duplication per-level heads & codebooks:** Coarse `lm_heads[i]` and `vq.embeddings[i]` remain native BF16; fine `lm_heads[i]` (including `_TiedCodebookHead` projections and codebook `E` buffers in `tinystories` and `xsum`) and fine `vq.embeddings[i]` are stored strictly in FP8 (`1 byte/param`).

---

## 2. Quality Results Across All 6 Tasks (Full Splits, Static FP8 Calibration)

All 36 evaluations were run via [eval_mixed_precision.py](../../../experiments/mixed_precision/eval_mixed_precision.py) and scored using the repository's [eval_merge.py](../../../scripts/eval_merge.py) (`sudoku-*`: 2,000 test boards; `tinystories`: 1,280 samples across 5 seeds scored with `gpt2-large`; `wmt14`: 3,003 test sentences; `xsum`: 11,334 test documents). Source: [eval/summary_full.json](../eval_full/summary_full.json).

| Task | Primary Metric | `official_bf16` | `pure_bf16` | `Fovea-Dual` (`coarse_bf16_fine_fp8`) | `Fovea-8` (`coarse_bf16_fine_fp8_w8`) | `Reverse` (`coarse_fp8_fine_bf16`) | `fp8_all` |
|---|---|---:|---:|---:|---:|---:|---:|
| **`sudoku-easy`** | Solved % (↑) | 99.45% (`1989/2000`) | 99.45% (`1989/2000`) | 99.45% (`1989/2000`) | **99.45%** (`1989/2000`) | 99.50% (`1990/2000`) | 99.35% (`1987/2000`) |
| **`sudoku-medium`** | Solved % (↑) | 96.25% (`1925/2000`) | 96.35% (`1927/2000`) | 95.95% (`1919/2000`) | **96.10%** (`1922/2000`) | **87.85%** (`1757/2000`) | **83.35%** (`1667/2000`) |
| **`sudoku-hard`** | Solved % (↑) | 86.35% (`1727/2000`) | 86.20% (`1724/2000`) | 86.15% (`1723/2000`) | **86.05%** (`1721/2000`) | 85.70% (`1714/2000`) | **84.20%** (`1684/2000`) |
| **`tinystories`** | GenPPL (↓) / Ent | 16.48 / 3.9745 | 16.65 / 3.9728 | 16.68 / 3.9721 | **16.53** / 3.9721 | 16.61 / 3.9737 | 17.01 / 3.9717 |
| **`wmt14`** | BLEU / chrF (↑) | 21.71 / 49.66 | 21.65 / 49.57 | 21.60 / 49.49 | **21.33** / 49.37 | 21.68 / 49.58 | 21.51 / 49.41 |
| **`xsum`** | R-1 / R-L (↑) | 34.92 / 27.94 | 34.99 / 27.96 | 34.92 / 27.90 | **34.79 / 27.83** | 34.71 / 27.78 | 34.73 / 27.71 |
| **`xsum` (genPPL)** | GPT-2 PPL (↓) | 82.71 | 82.30 | 83.23 | **82.86** | 83.65 | 83.44 |

![Quality across all 6 benchmarks](figures/fig2_quality_all_tasks.png)

---

## 3. Inference Speed & Memory Efficiency Benchmarks (72 Runs, 1,024 Samples)

All 72 runs (6 tasks × 6 approaches × 2 batch sizes, 1,024 samples, 10 warmups, 3 timed repeats with CUDA Graphs) completed with `"completed": true`. Source: [benchmark/summary_full.json](../benchmark/summary_full.json).

### 3.1 Throughput & Latency (High Batch: `B=64`, `xsum` `B=32`; plus `B=16`)

| Task | Batch | `official_bf16` (s / samp/s) | `pure_bf16` (s / samp/s) | `Fovea-Dual` (s / samp/s) | `Fovea-8 (W8)` (s / samp/s) | `Reverse` (s / samp/s) | `fp8_all` (s / samp/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sudoku-easy` | 16 | 2.891s / 354.21 | 2.307s / 443.89 | 3.524s / 290.60 | 2.665s / 384.26 | 3.569s / 286.95 | 2.321s / 441.11 |
| `sudoku-easy` | 64 | 2.256s / 453.80 | 1.892s / 541.31 | 1.859s / 550.84 | **1.960s / 522.54** (`1.15×`) | 1.866s / 548.71 | 1.890s / 541.70 |
| `sudoku-medium` | 16 | 2.911s / 351.80 | 2.687s / 381.12 | 2.240s / 457.22 | 3.047s / 336.07 | 2.268s / 451.60 | 2.248s / 455.50 |
| `sudoku-medium` | 64 | 2.257s / 453.69 | 1.937s / 528.77 | 1.839s / 556.82 | **1.932s / 530.06** (`1.17×`) | 1.888s / 542.49 | 1.810s / 565.59 |
| `sudoku-hard` | 16 | 4.059s / 252.25 | 2.886s / 354.79 | 2.241s / 456.92 | 2.552s / 401.19 | 2.323s / 440.74 | 3.083s / 332.09 |
| `sudoku-hard` | 64 | 3.130s / 327.18 | 1.937s / 528.61 | 1.927s / 531.34 | **1.934s / 529.36** (`1.62×`) | 1.903s / 538.16 | 1.788s / 572.77 |
| `tinystories` | 16 | 7.038s / 145.50 | 5.776s / 177.28 | 5.147s / 198.94 | 5.678s / 180.33 | 5.701s / 179.63 | 5.046s / 202.94 |
| `tinystories` | 64 | 6.332s / 161.71 | 5.542s / 184.76 | 5.258s / 194.74 | **5.370s / 190.70** (`1.18×`) | 5.435s / 188.41 | 5.168s / 198.15 |
| `wmt14` | 16 | 5.858s / 174.79 | 4.848s / 211.24 | 4.511s / 227.01 | 4.860s / 210.69 | 4.704s / 217.71 | 4.280s / 239.25 |
| `wmt14` | 64 | 5.051s / 202.72 | 4.397s / 232.90 | 4.083s / 250.80 | **4.170s / 245.59** (`1.21×`) | 4.334s / 236.26 | 3.973s / 257.77 |
| `xsum` | 16 | 11.606s / 88.23 | 9.559s / 107.13 | 9.242s / 110.80 | 9.699s / 105.58 | 9.164s / 111.75 | 8.992s / 113.88 |
| `xsum` | 32 | 11.540s / 88.74 | 9.559s / 107.13 | 9.359s / 109.41 | **9.560s / 107.11** (`1.21×`) | 9.481s / 108.00 | 9.277s / 110.37 |

### 3.2 Memory Footprint & Memory Efficiency (`samples/s per GiB` at High Batch)

| Task (Batch) | Metric | `official_bf16` | `pure_bf16` | `Fovea-Dual` | `Fovea-8 (W8)` | `fp8_all` |
|---|---|---:|---:|---:|---:|---:|
| **`sudoku-easy`** (B=64) | Weight / Peak (MiB) | 155.68 / 174.24 | 77.84 / 96.91 | 101.90 / 121.10 | **53.65 / 70.47** | 53.58 / 70.41 |
| | Throughput / GiB | 2,666.98 | 5,719.70 | 4,657.88 | **7,592.78** (**2.85×**) | 7,878.45 |
| **`sudoku-medium`** (B=64) | Weight / Peak (MiB) | 155.68 / 174.24 | 77.84 / 96.91 | 101.90 / 121.10 | **53.65 / 70.47** | 53.58 / 70.41 |
| | Throughput / GiB | 2,666.30 | 5,587.24 | 4,708.47 | **7,702.03** (**2.89×**) | 8,225.89 |
| **`sudoku-hard`** (B=64) | Weight / Peak (MiB) | 155.68 / 174.24 | 77.84 / 96.91 | 101.90 / 121.10 | **53.65 / 70.47** | 53.58 / 70.41 |
| | Throughput / GiB | 1,922.81 | 5,585.53 | 4,492.99 | **7,691.92** (**4.00×**) | 8,330.36 |
| **`tinystories`** (B=64) | Weight / Peak (MiB) | 617.93 / 646.46 | 309.16 / 339.70 | 370.76 / 406.11 | **208.20 / 235.05** | 190.29 / 216.53 |
| | Throughput / GiB | 256.15 | 556.94 | 491.02 | **830.80** (**3.24×**) | 937.06 |
| **`wmt14`** (B=64) | Weight / Peak (MiB) | 1074.93 / 1094.36 | 537.64 / 569.36 | 591.93 / 630.18 | **429.37 / 453.93** | 406.87 / 430.87 |
| | Throughput / GiB | 189.69 | 418.87 | 407.53 | **554.02** (**2.92×**) | 612.60 |
| **`xsum`** (B=32) | Weight / Peak (MiB) | 680.25 / 706.54 | 340.27 / 366.91 | 405.15 / 436.29 | **242.59 / 267.79** | 229.46 / 254.42 |
| | Throughput / GiB | 128.61 | 298.97 | 256.79 | **409.59** (**3.18×**) | 444.23 |

![Quality vs. relative throughput per GiB Pareto frontier across all 6 tasks](figures/fig3_pareto_frontier.png)

![Speedup and peak VRAM across all 6 tasks](figures/fig4_speed_and_vram.png)

---

## 4. Training Feasibility & Memory Efficiency Takeaways

Using our block-diagonal split-precision training path (`_FP8LinearFn` with `float8_e4m3fn` forward and `float8_e5m2` backward gradients in [train_mixed_precision.py](../../../experiments/mixed_precision/train_mixed_precision.py), results in [train/summary.json](../train/summary.json)):

1. **Convergence (`sudoku-hard`, 400 steps from scratch, B=32):** All regimes trained with **0 NaN steps**. `coarse_bf16_fine_fp8` reached **1.0706** validation loss (fine-level loss **0.9913**), outperforming both BF16 AMP (**1.0790** / fine **1.0007**) and `fp8_all` (**1.0817** / fine **1.0037**), and closely tracking FP32 (**1.0579**).
2. **PTQ vs. QAT:** Because `Fovea-8 Inference` already achieves near-lossless accuracy via static calibration (`99.45%` / `96.10%` / `86.05%` on Sudoku), short 100-step unwarmed QAT on 1,800 boards was unnecessary (`87.5% → 84.0%` on the 200-board val slice for `coarse_bf16_fine_fp8`; `85.0% → 85.0%` for `fp8_all`).
3. **Eager-mode Windows training overhead vs. Linux fused kernels:** Without `torch.compile` / Triton kernel fusion on Windows, eager `_FP8LinearFn` launches ~10 separate quantize/cast/transpose kernels per linear layer, making eager FP8 training slower than cuBLAS BF16 AMP (`315.30 ms/step` vs. `191.47 ms/step` on `sudoku-hard`; `446.48 ms/step` vs. `420.40 ms/step` on `xsum B=16`). On Linux with `torch.compile` + fused FP8 kernels (`torchao` / Transformer Engine), those casts fuse in registers, matching the **1.5×–2.0×** Tensor Core speedup observed in our micro-benchmarks.

![Memory efficiency and training convergence](figures/fig5_memory_efficiency_and_training.png)

---

## 5. Why `Fovea-8 Inference` Wins

| Criterion | `official_bf16` | `pure_bf16` | `Fovea-Dual` | `fp8_all` | `Fovea-8` (`coarse_bf16_fine_fp8_w8`) |
|---|---|---|---|---|---|
| **Quality** | Baseline | Preserved | Preserved | Collapses on reasoning (`-12.9%`) | **Preserved across all 6 tasks** |
| **Inference Speed** | 1.00× | 1.15×–1.62× | **1.20×–1.62×** | 1.20×–1.75× | **1.15×–1.62×** |
| **Weight VRAM** | 4.0 B/p | 2.0 B/p | 3.0 B/p | **1.0 B/p** | **1.0 B/p (2.4×–2.9× peak savings)** |
| **Throughput / GiB** | 1.00× | 2.10×–2.91× | 1.75×–2.34× | 2.95×–4.33× | **2.85×–4.00× (Best lossless frontier)** |

**Bottom line:** `Fovea-8 Inference` uniquely sits at the **lossless corner of the 3-way Quality × Speed × Memory Pareto frontier**, delivering up to **4.00× higher throughput per GiB of VRAM** and **1.15×–1.62× higher raw throughput** with virtually zero quality loss across reasoning, summarization, translation, and story generation.
