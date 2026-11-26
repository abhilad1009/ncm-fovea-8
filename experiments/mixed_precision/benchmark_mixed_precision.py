"""Benchmark inference speed and VRAM across NCM mixed-precision regimes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from utils.fast_generation import CUDAGraphRunner, enable_resize_matmuls
from fp8_layers import apply_inference_precision, finalize_calibration, measure_module_vram_mib

TASK_PATHS = {
    "sudoku-easy": ("config/eval/sudoku-easy.yaml", "artifacts/sudoku/easy", "data/sudoku/easy"),
    "sudoku-medium": ("config/eval/sudoku-medium.yaml", "artifacts/sudoku/medium", "data/sudoku/medium"),
    "sudoku-hard": ("config/eval/sudoku-hard.yaml", "artifacts/sudoku/hard", "data/sudoku/hard"),
    "tinystories": ("config/eval/tinystories.yaml", "artifacts/tinystories", ""),
    "wmt14": ("config/eval/wmt14.yaml", "artifacts/wmt14", "data/wmt14"),
    "xsum": ("config/eval/xsum.yaml", "artifacts/xsum", "data/xsum"),
}

TASK_BATCH_SIZES = {
    "sudoku-easy": [16, 64],
    "sudoku-medium": [16, 64],
    "sudoku-hard": [16, 64],
    "tinystories": [16, 64],
    "wmt14": [16, 64],
    "xsum": [16, 32],
}


def run_benchmark(
    task: str,
    mode: str,
    batch_size: int = 16,
    samples: int = 512,
    warmup: int = 10,
    repeats: int = 3,
    act_scaling: str = "dynamic",
) -> dict:
    cfg_rel, ck_rel, dd_rel = TASK_PATHS[task]
    cfg = OmegaConf.load(ROOT / cfg_rel)
    ck = str(ROOT / ck_rel)
    dd = str(ROOT / dd_rel) if dd_rel else ""

    G = cfg.generation
    temp = float(G.get("temperature", 1.0))
    top_k = int(G.get("top_k", 1))
    top_p = float(G.get("top_p", 0.0))
    cfg_scale = float(G.get("cfg_scale", 0.0))
    freq = float(G.get("frequency_penalty", 0.0))
    mu = float(G.get("frequency_deficit", 0.0))
    adj = str(G.get("adjacency_penalty", "") or "")
    split = str(cfg.eval.get("split", "test"))

    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ck, cfg.checkpoints[name])

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
    sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    enable_resize_matmuls(tk.vqvae.vq)
    ctx = apply_inference_precision(net, tk, mode=mode, act_scaling=act_scaling)
    vram_weights = measure_module_vram_mib(net, tk)

    P = int(cfg.model.prefix_len)
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

    def load_allowance(spec: str):
        if ":" in spec:
            path, ref = spec.split(":", 1)
            obj = torch.load(os.path.join(ck, path), map_location="cuda", weights_only=False)
            for k in ref.split("."):
                obj = obj[k]
            return obj.double()
        return torch.load(os.path.join(ck, spec), map_location="cuda").double()

    ALLOW = load_allowance(str(G.frequency_allowance)) if G.get("frequency_allowance") else None
    L0_P = load_allowance(str(G.level0_unigram)).cpu() if G.get("level0_unigram") else None
    UNCOND = not cfg.data or not cfg.data.get(split)
    ADJ = [float(v) for v in adj.split(",")] if adj else None
    L0 = tk.vqvae.vq.levels[0]
    out_len = int(tk.vqvae.vq.levels[-1])

    if UNCOND:
        items = []
        src_tok = None
    else:
        items = pickle.load(open(os.path.join(dd, cfg.data[split]), "rb"))[:samples]
        src_tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None

    def prefix(item):
        ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
        return (ids[:P] + [PAD] * P)[:P]

    def pick_tokens(logits):
        if freq <= 0 and not ADJ:
            return logits.argmax(-1)
        B, T, V = logits.shape
        lg = logits.double()
        allow = (ALLOW[:V] * T)[None, :] if ALLOW is not None else None
        al = ADJ[0] if ADJ else 0.0
        aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1
        ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0
        cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
        out = torch.empty(B, T, dtype=torch.long, device=lg.device)
        hist = []
        for t in range(T):
            step = lg[:, t, :].clone()
            if freq > 0:
                excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
                step -= freq * excess
                if mu > 0 and allow is not None:
                    step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
        return out

    CUT_AT_EOS = bool(G.get("cut_at_eos", str(cfg.eval.metric) != "genppl_flm"))

    def dec(ids):
        ids = list(ids)
        if CUT_AT_EOS and EOS in ids:
            ids = ids[:ids.index(EOS)]
        return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)

    @torch.inference_mode()
    def gpu_generate(x, l0=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, latent = net.generate(
                l0, x, temperature=temp, top_k=top_k, top_p=top_p,
                use_kv_cache=True, cfg_scale=cfg_scale, cfg_null_eos=EOS,
                return_latent=True,
            )
            return pick_tokens(tk.vqvae.decoder(latent))

    def make_request(start: int, size: int):
        if UNCOND:
            l0 = torch.multinomial(L0_P, size * L0, replacement=True).view(size, L0).cuda() if L0_P is not None else None
            x = torch.full((size, 1), EOS, dtype=torch.long, device="cuda")
            return x, l0
        x = torch.tensor([prefix(it) for it in items[start:start + size]], dtype=torch.long, device="cuda")
        return x, None

    if ctx is not None and act_scaling == "static" and mode != "pure_bf16":
        ctx.calibrating = True
        with torch.inference_mode():
            torch.manual_seed(55)
            for s in range(0, min(batch_size * 4, samples), batch_size):
                sz = min(batch_size, samples - s)
                x_c, l0_c = make_request(s, sz)
                gpu_generate(x_c, l0_c)
        ctx.calibrating = False
        finalize_calibration(net)

    runner = CUDAGraphRunner(gpu_generate)
    batches = [(s, min(batch_size, samples - s)) for s in range(0, samples, batch_size)]

    # Warmup every distinct batch shape (including final partial batch if any)
    torch.manual_seed(55)
    seen_sizes = {}
    for start, sz in batches:
        seen_sizes.setdefault(sz, start)
    for sz, start in seen_sizes.items():
        for _ in range(warmup):
            x_w, l0_w = make_request(start, sz)
            ids = runner(x_w, l0_w).tolist()
            _ = [dec(r) for r in ids]
    torch.cuda.synchronize()
    captures_before = len(runner.graphs)

    run_secs = []
    peak_alloc = 0
    peak_res = 0
    for _ in range(repeats):
        torch.manual_seed(55)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        texts = []
        for start, sz in batches:
            x_r, l0_r = make_request(start, sz)
            ids = runner(x_r, l0_r).tolist()
            texts.extend(dec(r) for r in ids)
        torch.cuda.synchronize()
        run_secs.append(time.perf_counter() - t0)
        peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
        peak_res = max(peak_res, torch.cuda.max_memory_reserved())

    captures_after = len(runner.graphs)
    med = statistics.median(run_secs)
    samp_per_s = samples / med
    peak_alloc_mib = peak_alloc / (1024 ** 2)
    peak_alloc_gib = peak_alloc / (1024 ** 3)
    res = {
        "task": task,
        "mode": mode,
        "act_scaling": act_scaling,
        "batch_size": batch_size,
        "samples": samples,
        "median_seconds": round(med, 4),
        "samples_per_second": round(samp_per_s, 2),
        "amortized_batch_ms": round(med / len(batches) * 1000, 2),
        "output_positions_per_second": round(samples * out_len / med, 2),
        "weight_vram_mib": {k: round(v, 2) for k, v in vram_weights.items()},
        "peak_allocated_mib": round(peak_alloc_mib, 2),
        "peak_reserved_mib": round(peak_res / (1024 ** 2), 2),
        "samples_per_sec_per_gib": round(samp_per_s / max(peak_alloc_gib, 1e-9), 2),
        "completed": captures_before == captures_after,
    }
    print(
        f"[BENCH] {task:11s} B={batch_size:2d} | {mode:24s} ({act_scaling:7s}) | "
        f"wt={vram_weights['total_mib']:6.1f} MiB | peak={res['peak_allocated_mib']:6.1f} MiB | "
        f"{med:6.3f}s | {res['samples_per_second']:7.2f} samp/s",
        flush=True,
    )

    del net, tk, runner, sd, ALLOW, L0_P
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sudoku-easy", "sudoku-medium", "sudoku-hard", "tinystories", "wmt14", "xsum"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--auto-batch-sizes", action="store_true")
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-summary", default="outputs/mixed_precision/benchmark/summary_full.json")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = ROOT / args.out_summary
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = [] if args.overwrite or not out_path.exists() else json.loads(out_path.read_text())

    configs = [
        ("official_bf16", "dynamic"),
        ("pure_bf16", "dynamic"),
        ("coarse_bf16_fine_fp8", "static"),
        ("coarse_bf16_fine_fp8_w8", "static"),
        ("coarse_fp8_fine_bf16", "static"),
        ("fp8_all", "static"),
    ]

    for t in args.tasks:
        batch_list = TASK_BATCH_SIZES[t] if args.auto_batch_sizes else args.batch_sizes
        for bs in batch_list:
            for mode, act_s in configs:
                try:
                    r = run_benchmark(
                        t, mode, batch_size=bs, samples=args.samples,
                        warmup=args.warmup, repeats=args.repeats, act_scaling=act_s,
                    )
                    summary.append(r)
                    out_path.write_text(json.dumps(summary, indent=2) + "\n")
                except torch.OutOfMemoryError as e:
                    print(f"[OOM] {t} B={bs} {mode}: {e}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
