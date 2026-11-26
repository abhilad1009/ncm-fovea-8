"""Training convergence, QAT recovery, and speed/VRAM scaling study for mixed-precision NCM."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.vqvae_tokenizer import VQVAETokenizer
from fp8_layers import (
    apply_inference_precision,
    patch_model_for_train_precision,
    scale_adaptive_train_forward,
)


def load_sudoku_hard_data():
    cfg = OmegaConf.load(ROOT / "config/eval/sudoku-hard.yaml")
    ck = ROOT / "artifacts/sudoku/hard"
    cfg.tokenizer.vqvae.checkpoint_path = str(ck / cfg.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    for p in tk.vqvae.parameters():
        p.requires_grad_(False)

    puzzles = pickle.load(open(ROOT / "data/sudoku/hard/puzzles.pickle", "rb"))
    solutions = pickle.load(open(ROOT / "data/sudoku/hard/solutions.pickle", "rb"))
    P = int(cfg.model.prefix_len)
    PAD = tk.pre_tokenizer.pad_token_id
    puz_t = torch.tensor([(list(it)[:P] + [PAD] * P)[:P] for it in puzzles], dtype=torch.long)
    sol_t = torch.tensor([list(it) for it in solutions], dtype=torch.long)

    # Pre-encode solutions with the frozen VQ-VAE tokenizer (standard NCM stage-2 dataset prep).
    sol_idx_list = []
    with torch.no_grad():
        for i in range(0, len(sol_t), 256):
            sol_idx_list.append(tk.vqvae.encode(sol_t[i:i + 256].cuda()).cpu())
    sol_idx = torch.cat(sol_idx_list, dim=0)
    return cfg, tk, puzzles, solutions, puz_t, sol_idx


def build_sudoku_batch(net, tk, puz_batch: torch.Tensor, idx_batch: torch.Tensor):
    with torch.no_grad():
        z_prefix = tk.vqvae.encoder.wte(puz_batch).detach()
        l0_q = net.build_l0_query_input(puz_batch.shape[0], dtype=z_prefix.dtype)
        rest_in = tk.vqvae.vq.idx_to_ncm_input(idx_batch).detach()
    x = torch.cat([z_prefix, l0_q, rest_in], dim=1)
    return x, idx_batch


def compute_split_loss(logits: torch.Tensor, targets: torch.Tensor, levels: list[int], split_level: int):
    # Compute loss in FP32, matching standard mixed-precision CE.
    B, T, V = logits.shape
    per_tok = F.cross_entropy(logits.float().reshape(-1, V), targets.reshape(-1), reduction="none").view(B, T)
    coarse_len = sum(levels[:split_level])
    total = per_tok.mean()
    coarse = per_tok[:, :coarse_len].mean()
    fine = per_tok[:, coarse_len:].mean()
    return total, coarse, fine


def eval_sudoku_exact(net, tk, puzzles_sub, refs_sub, P: int, bs: int = 100) -> dict:
    PAD = tk.pre_tokenizer.pad_token_id
    EOS = tk.pre_tokenizer.eos_token_id
    net.eval()

    def dec(ids):
        ids = list(ids)
        if EOS in ids:
            ids = ids[:ids.index(EOS)]
        return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)

    hyps = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(puzzles_sub), bs):
            batch = puzzles_sub[s:s + bs]
            x = torch.tensor([(list(it)[:P] + [PAD] * P)[:P] for it in batch], device="cuda", dtype=torch.long)
            _, latent = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, use_kv_cache=True, cfg_scale=0.0, return_latent=True)
            ids = tk.vqvae.decoder(latent).argmax(-1).tolist()
            hyps.extend(dec(r) for r in ids)

    refs = [dec(r) for r in refs_sub]
    n = len(hyps)
    solved = sum(h == r for h, r in zip(hyps, refs))
    cell = sum(sum(a == b for a, b in zip(h.replace("|", ""), r.replace("|", ""))) / max(len(r.replace("|", "")), 1) for h, r in zip(hyps, refs)) / n
    return {"solved_pct": round(100.0 * solved / n, 2), "solved": solved, "total": n, "cell_pct": round(100.0 * cell, 2)}


def run_exp_5a(cfg, tk, puz_t, sol_idx, steps: int = 400, bs: int = 32) -> dict:
    print("\n=== Experiment 5A: From-Scratch Training Convergence (sudoku-hard, 400 steps) ===", flush=True)
    train_puz, val_puz = puz_t[:1800], puz_t[1800:]
    train_idx, val_idx = sol_idx[:1800], sol_idx[1800:]
    levels = list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)
    split_level = len(levels) // 2

    results = {}
    for mode in ["fp32", "bf16", "coarse_bf16_fine_fp8", "fp8_all"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
        patch_model_for_train_precision(net, mode=mode, split_level=split_level)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=3e-5)

        use_amp = mode != "fp32"
        train_losses = []
        max_gnorm = 0.0
        nan_steps = 0

        # Warmup 5 steps outside timer
        t_start = None
        for step in range(steps):
            if step == 5:
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            perm = torch.randint(0, len(train_puz), (bs,))
            xb, yb = build_sudoku_batch(net, tk, train_puz[perm].cuda(), train_idx[perm].cuda())
            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if mode in ("fp32", "bf16"):
                    logits = net(xb)
                else:
                    logits = scale_adaptive_train_forward(net, xb, mode=mode, split_level=split_level)
                loss, _, _ = compute_split_loss(logits, yb, levels, split_level)

            if not torch.isfinite(loss):
                nan_steps += 1
                continue

            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0).item())
            if math.isfinite(gn):
                max_gnorm = max(max_gnorm, gn)
            else:
                nan_steps += 1
            opt.step()
            sched.step()
            train_losses.append(float(loss.item()))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        ms_per_step = elapsed / max(steps - 5, 1) * 1000
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Evaluate val loss (split coarse vs fine)
        net.eval()
        val_tot, val_c, val_f, n_b = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_puz), bs):
                xb, yb = build_sudoku_batch(net, tk, val_puz[i:i + bs].cuda(), val_idx[i:i + bs].cuda())
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    if mode in ("fp32", "bf16"):
                        logits = net(xb)
                    else:
                        logits = scale_adaptive_train_forward(net, xb, mode=mode, split_level=split_level)
                    lt, lc, lf = compute_split_loss(logits, yb, levels, split_level)
                val_tot += float(lt.item())
                val_c += float(lc.item())
                val_f += float(lf.item())
                n_b += 1

        res = {
            "mode": mode,
            "steps": steps,
            "final_train_loss": round(sum(train_losses[-20:]) / max(len(train_losses[-20:]), 1), 4),
            "val_loss": round(val_tot / n_b, 4),
            "val_coarse_loss": round(val_c / n_b, 4),
            "val_fine_loss": round(val_f / n_b, 4),
            "max_grad_norm": round(max_gnorm, 3),
            "nan_steps": nan_steps,
            "ms_per_step": round(ms_per_step, 2),
            "peak_vram_mib": round(peak_mib, 2),
        }
        results[mode] = res
        print(f"  [5A] {mode:22s} | val={res['val_loss']:.4f} (coarse={res['val_coarse_loss']:.4f}, fine={res['val_fine_loss']:.4f}) | ms/step={res['ms_per_step']:.2f} | peak={res['peak_vram_mib']:.1f} MiB", flush=True)
        del net, opt, sched
    return results


def run_exp_5b(cfg, tk, puzzles, solutions, puz_t, sol_idx, steps: int = 100, bs: int = 32) -> dict:
    print("\n=== Experiment 5B: Pretrained sudoku-hard Short QAT Recovery (100 steps) ===", flush=True)
    train_puz, val_puz = puz_t[:1800], puzzles[1800:]
    train_idx, val_refs = sol_idx[:1800], solutions[1800:]
    levels = list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)
    split_level = len(levels) // 2
    P = int(cfg.model.prefix_len)

    results = {}
    for mode in ["coarse_bf16_fine_fp8", "fp8_all"]:
        torch.manual_seed(42)
        gc.collect()
        torch.cuda.empty_cache()

        # Load fresh pretrained model for PTQ baseline vs QAT.
        net_ptq = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
        sd = torch.load(ROOT / "artifacts/sudoku/hard/generator.pt", map_location="cpu", weights_only=False)
        net_ptq.load_state_dict(sd["model"], strict=True)
        net_ptq.eval()
        apply_inference_precision(net_ptq, tk, mode=mode, act_scaling="dynamic")
        ptq_val = eval_sudoku_exact(net_ptq, tk, val_puz, val_refs, P=P)
        del net_ptq

        # Reload fresh tokenizer embeddings if fp8_all quantized tk.vqvae.vq.embeddings
        cfg_fresh, tk_fresh, _, _, _, _ = load_sudoku_hard_data() if mode == "fp8_all" else (cfg, tk, None, None, None, None)

        net = NextConceptTransformer(vqvae_local=tk_fresh.vqvae, config=cfg.model).cuda()
        net.load_state_dict(sd["model"], strict=True)
        patch_model_for_train_precision(net, mode=mode, split_level=split_level)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-5, weight_decay=0.01)

        for _ in range(steps):
            perm = torch.randint(0, len(train_puz), (bs,))
            xb, yb = build_sudoku_batch(net, tk_fresh, train_puz[perm].cuda(), train_idx[perm].cuda())
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = scale_adaptive_train_forward(net, xb, mode=mode, split_level=split_level)
                loss, _, _ = compute_split_loss(logits, yb, levels, split_level)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        # Convert trained weights to inference FP8 and evaluate on the same 200 val boards.
        net.eval()
        # Restore plain nn.Linear weights before apply_inference_precision
        for blk in net.transformer.blocks:
            for name, mod in list(blk.named_modules()):
                if hasattr(mod, "weight") and mod.__class__.__name__ == "ScaleAdaptiveTrainLinear":
                    parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = blk.get_submodule(parent_name) if parent_name else blk
                    lin = torch.nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None, device="cuda", dtype=torch.float32)
                    lin.weight.data.copy_(mod.weight.data)
                    if mod.bias is not None:
                        lin.bias.data.copy_(mod.bias.data)
                    setattr(parent, attr, lin)

        apply_inference_precision(net, tk_fresh, mode=mode, act_scaling="dynamic")
        qat_val = eval_sudoku_exact(net, tk_fresh, val_puz, val_refs, P=P)
        results[mode] = {"before_qat_ptq": ptq_val, "after_qat_100steps": qat_val}
        print(f"  [5B] {mode:22s} | before={ptq_val['solved_pct']}% ({ptq_val['solved']}/{ptq_val['total']}) -> after 100 steps={qat_val['solved_pct']}% ({qat_val['solved']}/{qat_val['total']})", flush=True)
        del net, opt
        if mode == "fp8_all":
            del tk_fresh
    return results


def run_exp_5c() -> list[dict]:
    print("\n=== Experiment 5C: Training Speed & Peak Memory Scaling on Larger NCM Models ===", flush=True)
    specs = [
        ("tinystories", "config/eval/tinystories.yaml", 16),
        ("tinystories", "config/eval/tinystories.yaml", 32),
        ("xsum", "config/eval/xsum.yaml", 16),
        ("lm1b-medium", "config/eval/lm1b-medium.yaml", 16),
    ]
    all_rows = []
    for task, cfg_rel, bs in specs:
        cfg = OmegaConf.load(ROOT / cfg_rel)
        # Instantiate VQVAE architecture on CPU->CUDA (no checkpoint needed for speed/VRAM bench)
        vqvae = VQVAE(**cfg.tokenizer.vqvae).cuda().eval()
        levels = list(vqvae.vq.levels)
        split_level = len(levels) // 2
        P = int(cfg.model.prefix_len)
        T = sum(levels[1:]) if not bool(cfg.model.get("query_level0", False)) else sum(levels)
        D = int(cfg.tokenizer.vqvae.embed_dim)
        V = int(cfg.model.vocab_size[0])

        for mode in ["fp32", "bf16", "coarse_bf16_fine_fp8", "fp8_all"]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model).cuda()
                patch_model_for_train_precision(net, mode=mode, split_level=split_level)
                net.train()
                opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
                use_amp = mode != "fp32"

                xb = torch.randn(bs, P + T, D, device="cuda")
                yb = torch.randint(0, V, (bs, T), device="cuda")

                def step_fn():
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        if mode in ("fp32", "bf16"):
                            logits = net(xb)
                        else:
                            logits = scale_adaptive_train_forward(net, xb, mode=mode, split_level=split_level)
                        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), yb.reshape(-1))
                    loss.backward()
                    opt.step()

                for _ in range(5):
                    step_fn()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                iters = 15
                for _ in range(iters):
                    step_fn()
                torch.cuda.synchronize()
                ms_step = (time.perf_counter() - t0) / iters * 1000
                peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
                row = {
                    "task": task,
                    "batch_size": bs,
                    "mode": mode,
                    "ms_per_step": round(ms_step, 2),
                    "samples_per_sec": round(bs / (ms_step / 1000), 2),
                    "peak_vram_mib": round(peak_mib, 2),
                    "oom": False,
                }
                print(f"  [5C] {task:11s} B={bs:2d} | {mode:22s} | {row['ms_per_step']:7.2f} ms/step | {row['samples_per_sec']:6.2f} samp/s | peak={row['peak_vram_mib']:7.1f} MiB", flush=True)
                all_rows.append(row)
                del net, opt
            except torch.OutOfMemoryError:
                print(f"  [5C] {task:11s} B={bs:2d} | {mode:22s} | OOM", flush=True)
                all_rows.append({"task": task, "batch_size": bs, "mode": mode, "oom": True})
        del vqvae
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-5a", action="store_true")
    ap.add_argument("--skip-5b", action="store_true")
    ap.add_argument("--skip-5c", action="store_true")
    args = ap.parse_args()

    out_path = ROOT / "outputs" / "mixed_precision" / "train" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(out_path.read_text()) if out_path.exists() else {}

    if not args.skip_5a or not args.skip_5b:
        cfg, tk, puzzles, solutions, puz_t, sol_idx = load_sudoku_hard_data()
        if not args.skip_5a:
            summary["exp_5a_from_scratch"] = run_exp_5a(cfg, tk, puz_t, sol_idx)
            out_path.write_text(json.dumps(summary, indent=2) + "\n")
        if not args.skip_5b:
            summary["exp_5b_qat_recovery"] = run_exp_5b(cfg, tk, puzzles, solutions, puz_t, sol_idx)
            out_path.write_text(json.dumps(summary, indent=2) + "\n")
        del tk, puz_t, sol_idx
        gc.collect()
        torch.cuda.empty_cache()

    if not args.skip_5c:
        summary["exp_5c_scaling_bench"] = run_exp_5c()
        out_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
