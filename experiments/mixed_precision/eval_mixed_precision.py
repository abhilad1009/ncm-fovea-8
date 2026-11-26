"""Evaluate NCM quality across mixed-precision regimes using the exact official eval pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import subprocess
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


def parse_eval_merge_output(metric: str, text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if metric == "exact":
        for ln in reversed(lines):
            m = re.match(r"^([\d.]+)\s+(\d+)/(\d+)\s+([\d.]+)\s+([\d.]+)$", ln)
            if m:
                return {
                    "solved_pct": float(m.group(1)),
                    "solved": int(m.group(2)),
                    "total": int(m.group(3)),
                    "cell_pct": float(m.group(4)),
                    "valid_pct": float(m.group(5)),
                }
    elif metric == "genppl":
        for ln in reversed(lines):
            m = re.match(r"^mean\s+([\d.]+)\s+([\d.]+)\s+\(\+/-\s+([\d.]+)\s+over\s+(\d+)\s+seeds,\s+(\d+)\s+samples\)$", ln)
            if m:
                return {
                    "mean_genppl": float(m.group(1)),
                    "mean_entropy": float(m.group(2)),
                    "std_genppl": float(m.group(3)),
                    "n_seeds": int(m.group(4)),
                    "n_samples": int(m.group(5)),
                }
    elif metric == "bleu":
        for ln in reversed(lines):
            m = re.match(r"^([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", ln)
            if m:
                return {
                    "bleu": float(m.group(1)),
                    "bleu_lc": float(m.group(2)),
                    "chrf": float(m.group(3)),
                    "len_ratio": float(m.group(4)),
                }
    elif metric == "rouge":
        for ln in reversed(lines):
            m = re.match(r"^([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", ln)
            if m:
                return {
                    "rouge1": float(m.group(1)),
                    "rouge2": float(m.group(2)),
                    "rougeL": float(m.group(3)),
                    "genppl": float(m.group(4)),
                    "avg_words": float(m.group(5)),
                }
    return {"raw": text}


def run_single_eval(task: str, mode: str, act_scaling: str = "dynamic", out_root: str = "outputs/mixed_precision/eval_full") -> dict:
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
    n_docs = int(cfg.eval.get("n_docs", 0))
    bs = int(cfg.eval.batch_size)

    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ck, cfg.checkpoints[name])

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
    sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    enable_resize_matmuls(tk.vqvae.vq)
    ctx = apply_inference_precision(net, tk, mode=mode, act_scaling=act_scaling)
    vram_stats = measure_module_vram_mib(net, tk)

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
    L0_P = load_allowance(str(G.level0_unigram)).float() if G.get("level0_unigram") else None
    UNCOND = not cfg.data or not cfg.data.get(split)
    ADJ = [float(v) for v in adj.split(",")] if adj else None

    if UNCOND:
        seeds = [int(x) for x in cfg.eval.seeds]
        n = int(cfg.eval.n_samples)
        items, refs, idxs, src_tok = [], [], [], None
    else:
        items = pickle.load(open(os.path.join(dd, cfg.data[split]), "rb"))[:n_docs]
        refs = pickle.load(open(os.path.join(dd, cfg.data[f"{split}_refs"]), "rb"))[:n_docs]
        idxs = list(range(len(items)))
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

    if ctx is not None and act_scaling == "static" and mode != "pure_bf16":
        ctx.calibrating = True
        with torch.inference_mode():
            if UNCOND:
                torch.manual_seed(seeds[0])
                L0 = tk.vqvae.vq.levels[0]
                for s in range(0, min(bs * 4, n), bs):
                    B = min(bs, n - s)
                    l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).cuda() if L0_P is not None else None
                    gpu_generate(torch.full((B, 1), EOS, device="cuda"), l0)
            else:
                for s in range(0, min(bs * 4, len(idxs)), bs):
                    batch = idxs[s:s + bs]
                    gpu_generate(torch.tensor([prefix(items[i]) for i in batch], device="cuda"))
        ctx.calibrating = False
        finalize_calibration(net)

    generate_tokens = CUDAGraphRunner(gpu_generate)
    RAW = []

    def run(x, l0=None):
        ids = generate_tokens(x, l0).tolist()
        RAW.extend(ids)
        return [dec(r) for r in ids]

    t0 = time.perf_counter()
    hyps, sample_seed = [], []
    if UNCOND:
        L0 = tk.vqvae.vq.levels[0]
        for seed in seeds:
            torch.manual_seed(seed)
            for s in range(0, n, bs):
                B = min(bs, n - s)
                l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).cuda() if L0_P is not None else None
                hyps += run(torch.full((B, 1), EOS, device="cuda"), l0)
                sample_seed += [seed] * B
        idxs = list(range(len(hyps)))
        toks = [tk.pre_tokenizer.encode(h) for h in hyps]
        out = {"idxs": idxs, "hyps": hyps, "refs": [""] * len(hyps), "seeds": sample_seed, "toks": toks, "raw_ids": RAW}
    else:
        for s in range(0, len(idxs), bs):
            batch = idxs[s:s + bs]
            hyps += run(torch.tensor([prefix(items[i]) for i in batch], device="cuda"))
        out = {"idxs": idxs, "hyps": hyps, "refs": [r if isinstance(r, str) else dec(r) for r in (refs[i] for i in idxs)]}
    gen_seconds = time.perf_counter() - t0

    out_dir = ROOT / out_root / mode / task
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / "shard_0.pkl"
    with shard_path.open("wb") as f:
        pickle.dump(out, f)

    # Free generator VRAM before eval_merge loads scorer (e.g. gpt2-large).
    del net, tk, generate_tokens, sd, ALLOW, L0_P
    gc.collect()
    torch.cuda.empty_cache()

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_merge.py"), "--config", str(ROOT / cfg_rel), "--glob", str(shard_path)],
        cwd=str(ROOT), capture_output=True, text=True, check=True,
    )
    metrics = parse_eval_merge_output(str(cfg.eval.metric), proc.stdout)
    result = {
        "task": task,
        "mode": mode,
        "act_scaling": act_scaling,
        "gen_seconds": round(gen_seconds, 2),
        "vram_weights_mib": {k: round(v, 2) for k, v in vram_stats.items()},
        "metrics": metrics,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[EVAL] {task:13s} | {mode:24s} | vram={vram_stats['total_mib']:6.1f} MiB | {gen_seconds:5.1f}s | {metrics}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sudoku-easy", "sudoku-medium", "sudoku-hard", "tinystories", "wmt14", "xsum"])
    ap.add_argument("--modes", nargs="+", default=["official_bf16", "pure_bf16", "coarse_bf16_fine_fp8", "coarse_bf16_fine_fp8_w8", "coarse_fp8_fine_bf16", "fp8_all"])
    ap.add_argument("--include-w8-subset", action="store_true")
    ap.add_argument("--act-scaling", choices=["dynamic", "static"], default="static")
    ap.add_argument("--out-root", default="outputs/mixed_precision/eval_full")
    ap.add_argument("--out-summary", default="outputs/mixed_precision/eval_full/summary_full.json")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    summary_path = ROOT / args.out_summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results = {} if args.overwrite or not summary_path.exists() else json.loads(summary_path.read_text())

    runs = [(t, m) for t in args.tasks for m in args.modes]
    if args.include_w8_subset:
        for t in ("sudoku-hard", "tinystories", "wmt14"):
            if t in args.tasks and (t, "coarse_bf16_fine_fp8_w8") not in runs:
                runs.append((t, "coarse_bf16_fine_fp8_w8"))

    for t, m in runs:
        res = run_single_eval(t, m, act_scaling=args.act_scaling, out_root=args.out_root)
        results.setdefault(t, {})[m] = res
        summary_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
