"""Unit tests for fp8_layers.py: inference CUDAGraphs, tied heads, and training equivalence."""

import os
import sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "mixed_precision"))

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from utils.fast_generation import CUDAGraphRunner, enable_resize_matmuls
from fp8_layers import (
    apply_inference_precision,
    finalize_calibration,
    measure_module_vram_mib,
    patch_model_for_train_precision,
    scale_adaptive_train_forward,
)


def load_task(task: str, ckpt_dir: str):
    cfg = OmegaConf.load(f"config/eval/{task}.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ckpt_dir, cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
    sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    return cfg, tk, net


def test_inference():
    import pickle
    from transformers import AutoTokenizer
    print("=== Test 1: Inference + CUDAGraphs on sudoku-hard & xsum ===")
    for task, ckpt_dir, data_path in [
        ("sudoku-hard", "artifacts/sudoku/hard", "data/sudoku/hard/puzzles.pickle"),
        ("xsum", "artifacts/xsum", "data/xsum/passages.pickle"),
    ]:
        items = pickle.load(open(data_path, "rb"))[:8]
        for mode in ["pure_bf16", "coarse_bf16_fine_fp8", "coarse_bf16_fine_fp8_w8", "fp8_all"]:
            cfg, tk, net = load_task(task, ckpt_dir)
            enable_resize_matmuls(tk.vqvae.vq)
            ctx = apply_inference_precision(net, tk, mode=mode, act_scaling="static")
            vram = measure_module_vram_mib(net, tk)
            P = int(cfg.model.prefix_len)
            PAD = tk.pre_tokenizer.pad_token_id
            src_tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None
            x = torch.tensor(
                [((src_tok(it)["input_ids"] if src_tok is not None else list(it))[:P] + [PAD] * P)[:P] for it in items],
                device="cuda", dtype=torch.long,
            )
            B = x.shape[0]

            @torch.inference_mode()
            def gpu_gen(inp, l0=None):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, latent = net.generate(
                        l0, inp,
                        temperature=1.0, top_k=1, top_p=0.0,
                        use_kv_cache=True, cfg_scale=0.0,
                        cfg_null_eos=tk.pre_tokenizer.eos_token_id,
                        return_latent=True,
                    )
                    return tk.vqvae.decoder(latent).argmax(-1)

            if ctx is not None and mode != "pure_bf16":
                ctx.calibrating = True
                gpu_gen(x)
                ctx.calibrating = False
                finalize_calibration(net)

            runner = CUDAGraphRunner(gpu_gen)
            out1 = runner(x)
            out2 = runner(x)
            assert out1.shape == (B, tk.vqvae.context_length), out1.shape
            assert torch.equal(out1, out2), "Greedy CUDAGraph replay mismatch!"
            print(f"  PASS {task} [{mode}]: shape={tuple(out1.shape)} vram={vram['total_mib']:.1f} MiB (gen={vram['generator_mib']:.1f}, vq={vram['vqvae_mib']:.1f})")
            del net, tk, runner
            torch.cuda.empty_cache()


def test_training_equivalence():
    print("\n=== Test 2: Split-precision training forward/backward equivalence (sudoku-hard) ===")
    cfg, tk, net_ref = load_task("sudoku-hard", "artifacts/sudoku/hard")
    _, _, net_test = load_task("sudoku-hard", "artifacts/sudoku/hard")
    net_ref.eval()
    net_test.eval()

    B = 4
    P = int(cfg.model.prefix_len)
    L_tgt = sum(net_ref.levels[net_ref._lvl_first:])
    x = torch.randn(B, P + L_tgt, tk.vqvae.embed_dim, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits_ref = net_ref(x)
        logits_split = scale_adaptive_train_forward(net_test, x, mode="bf16")

    diff = (logits_ref - logits_split).abs().max().item()
    print(f"  BF16 split forward max_abs_diff vs NextConceptTransformer.forward (eval): {diff:.6e}")
    assert diff < 1e-1, f"Forward diff too large: {diff}"

    net_ref.train()
    net_test.train()

    loss_ref = logits_ref.float().pow(2).mean()
    loss_split = logits_split.float().pow(2).mean()
    loss_ref.backward()
    loss_split.backward()

    g_diffs = []
    for (n1, p1), (n2, p2) in zip(net_ref.named_parameters(), net_test.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            g_diffs.append((p1.grad - p2.grad).abs().max().item())
    max_g_diff = max(g_diffs)
    print(f"  BF16 split backward max_grad_diff: {max_g_diff:.6e}")
    assert max_g_diff < 1e-2, f"Grad diff too large: {max_g_diff}"

    # Now test FP8 split forward + backward finite grads
    net_fp8 = net_test
    net_fp8.zero_grad(set_to_none=True)
    patch_model_for_train_precision(net_fp8, mode="coarse_bf16_fine_fp8")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits_fp8 = scale_adaptive_train_forward(net_fp8, x, mode="coarse_bf16_fine_fp8")
    loss_fp8 = logits_fp8.float().pow(2).mean()
    loss_fp8.backward()
    fp8_diff = (logits_ref.detach() - logits_fp8.detach()).abs().mean().item()
    nonfinite = [n for n, p in net_fp8.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"  coarse_bf16_fine_fp8 train forward mean_abs_diff vs BF16: {fp8_diff:.6f}, nonfinite grads: {len(nonfinite)}")
    assert not nonfinite, f"Nonfinite grads: {nonfinite}"
    print("  ALL UNIT TESTS PASSED!")


if __name__ == "__main__":
    test_inference()
    test_training_equivalence()
