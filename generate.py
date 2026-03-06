"""Core generation loop for autoloop experiments.

Runs a single closed-loop autoregressive generation session: pre-fills a context
window of length L at temperature T, then generates N tokens with a sliding window,
logging per-step data to Parquet with a JSON metadata sidecar.

Checkpoints every 1k steps for resume and run extension.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from utils import compressibility

log = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoloop generation run")
    parser.add_argument("--context-length", type=int, required=True, help="Context window size L")
    parser.add_argument("--temperature", type=float, required=True, help="Sampling temperature T")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--num-tokens", type=int, required=True, help="Number of post-pre-fill tokens N")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to local model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--device", type=str, required=True, help="Torch device (e.g. cuda, cpu)")
    return parser.parse_args()


def compute_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy of the softmax distribution (in nats)."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs).item()
    return entropy


def save_checkpoint(
    checkpoint_path: Path,
    parquet_path: Path,
    step: int,
    context: torch.Tensor,
    records: list[dict],
) -> None:
    torch.save({
        "step": step,
        "context": context.cpu(),
        "records": records,
        "rng_torch": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
    }, checkpoint_path)
    pd.DataFrame(records).to_parquet(parquet_path, index=False)


def load_checkpoint(checkpoint_path: Path, device: str) -> dict:
    ckpt = torch.load(checkpoint_path, weights_only=False)
    torch.random.set_rng_state(ckpt["rng_torch"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"])
    np.random.set_state(ckpt["rng_numpy"])
    ckpt["context"] = ckpt["context"].to(device)
    return ckpt


def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_length: int,
    temperature: float,
    seed: int,
    num_tokens: int,
    device: str,
    checkpoint_path: Path,
    parquet_path: Path,
) -> list[dict]:
    """Run the full generation loop (pre-fill + experiment) and return per-step records."""
    total_steps = context_length + num_tokens
    eos_token_id = tokenizer.eos_token_id

    # Resume from checkpoint or start fresh
    if checkpoint_path.exists():
        ckpt = load_checkpoint(checkpoint_path, device)
        start_step = ckpt["step"] + 1
        context = ckpt["context"]
        records = ckpt["records"]
        log.info("Resumed from checkpoint at step %d", ckpt["step"])
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        bos_token_id = tokenizer.bos_token_id
        context = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        records = []
        start_step = 1

    interval_start = time.monotonic()

    for t in range(start_step, total_steps + 1):
        is_prefill = t <= context_length
        phase = "prefill" if is_prefill else "experiment"

        with torch.no_grad():
            outputs = model(input_ids=context)
        logits = outputs.logits[0, -1, :]

        entropy = compute_entropy(logits)

        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()

        log_prob = torch.log_softmax(scaled_logits, dim=-1)[token_id].item()

        decoded_text = tokenizer.decode([token_id])
        is_eos = token_id == eos_token_id

        records.append({
            "step": t,
            "phase": phase,
            "token_id": token_id,
            "decoded_text": decoded_text,
            "entropy": entropy,
            "log_prob": log_prob,
            "temperature": temperature,
            "eos": is_eos,
        })

        new_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        context = torch.cat([context, new_token], dim=1)
        if context.shape[1] > context_length:
            context = context[:, -context_length:]

        if t % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(checkpoint_path, parquet_path, t, context, records)
            now = time.monotonic()
            tok_s = CHECKPOINT_INTERVAL / (now - interval_start)
            # Recent stats over trailing context_length tokens
            recent = records[-context_length:]
            ent_vals = [r["entropy"] for r in recent]
            ent_mean = sum(ent_vals) / len(ent_vals)
            # Trailing-window compressibility (W=L)
            trail_text = "".join(r["decoded_text"] for r in recent)
            comp = compressibility(trail_text.encode("utf-8"))
            log.info(
                "step %d/%d (%.0f%%) | %.1f tok/s | ent=%.2f comp=%.3f",
                t, total_steps, 100 * t / total_steps, tok_s, ent_mean, comp,
            )
            interval_start = now

        log.debug(
            "step=%d phase=%s token=%d text=%r entropy=%.4f log_prob=%.4f eos=%s",
            t, phase, token_id, decoded_text, entropy, log_prob, is_eos,
        )

    # Final checkpoint
    save_checkpoint(checkpoint_path, parquet_path, total_steps, context, records)

    return records


def build_run_name(context_length: int, temperature: float, seed: int) -> str:
    return f"L{context_length:04d}_T{temperature:.2f}_S{seed}"


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_name = build_run_name(args.context_length, args.temperature, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{run_name}.parquet"
    meta_path = output_dir / f"{run_name}.meta.json"
    checkpoint_path = output_dir / f"{run_name}.ckpt"

    log.info("Starting run: %s", run_name)
    log.info(
        "L=%d T=%.2f seed=%d N=%d device=%s",
        args.context_length, args.temperature, args.seed, args.num_tokens, args.device,
    )

    log.info("Loading model from %s", args.model_dir)
    disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32,
    ).to(args.device)
    model.eval()
    log.info("Model loaded: %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")

    t_start = time.monotonic()
    records = run_generation(
        model=model,
        tokenizer=tokenizer,
        context_length=args.context_length,
        temperature=args.temperature,
        seed=args.seed,
        num_tokens=args.num_tokens,
        device=args.device,
        checkpoint_path=checkpoint_path,
        parquet_path=parquet_path,
    )
    elapsed = time.monotonic() - t_start

    metadata = {
        "run_name": run_name,
        "context_length": args.context_length,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_tokens": args.num_tokens,
        "device": args.device,
        "model_dir": args.model_dir,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round((args.context_length + args.num_tokens) / elapsed, 1),
        "total_steps": args.context_length + args.num_tokens,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Wrote metadata to %s", meta_path)

    log.info(
        "Done: %d steps in %.1fs (%.1f tok/s)",
        metadata["total_steps"], elapsed, metadata["tokens_per_second"],
    )


if __name__ == "__main__":
    main()
