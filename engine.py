"""Token generation engine with sensor feedback.

StepEngine wraps a language model and provides:
- step(L, T): generate one token
- run_segment(L, T, n_steps): generate a segment of tokens
- read_sensors(): compute trailing-window metrics (entropy, β, comp)
- snapshot()/restore(): state capture for rollback
- save/load: checkpoint persistence

Used by experiment.py to run any kind of experiment.
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from utils import compressibility, fix_decoded_texts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SensorReading:
    """Sensor snapshot after a segment."""
    step: int
    L: int
    T: float
    entropy_mean: float
    entropy_std: float
    comp_W64: float
    heaps_beta: float
    n_words: int
    n_unique: int


@dataclasses.dataclass
class Snapshot:
    """Engine state capture for rollback."""
    n_records: int
    step: int
    context: torch.Tensor
    rng_torch: torch.Tensor
    rng_cuda: torch.Tensor | None
    rng_numpy: dict


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str, device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer, log parameter count."""
    log.info("Loading model from %s", model_dir)
    disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: %s parameters", f"{n_params:,}")
    return model, tokenizer


def compute_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy of the softmax distribution (in nats)."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs).item()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StepEngine:
    """Token generation engine with sensor feedback and rollback.

    Owns the model, tokenizer, context window, and record list.
    All generation goes through step() — one token at a time.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
        seed: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed
        self.eos_token_id = tokenizer.eos_token_id

        # Mutable state
        self.records: list[dict] = []
        self.context = torch.tensor(
            [[tokenizer.bos_token_id]], dtype=torch.long, device=device,
        )
        self._step = 0

        # Seed RNGs
        torch.manual_seed(seed)
        np.random.seed(seed)

    @property
    def current_step(self) -> int:
        return self._step

    # -- Generation ---------------------------------------------------------

    def step(self, L: int, T: float, phase: str = "experiment") -> dict:
        """Generate one token at given L/T. Returns record dict."""
        self._step += 1

        with torch.no_grad():
            outputs = self.model(input_ids=self.context)
        logits = outputs.logits[0, -1, :]

        entropy = compute_entropy(logits)
        scaled_logits = logits / T
        probs = torch.softmax(scaled_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        log_prob = torch.log_softmax(scaled_logits, dim=-1)[token_id].item()
        decoded_text = self.tokenizer.decode([token_id])
        is_eos = token_id == self.eos_token_id

        record = {
            "step": self._step,
            "phase": phase,
            "token_id": token_id,
            "decoded_text": decoded_text,
            "entropy": entropy,
            "log_prob": log_prob,
            "temperature": T,
            "context_length": L,
            "eos": is_eos,
        }
        self.records.append(record)

        new_token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        self.context = torch.cat([self.context, new_token], dim=1)
        if self.context.shape[1] > L:
            self.context = self.context[:, -L:]

        return record

    def run_segment(
        self, L: int, T: float, n_steps: int, phase: str = "experiment",
    ) -> None:
        """Generate n_steps tokens at fixed L/T."""
        for _ in range(n_steps):
            self.step(L, T, phase)

    def set_context(self, token_ids: list[int]) -> None:
        """Set context directly from token IDs (for prefill-text)."""
        self.context = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device,
        )

    # -- Sensors ------------------------------------------------------------

    def read_sensors(
        self, window: int = 0, segment_steps: int = 0,
    ) -> SensorReading:
        """Compute sensor values from trailing window of records.

        Default window: max(5× segment_steps, 500).
        """
        if window == 0:
            window = max(5 * segment_steps, 500) if segment_steps > 0 else 2000
        tail = self.records[-window:] if len(self.records) > window else self.records
        exp_tail = [r for r in tail if r["phase"] == "experiment"]
        if not exp_tail:
            exp_tail = tail

        # Entropy
        ent = [r["entropy"] for r in exp_tail]
        ent_mean = sum(ent) / len(ent)
        ent_std = (sum((e - ent_mean) ** 2 for e in ent) / len(ent)) ** 0.5

        # Compressibility (W=64 from trailing text)
        texts = [r["decoded_text"] for r in exp_tail]
        chunk = "".join(texts[-64:]) if len(texts) >= 64 else "".join(texts)
        comp_w64 = compressibility(chunk.encode("utf-8")) if len(chunk) > 10 else 0.0

        # Heaps' β from trailing window
        text = "".join(texts)
        words = [w.lower() for w in text.split() if len(w) > 1]
        n_words = len(words)

        beta = 0.0
        n_unique = 0
        if n_words >= 50:
            seen: set[str] = set()
            checkpoints = 20
            step_size = max(1, n_words // checkpoints)
            ns, vs = [], []
            for i, w in enumerate(words):
                seen.add(w)
                if (i + 1) % step_size == 0:
                    ns.append(i + 1)
                    vs.append(len(seen))
            n_unique = len(seen)

            if len(ns) >= 3:
                log_n = np.log(np.array(ns, dtype=float))
                log_v = np.log(np.array(vs, dtype=float))
                n_pts = len(log_n)
                sum_x = log_n.sum()
                sum_y = log_v.sum()
                sum_xy = (log_n * log_v).sum()
                sum_x2 = (log_n ** 2).sum()
                denom = n_pts * sum_x2 - sum_x ** 2
                if abs(denom) > 1e-12:
                    beta = float((n_pts * sum_xy - sum_x * sum_y) / denom)

        last = exp_tail[-1] if exp_tail else self.records[-1]
        return SensorReading(
            step=last["step"],
            L=last["context_length"],
            T=last["temperature"],
            entropy_mean=ent_mean,
            entropy_std=ent_std,
            comp_W64=comp_w64,
            heaps_beta=beta,
            n_words=n_words,
            n_unique=n_unique,
        )

    # -- Rollback -----------------------------------------------------------

    def snapshot(self) -> Snapshot:
        """Capture current state for potential rollback."""
        return Snapshot(
            n_records=len(self.records),
            step=self._step,
            context=self.context.clone(),
            rng_torch=torch.random.get_rng_state(),
            rng_cuda=(torch.cuda.get_rng_state()
                      if torch.cuda.is_available() else None),
            rng_numpy=np.random.get_state(),
        )

    def restore(self, snap: Snapshot) -> None:
        """Restore engine state from snapshot, undoing records since."""
        self.records[:] = self.records[:snap.n_records]
        self._step = snap.step
        self.context = snap.context
        torch.random.set_rng_state(snap.rng_torch)
        if snap.rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(snap.rng_cuda)
        np.random.set_state(snap.rng_numpy)

    # -- Persistence --------------------------------------------------------

    def save_checkpoint(
        self, path: Path, parquet_path: Path, spec: str = "",
    ) -> None:
        """Save checkpoint (context + RNG + records) and parquet snapshot."""
        torch.save({
            "step": self._step,
            "context": self.context.cpu(),
            "records": self.records,
            "rng_torch": torch.random.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state()
                         if torch.cuda.is_available() else None),
            "rng_numpy": np.random.get_state(),
            "schedule_spec": spec,
        }, path)
        pd.DataFrame(self.records).to_parquet(parquet_path, index=False)

    def load_checkpoint(self, path: Path) -> dict:
        """Load checkpoint, restore all state. Returns checkpoint dict."""
        ckpt = torch.load(path, weights_only=False)
        torch.random.set_rng_state(ckpt["rng_torch"])
        if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["rng_cuda"])
        np.random.set_state(ckpt["rng_numpy"])
        self.context = ckpt["context"].to(self.device)
        self.records = ckpt["records"]
        self._step = ckpt["step"]
        return ckpt

    def fix_texts(self) -> None:
        """Fix multi-byte UTF-8 decoded texts in-place."""
        all_ids = [r["token_id"] for r in self.records]
        all_texts = [r["decoded_text"] for r in self.records]
        fixed = fix_decoded_texts(self.tokenizer, all_ids, all_texts)
        for r, txt in zip(self.records, fixed):
            r["decoded_text"] = txt

    def save_parquet(self, path: Path) -> None:
        """Write current records to parquet."""
        pd.DataFrame(self.records).to_parquet(path, index=False)
