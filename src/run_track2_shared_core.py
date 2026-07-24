"""Phase V, Track 2: does one shared RWKV core trained on both text
(next-token CE) and synthetic EEG->6DOF motor-intent regression (MSE) do
at least as well as two separate, identically-sized single-modality
cores? Toy CPU scale, per the small-scale-first rule (see
core_principle.md / no_scaleup_without_proof) -- this decides whether the
cross-modality half of the "one fluent model" objective is worth a real
modality-router architecture next. It does not decide the objective
itself (already confirmed as this repo's North Star).

Reuses noosphere's own synthetic_eeg.py generator directly, unmodified,
by file path (zero hardware/subject risk, off the safety-critical path
entirely) and this repo's existing rj text corpus / tokenizer /
RWKVBlock. Nothing here is wired into Ducky, Uchi, or Noosphere's real
pipeline; no Noosphere production file is imported or touched.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_lm_batch, load_lm_corpus
from rwkv_model import RWKVBlock
from tokenizer import Tokenizer

NOOSPHERE_EEG = Path(
    "/home/redleadr/workspace/noosphere/v2_digital_self_replication/data/synthetic_eeg.py"
)
_spec = importlib.util.spec_from_file_location("synthetic_eeg", NOOSPHERE_EEG)
synthetic_eeg = importlib.util.module_from_spec(_spec)
sys.modules["synthetic_eeg"] = synthetic_eeg  # dataclass's own __module__ resolution
                                               # needs this registered before exec
_spec.loader.exec_module(synthetic_eeg)

D_MODEL = 32
N_LAYER = 2
TEXT_VOCAB = 1024
TEXT_BLOCK = 128
N_CH = synthetic_eeg.N_CH  # 21
N_DOF = 6
EEG_T = 128
EEG_FS = 64  # downsampled from the generator's 256Hz default -> matches
             # TEXT_BLOCK's toy sequence length instead of a 1024-step loop


class Core(nn.Module):
    """One RWKVBlock stack, plus modality-specific embed/head pairs only
    for the modalities this instance owns. `modalities={"text","eeg"}` is
    the shared core; a single-element set is a size-matched baseline --
    identical D_MODEL/N_LAYER block stack, just not shared with the other
    modality.
    """

    def __init__(self, modalities):
        super().__init__()
        self.modalities = modalities
        self.blocks = nn.ModuleList([RWKVBlock(D_MODEL) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        if "text" in modalities:
            self.text_emb = nn.Embedding(TEXT_VOCAB, D_MODEL)
            self.text_head = nn.Linear(D_MODEL, TEXT_VOCAB, bias=False)
            self.text_head.weight = self.text_emb.weight
        if "eeg" in modalities:
            self.eeg_proj = nn.Linear(N_CH, D_MODEL)
            self.eeg_head = nn.Linear(D_MODEL, N_DOF)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # RWKVModel's own init (std=0.02) -- skipped by default since Core
        # builds raw RWKVBlock modules directly, not through RWKVModel.
        # Without this, nn.Embedding's default std=1 init blew up the
        # tied text_head logits (~30 nats CE instead of ln(1024)~6.9).
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, modality, x):
        h = self.text_emb(x) if modality == "text" else self.eeg_proj(x)
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln_f(h)
        return self.text_head(h) if modality == "text" else self.eeg_head(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_eeg_data(n_subjects=3, n_trials=10, seed=0):
    """(train_eeg, train_cmd, val_eeg, val_cmd), each (N, EEG_T, C) float32.
    Trial-level 80/20 split -- held-out trials, not held-out timesteps
    within a trial (those are strongly autocorrelated).
    """
    data = synthetic_eeg.make_training_batch(
        n_subjects=n_subjects,
        n_trials=n_trials,
        trial_duration_s=EEG_T / EEG_FS,
        fs=EEG_FS,
        seed=seed,
    )
    eeg_all, cmd_all = [], []
    for sub in data.values():
        eeg_all.append(torch.from_numpy(sub["eeg"]))
        cmd_all.append(torch.from_numpy(sub["commands"]))
    eeg_all = torch.cat(eeg_all, dim=0)
    cmd_all = torch.cat(cmd_all, dim=0)
    n = eeg_all.shape[0]
    n_val = max(1, int(n * 0.2))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return eeg_all[train_idx], cmd_all[train_idx], eeg_all[val_idx], cmd_all[val_idx]


def get_eeg_batch(eeg, cmd, batch_size, generator=None):
    idx = torch.randint(0, eeg.shape[0], (batch_size,), generator=generator)
    return eeg[idx], cmd[idx]


@torch.no_grad()
def eval_text(model, val_ids, n_batches=10):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, targets = get_lm_batch(val_ids, batch_size=16, block_size=TEXT_BLOCK)
        logits = model("text", x)
        loss = F.cross_entropy(
            logits.reshape(-1, TEXT_VOCAB), targets[:, :, 0].reshape(-1)
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def eval_eeg(model, eeg_val, cmd_val, n_batches=10):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_eeg_batch(eeg_val, cmd_val, batch_size=4)
        pred = model("eeg", x)
        losses.append(F.mse_loss(pred, y).item())
    model.train()
    return sum(losses) / len(losses)


def train_one(model, modalities, text_ids, eeg_train, cmd_train, steps, lr, log_every, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_ids, _ = text_ids
    for step in range(1, steps + 1):
        if "text" in modalities and ("eeg" not in modalities or step % 2 == 1):
            x, targets = get_lm_batch(train_ids, batch_size=16, block_size=TEXT_BLOCK)
            logits = model("text", x)
            loss = F.cross_entropy(
                logits.reshape(-1, TEXT_VOCAB), targets[:, :, 0].reshape(-1)
            )
        else:
            x, y = get_eeg_batch(eeg_train, cmd_train, batch_size=4)
            pred = model("eeg", x)
            loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"  [{tag}] step {step}/{steps} loss={loss.item():.4f}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=57)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print("Building text corpus (rj, vocab=1024)...")
    tok = Tokenizer(vocab_size=TEXT_VOCAB)
    text_train, text_val = load_lm_corpus("rj", tok)

    print("Building synthetic EEG corpus (noosphere/v2 generator)...")
    eeg_train, cmd_train, eeg_val, cmd_val = build_eeg_data(seed=args.seed)
    print(
        f"  train trials={eeg_train.shape[0]} val trials={eeg_val.shape[0]} "
        f"T={EEG_T} @ {EEG_FS}Hz"
    )

    torch.manual_seed(args.seed)
    shared = Core({"text", "eeg"})
    torch.manual_seed(args.seed + 1)
    text_only = Core({"text"})
    torch.manual_seed(args.seed + 2)
    eeg_only = Core({"eeg"})

    print(
        f"\nParam counts -- shared: {shared.num_params():,}  "
        f"text_only: {text_only.num_params():,}  eeg_only: {eeg_only.num_params():,}"
    )
    print("(blocks stack is the same D_MODEL/N_LAYER in all three -- the "
          "comparison is 'one shared trunk' vs 'two separate trunks of the "
          "same size', not a raw total-param match.)\n")

    print("Training shared core (alternating text/eeg steps)...")
    train_one(
        shared, {"text", "eeg"}, (text_train, text_val), eeg_train, cmd_train,
        args.steps, args.lr, args.log_every, "shared",
    )
    print("Training text-only baseline...")
    train_one(
        text_only, {"text"}, (text_train, text_val), eeg_train, cmd_train,
        args.steps, args.lr, args.log_every, "text_only",
    )
    print("Training eeg-only baseline...")
    train_one(
        eeg_only, {"eeg"}, (text_train, text_val), eeg_train, cmd_train,
        args.steps, args.lr, args.log_every, "eeg_only",
    )

    shared_text = eval_text(shared, text_val)
    shared_eeg = eval_eeg(shared, eeg_val, cmd_val)
    baseline_text = eval_text(text_only, text_val)
    baseline_eeg = eval_eeg(eeg_only, eeg_val, cmd_val)

    print("\n=== Track 2 result ===")
    print(f"text (CE, val):  shared={shared_text:.4f}  text_only_baseline={baseline_text:.4f}")
    print(f"eeg  (MSE, val): shared={shared_eeg:.4f}  eeg_only_baseline={baseline_eeg:.4f}")

    text_ok = shared_text <= baseline_text
    eeg_ok = shared_eeg <= baseline_eeg
    print(f"\nshared <= text baseline: {text_ok}")
    print(f"shared <= eeg baseline:  {eeg_ok}")
    if text_ok and eeg_ok:
        print("VERDICT: shared core did not lose to either baseline -- hypothesis alive at toy scale.")
    else:
        print("VERDICT: shared core lost on at least one side -- real evidence about this toy config, not the goal.")


if __name__ == "__main__":
    main()
