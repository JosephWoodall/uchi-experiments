"""Phase V, Track 2, third fusion mechanism: shared RWKV trunk + small
per-modality bottleneck adapters (Houlsby et al. 2019, arXiv:1902.00751),
at a bigger toy size than the first two passes.

Both prior mechanisms lost on every metric at d_model=32/n_layer=2:
run_track2_shared_core.py (continuous EEG projection + MSE head, lost
3/3 seeds on text) and run_track2_tokenized.py (full unified-vocab
fusion via VQ-VAE-discretized EEG, lost on all 3 masked-CE metrics).
Both forced one undifferentiated set of weights to represent every
modality identically. Adapters keep the same one-embedding/one-softmax
unified vocab (still "comprehensively fluent on any token that comes
through," not a blend) but let each modality bend the shared
representation locally after every block, instead of requiring the
trunk's core weights to serve both distributions with zero
modality-specific capacity.

Per the user's explicit choice this round: also a real size step up
(d_model 32->64, n_layer 2->3 -- this project's own established "xs" ->
"s" rung, see train.py's SIZES), and per their explicit instruction, this
pass does not gate on beating the baselines by a specific margin --
that's deferred, not abandoned.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_track2_tokenized import (
    CMD_CODES, CMD_OFFSET, EEG_BASELINE_BOS, EEG_BASELINE_VOCAB, EEG_CODES,
    EEG_OFFSET, MARK_EEG, MARK_TEXT, TEXT_BASELINE_VOCAB, TEXT_VOCAB,
    UNIFIED_VOCAB, _masked_ce, build_eeg_data, eeg_batch, eval_eeg,
    eval_text, text_batch, train_codecs, train_one,
)
from data import load_lm_corpus
from rwkv_model import RWKVBlock
from tokenizer import Tokenizer

D_MODEL = 64   # up from 32 -- this project's "s" rung, not "xs"
N_LAYER = 3    # up from 2
BOTTLENECK = 16  # 4x reduction, standard Houlsby ratio


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


class Adapter(nn.Module):
    """Down-project, GELU, up-project, residual. Up-proj zero-initialized
    so the adapter starts as an identity passthrough -- the shared
    trunk's behavior is unperturbed at step 0, each modality only starts
    bending it as training actually finds a reason to.
    """

    def __init__(self, d_model, bottleneck):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class PlainCore(nn.Module):
    """Single-modality baseline: same trunk shape as AdapterCore's shared
    trunk (fair block-stack comparison, same convention as the first two
    Track 2 passes), no adapters -- a single-modality model has nothing
    to adapt to.
    """

    def __init__(self, vocab_size, d_model=D_MODEL, n_layer=N_LAYER):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.apply(_init_weights)

    def forward(self, idx):
        h = self.emb(idx)
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln_f(h)
        return self.head(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class AdapterCore(nn.Module):
    """Shared trunk + one embedding/softmax head over the unified vocab
    (same fusion mechanism as run_track2_tokenized.py's UnifiedCore), plus
    a per-modality bottleneck adapter after every block.
    """

    def __init__(self, vocab_size, modalities=("text", "eeg"), d_model=D_MODEL,
                 n_layer=N_LAYER, bottleneck=BOTTLENECK):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model) for _ in range(n_layer)])
        self.adapters = nn.ModuleDict({
            m: nn.ModuleList([Adapter(d_model, bottleneck) for _ in range(n_layer)])
            for m in modalities
        })
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.apply(_init_weights)
        # self.apply above blanket-inits every nn.Linear, including each
        # Adapter's up-proj -- re-zero it so adapters still start as a
        # no-op, not undoing the point of the zero-init above.
        for mod_adapters in self.adapters.values():
            for a in mod_adapters:
                nn.init.zeros_(a.up.weight)
                nn.init.zeros_(a.up.bias)

    def forward(self, idx, modality):
        h = self.emb(idx)
        for block, adapter in zip(self.blocks, self.adapters[modality]):
            h, _ = block(h)
            h = adapter(h)
        h = self.ln_f(h)
        return self.head(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_shared_adapter(model, vocab_size, text_train_ids, eeg_codes_train, cmd_codes_train,
                          text_marker, eeg_offset, cmd_offset, eeg_bos, steps, lr, log_every,
                          text_ratio=0.5, generator=None):
    """text_ratio: probability a given step trains on text rather than
    eeg. Naive 50/50 alternation starved whichever domain actually needed
    more updates to converge -- not standard practice (T5/PaLM use
    examples/tokens-proportional mixing with a temperature, DoReMi learns
    the weights outright); this is the same fix in its simplest form, a
    fixed ratio instead of a strict step-parity split.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        if torch.rand(1, generator=generator).item() < text_ratio:
            inp, target = text_batch(text_train_ids, batch_size=16, marker_id=text_marker)
            logits = model(inp, "text")
        else:
            inp, target = eeg_batch(eeg_codes_train, cmd_codes_train, batch_size=4,
                                     eeg_offset=eeg_offset, cmd_offset=cmd_offset, bos_id=eeg_bos)
            logits = model(inp, "eeg")
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"  [shared_adapter] step {step}/{steps} loss={loss.item():.4f}")
    return model


@torch.no_grad()
def eval_text_adapter(model, val_ids, vocab_size, marker_id, restrict, n_batches=10):
    model.eval()
    raw, masked = [], []
    for _ in range(n_batches):
        inp, target = text_batch(val_ids, batch_size=16, marker_id=marker_id)
        logits = model(inp, "text")
        raw.append(F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1)).item())
        masked.append(_masked_ce(logits, target, *restrict))
    model.train()
    return sum(raw) / len(raw), sum(masked) / len(masked)


@torch.no_grad()
def eval_eeg_adapter(model, eeg_codes, cmd_codes, vocab_size, eeg_offset, cmd_offset, bos_id,
                      restrict, restrict_cmd, n_batches=10):
    model.eval()
    o_raw, o_masked, c_raw, c_masked = [], [], [], []
    for _ in range(n_batches):
        inp, target = eeg_batch(eeg_codes, cmd_codes, batch_size=4, eeg_offset=eeg_offset,
                                 cmd_offset=cmd_offset, bos_id=bos_id)
        logits = model(inp, "eeg")
        o_raw.append(F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1)).item())
        cmd_logits = logits[:, 32:, :]
        cmd_target = target[:, 32:]
        c_raw.append(F.cross_entropy(cmd_logits.reshape(-1, vocab_size), cmd_target.reshape(-1)).item())
        o_masked.append(_masked_ce(logits, target, *restrict))
        c_masked.append(_masked_ce(cmd_logits, cmd_target, *restrict_cmd))
    model.train()
    return (sum(o_raw) / len(o_raw), sum(o_masked) / len(o_masked),
            sum(c_raw) / len(c_raw), sum(c_masked) / len(c_masked))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--codec-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=57)
    ap.add_argument("--text-ratio", type=float, default=0.5,
                     help="probability a shared-model step trains on text vs. eeg "
                          "(0.5 = old naive alternation)")
    ap.add_argument("--d-model", type=int, default=D_MODEL)
    ap.add_argument("--n-layer", type=int, default=N_LAYER)
    ap.add_argument("--bottleneck", type=int, default=BOTTLENECK)
    ap.add_argument("--n-subjects", type=int, default=3,
                     help="synthetic EEG subjects (default 3 -- the original small pass)")
    ap.add_argument("--n-trials", type=int, default=10,
                     help="trials per subject (default 10 -- 30 total, 24 train/6 val)")
    ap.add_argument("--text-domain", choices=["rj", "text"], default="rj",
                     help="'rj' = Romeo & Juliet only (~50K tokens, the original Track 2 "
                          "setup -- too small for big baselines, see the l-tier overfitting "
                          "finding). 'text' = rj+gutenberg (~300M tokens), the fix.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"Building text corpus ({args.text_domain}, vocab=1024)...")
    tok = Tokenizer(vocab_size=TEXT_VOCAB)
    text_train, text_val = load_lm_corpus(args.text_domain, tok)

    print(f"Building synthetic EEG corpus (noosphere/v2 generator, "
          f"n_subjects={args.n_subjects} n_trials={args.n_trials})...")
    eeg_train, cmd_train, eeg_val, cmd_val = build_eeg_data(
        n_subjects=args.n_subjects, n_trials=args.n_trials, seed=args.seed
    )
    print(f"  train trials={eeg_train.shape[0]} val trials={eeg_val.shape[0]}")

    print("Training EEG + command VQ-VAE codecs on train trials only...")
    eeg_model, cmd_model, eeg_codes_train, cmd_codes_train = train_codecs(
        eeg_train, cmd_train, steps=args.codec_steps
    )
    with torch.no_grad():
        _, eeg_codes_val, _ = eeg_model(eeg_val.permute(0, 2, 1))
        _, cmd_codes_val, _ = cmd_model(cmd_val.permute(0, 2, 1))

    torch.manual_seed(args.seed)
    shared = AdapterCore(UNIFIED_VOCAB, d_model=args.d_model, n_layer=args.n_layer,
                         bottleneck=args.bottleneck)
    torch.manual_seed(args.seed + 1)
    text_only = PlainCore(TEXT_BASELINE_VOCAB, d_model=args.d_model, n_layer=args.n_layer)
    torch.manual_seed(args.seed + 2)
    eeg_only = PlainCore(EEG_BASELINE_VOCAB, d_model=args.d_model, n_layer=args.n_layer)

    print(
        f"\nd_model={args.d_model} n_layer={args.n_layer} bottleneck={args.bottleneck} "
        f"text_ratio={args.text_ratio}"
    )
    print(
        f"Param counts -- shared(+adapters): {shared.num_params():,}  "
        f"text_only: {text_only.num_params():,}  eeg_only: {eeg_only.num_params():,}\n"
    )

    print(f"Training shared core (adapters, text_ratio={args.text_ratio})...")
    train_shared_adapter(
        shared, UNIFIED_VOCAB, text_train, eeg_codes_train, cmd_codes_train,
        MARK_TEXT, EEG_OFFSET, CMD_OFFSET, MARK_EEG, args.steps, args.lr, args.log_every,
        text_ratio=args.text_ratio,
    )
    print("Training text-only baseline...")
    train_one(
        text_only, TEXT_BASELINE_VOCAB, {"text"}, text_train, eeg_codes_train, cmd_codes_train,
        None, 0, 0, 0, args.steps, args.lr, args.log_every, "text_only",
    )
    print("Training eeg-only baseline...")
    train_one(
        eeg_only, EEG_BASELINE_VOCAB, {"eeg"}, text_train, eeg_codes_train, cmd_codes_train,
        None, 0, EEG_CODES, EEG_BASELINE_BOS, args.steps, args.lr, args.log_every, "eeg_only",
    )

    shared_text_raw, shared_text_masked = eval_text_adapter(
        shared, text_val, UNIFIED_VOCAB, MARK_TEXT, (0, TEXT_VOCAB)
    )
    shared_eeg_o_raw, shared_eeg_o_masked, shared_eeg_c_raw, shared_eeg_c_masked = eval_eeg_adapter(
        shared, eeg_codes_val, cmd_codes_val, UNIFIED_VOCAB, EEG_OFFSET, CMD_OFFSET, MARK_EEG,
        (EEG_OFFSET, MARK_TEXT), (CMD_OFFSET, MARK_TEXT),
    )
    baseline_text_raw, _ = eval_text(text_only, text_val, TEXT_BASELINE_VOCAB, marker_id=None)
    baseline_eeg_o_raw, _, baseline_eeg_c_raw, _ = eval_eeg(
        eeg_only, eeg_codes_val, cmd_codes_val, EEG_BASELINE_VOCAB, 0, EEG_CODES, EEG_BASELINE_BOS
    )

    print(f"\n=== Track 2 (adapters, d_model={args.d_model}/n_layer={args.n_layer}, "
          f"text_ratio={args.text_ratio}) result ===")
    print(f"text CE (val):          shared_raw={shared_text_raw:.4f}  shared_masked={shared_text_masked:.4f}  "
          f"text_only_baseline={baseline_text_raw:.4f}")
    print(f"eeg overall CE (val):   shared_raw={shared_eeg_o_raw:.4f}  shared_masked={shared_eeg_o_masked:.4f}  "
          f"eeg_only_baseline={baseline_eeg_o_raw:.4f}")
    print(f"eeg cmd-only CE (val):  shared_raw={shared_eeg_c_raw:.4f}  shared_masked={shared_eeg_c_masked:.4f}  "
          f"eeg_only_baseline={baseline_eeg_c_raw:.4f}")

    text_ok = shared_text_masked <= baseline_text_raw
    eeg_ok = shared_eeg_o_masked <= baseline_eeg_o_raw
    cmd_ok = shared_eeg_c_masked <= baseline_eeg_c_raw
    print(f"\nshared (masked) <= text baseline:         {text_ok}")
    print(f"shared (masked) <= eeg overall baseline:  {eeg_ok}")
    print(f"shared (masked) <= eeg cmd-only baseline: {cmd_ok}")
    if text_ok and eeg_ok:
        print("VERDICT: shared core did not lose to either baseline -- hypothesis alive at toy scale.")
    else:
        print("VERDICT: shared core lost on at least one side -- real evidence about this toy config, not the goal.")


if __name__ == "__main__":
    main()
