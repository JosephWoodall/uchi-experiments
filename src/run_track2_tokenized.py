"""Phase V, Track 2, re-run: same shared-core-vs-baselines question as
run_track2_shared_core.py, but fusing EEG through the mechanism this repo
already proved works (Phase D's joint multimodal experiment: one shared
discrete vocab + one embedding/softmax head + per-example next-token CE
across all modalities) instead of run_track2_shared_core.py's untested
continuous-projection/MSE-head design, which lost to the text-only
baseline in 3/3 seeds.

EEG frames and their paired 6-DOF motor commands are each discretized by
a small from-scratch VQ-VAE (`eeg_codec.py`, same architecture family as
codec.py's PixelVQVAE/AudioVQVAE) into 1:1-aligned code sequences. Every
modality -- text, EEG codes, command codes -- then lives in one shared
token vocabulary and goes through the identical embedding table and
softmax head, exactly like Phase D's text/code/pixel/audio unification.
The EEG->motor-intent decode task becomes: predict the next command code
having seen the EEG codes and any earlier command codes, via the same
causal next-token CE the model already uses for text -- not a bespoke
regression head. This also removes the CE-vs-MSE loss-scale mismatch the
first Track 2 pass had.

Still a toy, CPU, off-the-safety-critical-path experiment (see
run_track2_shared_core.py's own docstring for the scope statement, which
still applies unchanged); nothing here touches Noosphere's production
code.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_lm_batch, load_lm_corpus
from eeg_codec import train_codecs
from rwkv_model import RWKVBlock
from run_track2_shared_core import build_eeg_data
from tokenizer import Tokenizer

D_MODEL = 32
N_LAYER = 2
TEXT_VOCAB = 1024
TEXT_BLOCK = 128
EEG_CODES = 64
CMD_CODES = 64
EEG_CONTENT_LEN = 32 + 32  # eeg codes then command codes, 1:1 time-aligned

# Unified vocab layout, same "offset ranges + trailing marker tokens"
# pattern as data.py's UNIFIED_VOCAB_SIZE for pixel/audio.
EEG_OFFSET = TEXT_VOCAB
CMD_OFFSET = EEG_OFFSET + EEG_CODES
MARK_TEXT = CMD_OFFSET + CMD_CODES
MARK_EEG = MARK_TEXT + 1
UNIFIED_VOCAB = MARK_EEG + 1

# Baselines get their own small scoped vocabs (own BOS token in place of a
# modality marker -- a single-modality model has nothing to disambiguate,
# but still needs *something* to condition position 0's prediction on).
TEXT_BASELINE_VOCAB = TEXT_VOCAB  # no marker needed, matches the plain
                                  # next-token-CE convention every other
                                  # text run in this repo already uses
EEG_BASELINE_VOCAB = EEG_CODES + CMD_CODES + 1
EEG_BASELINE_BOS = EEG_CODES + CMD_CODES


class UnifiedCore(nn.Module):
    """One embedding/softmax head over `vocab_size`, shared by every
    modality passed through it -- the actual fusion mechanism, not just a
    shared block stack. `vocab_size` scopes what this instance can
    represent: UNIFIED_VOCAB for the shared model, a modality-scoped
    vocab for a single-modality baseline.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL)
        self.blocks = nn.ModuleList([RWKVBlock(D_MODEL) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        h = self.emb(idx)
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln_f(h)
        return self.head(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def text_batch(ids, batch_size, marker_id=None):
    """(input, target), each (B, TEXT_BLOCK). If marker_id is given, input
    is [marker, tok0..tok126] and target is [tok0..tok127] (the marker
    donates one input slot, no change needed to get_lm_batch's fetch
    range). If not, plain teacher-forced next-token CE, matching every
    other text run in this repo.
    """
    x, targets = get_lm_batch(ids, batch_size=batch_size, block_size=TEXT_BLOCK)
    if marker_id is None:
        return x, targets[:, :, 0]
    marker_col = torch.full((x.shape[0], 1), marker_id, dtype=torch.long)
    inp = torch.cat([marker_col, x[:, :-1]], dim=1)
    return inp, x


def eeg_batch(eeg_codes, cmd_codes, batch_size, eeg_offset, cmd_offset, bos_id, generator=None):
    """(input, target), each (B, EEG_CONTENT_LEN). content = eeg codes
    (offset) then command codes (offset), input is [bos, content[:-1]],
    target is content -- same shift-by-one scheme as text_batch, so the
    causal CE at command-code positions is literally "predict this
    command code given the EEG codes and any earlier command codes",
    Track 2's actual decode task, expressed as sequence continuation.
    """
    idx = torch.randint(0, eeg_codes.shape[0], (batch_size,), generator=generator)
    content = torch.cat([eeg_codes[idx] + eeg_offset, cmd_codes[idx] + cmd_offset], dim=1)
    bos_col = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    inp = torch.cat([bos_col, content[:, :-1]], dim=1)
    return inp, content


def _masked_ce(logits, target, valid_start, valid_end):
    """CE restricted to the [valid_start, valid_end) logit slice --
    removes the raw-nats confound between the shared model's full
    UNIFIED_VOCAB-way softmax and a baseline's much smaller modality-
    scoped softmax (same "higher entropy floor from a bigger softmax,
    not directly comparable in raw nats" lesson this project already
    learned in Phase O/J's tokenizer-fairness work). Targets are always
    inside this range by construction, so this only removes classes that
    were never a real target, not real difficulty.
    """
    sub_logits = logits[..., valid_start:valid_end]
    local_target = target - valid_start
    return F.cross_entropy(sub_logits.reshape(-1, valid_end - valid_start), local_target.reshape(-1)).item()


@torch.no_grad()
def eval_text(model, val_ids, vocab_size, marker_id=None, restrict=None, n_batches=10):
    """Returns (raw_ce, masked_ce). masked_ce == raw_ce when restrict is
    None (a baseline whose vocab already *is* the restricted range)."""
    model.eval()
    raw, masked = [], []
    for _ in range(n_batches):
        inp, target = text_batch(val_ids, batch_size=16, marker_id=marker_id)
        logits = model(inp)
        raw.append(F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1)).item())
        if restrict is not None:
            masked.append(_masked_ce(logits, target, *restrict))
    model.train()
    raw_ce = sum(raw) / len(raw)
    masked_ce = sum(masked) / len(masked) if restrict is not None else raw_ce
    return raw_ce, masked_ce


@torch.no_grad()
def eval_eeg(model, eeg_codes, cmd_codes, vocab_size, eeg_offset, cmd_offset, bos_id,
             restrict=None, restrict_cmd=None, n_batches=10):
    """Returns (overall_raw, overall_masked, cmd_only_raw, cmd_only_masked).
    overall/cmd_only mirror eval_text's raw-vs-vocab-restricted split;
    cmd_only additionally isolates the real EEG->motor-intent decode task
    (target positions 32:) from EEG's own self-predictability (0:32).
    """
    model.eval()
    overall_raw, overall_masked, cmd_raw, cmd_masked = [], [], [], []
    for _ in range(n_batches):
        inp, target = eeg_batch(eeg_codes, cmd_codes, batch_size=4, eeg_offset=eeg_offset,
                                 cmd_offset=cmd_offset, bos_id=bos_id)
        logits = model(inp)
        overall_raw.append(F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1)).item())
        cmd_logits = logits[:, 32:, :]
        cmd_target = target[:, 32:]
        cmd_raw.append(F.cross_entropy(cmd_logits.reshape(-1, vocab_size), cmd_target.reshape(-1)).item())
        if restrict is not None:
            overall_masked.append(_masked_ce(logits, target, *restrict))
        if restrict_cmd is not None:
            cmd_masked.append(_masked_ce(cmd_logits, cmd_target, *restrict_cmd))
    model.train()
    o_raw = sum(overall_raw) / len(overall_raw)
    o_masked = sum(overall_masked) / len(overall_masked) if restrict is not None else o_raw
    c_raw = sum(cmd_raw) / len(cmd_raw)
    c_masked = sum(cmd_masked) / len(cmd_masked) if restrict_cmd is not None else c_raw
    return o_raw, o_masked, c_raw, c_masked


def train_one(model, vocab_size, modalities, text_train_ids, eeg_codes_train, cmd_codes_train,
              text_marker, eeg_offset, cmd_offset, eeg_bos, steps, lr, log_every, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        if "text" in modalities and ("eeg" not in modalities or step % 2 == 1):
            inp, target = text_batch(text_train_ids, batch_size=16, marker_id=text_marker)
        else:
            inp, target = eeg_batch(eeg_codes_train, cmd_codes_train, batch_size=4,
                                     eeg_offset=eeg_offset, cmd_offset=cmd_offset, bos_id=eeg_bos)
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"  [{tag}] step {step}/{steps} loss={loss.item():.4f}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--codec-steps", type=int, default=300)
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
    print(f"  train trials={eeg_train.shape[0]} val trials={eeg_val.shape[0]}")

    print("Training EEG + command VQ-VAE codecs on train trials only...")
    eeg_model, cmd_model, eeg_codes_train, cmd_codes_train = train_codecs(
        eeg_train, cmd_train, steps=args.codec_steps
    )
    with torch.no_grad():
        _, eeg_codes_val, _ = eeg_model(eeg_val.permute(0, 2, 1))
        _, cmd_codes_val, _ = cmd_model(cmd_val.permute(0, 2, 1))
    print(
        f"  code seq len: {eeg_codes_train.shape[1]} (eeg) + {cmd_codes_train.shape[1]} (cmd) "
        f"= {EEG_CONTENT_LEN} per trial"
    )

    torch.manual_seed(args.seed)
    shared = UnifiedCore(UNIFIED_VOCAB)
    torch.manual_seed(args.seed + 1)
    text_only = UnifiedCore(TEXT_BASELINE_VOCAB)
    torch.manual_seed(args.seed + 2)
    eeg_only = UnifiedCore(EEG_BASELINE_VOCAB)

    print(
        f"\nUnified vocab={UNIFIED_VOCAB} (text 0-{TEXT_VOCAB-1}, eeg {EEG_OFFSET}-{CMD_OFFSET-1}, "
        f"cmd {CMD_OFFSET}-{MARK_TEXT-1}, MARK_TEXT={MARK_TEXT}, MARK_EEG={MARK_EEG})"
    )
    print(
        f"Param counts -- shared: {shared.num_params():,}  text_only: {text_only.num_params():,}  "
        f"eeg_only: {eeg_only.num_params():,}\n"
        "(block stack is the same D_MODEL/N_LAYER in all three; embedding/head size scales with "
        "each model's own vocab scope, same convention as run_track2_shared_core.py.)\n"
    )

    print("Training shared core (alternating text/eeg steps, one unified vocab)...")
    train_one(
        shared, UNIFIED_VOCAB, {"text", "eeg"}, text_train, eeg_codes_train, cmd_codes_train,
        MARK_TEXT, EEG_OFFSET, CMD_OFFSET, MARK_EEG, args.steps, args.lr, args.log_every, "shared",
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

    # restrict ranges let the shared model's masked CE be compared
    # apples-to-apples against a baseline's own (much smaller) softmax --
    # see _masked_ce's docstring for why raw nats alone would mislead.
    shared_text_raw, shared_text_masked = eval_text(
        shared, text_val, UNIFIED_VOCAB, marker_id=MARK_TEXT, restrict=(0, TEXT_VOCAB)
    )
    shared_eeg_o_raw, shared_eeg_o_masked, shared_eeg_c_raw, shared_eeg_c_masked = eval_eeg(
        shared, eeg_codes_val, cmd_codes_val, UNIFIED_VOCAB, EEG_OFFSET, CMD_OFFSET, MARK_EEG,
        restrict=(EEG_OFFSET, MARK_TEXT), restrict_cmd=(CMD_OFFSET, MARK_TEXT),
    )
    baseline_text_raw, _ = eval_text(text_only, text_val, TEXT_BASELINE_VOCAB, marker_id=None)
    baseline_eeg_o_raw, _, baseline_eeg_c_raw, _ = eval_eeg(
        eeg_only, eeg_codes_val, cmd_codes_val, EEG_BASELINE_VOCAB, 0, EEG_CODES, EEG_BASELINE_BOS
    )

    print("\n=== Track 2 (tokenized) result ===")
    print(f"text CE (val):          shared_raw={shared_text_raw:.4f}  shared_masked={shared_text_masked:.4f}  "
          f"text_only_baseline={baseline_text_raw:.4f}")
    print(f"eeg overall CE (val):   shared_raw={shared_eeg_o_raw:.4f}  shared_masked={shared_eeg_o_masked:.4f}  "
          f"eeg_only_baseline={baseline_eeg_o_raw:.4f}")
    print(f"eeg cmd-only CE (val):  shared_raw={shared_eeg_c_raw:.4f}  shared_masked={shared_eeg_c_masked:.4f}  "
          f"eeg_only_baseline={baseline_eeg_c_raw:.4f}")
    print(
        "(masked = shared's logits restricted to the same class range the baseline already predicts "
        "over -- the fair comparison; raw is the model's real full-vocab softmax, always >= masked.)"
    )

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
