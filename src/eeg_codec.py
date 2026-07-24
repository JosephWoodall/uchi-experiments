"""VQ-VAE codecs for Track 2's re-run: turn EEG frames and 6-DOF motor
commands into discrete code sequences, same role codec.py's PixelVQVAE/
AudioVQVAE play for pixel/audio -- reusing `VectorQuantizer` directly
rather than duplicating it. Kept separate from codec.py (not added there)
because codec.py's own docstring scopes it to the single fixed
image.png/clip.wav inputs; here the input is a seed-dependent synthetic
batch of trials, so codes are never cached to disk (a stale cache tied to
the wrong seed's data is exactly the bug class this project has hit and
fixed multiple times elsewhere -- cheaper to just retrain, a few hundred
steps on tiny tensors).

Same conv1d-stride-2-twice architecture as codec.py's AudioVQVAE for
both, so EEG codes and command codes downsample by the identical 4x
factor and land in 1:1 temporal correspondence -- code i of the command
stream is the discretized command at the same time window as code i of
the EEG stream, no separate alignment/pooling step needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import VectorQuantizer

N_CODES = 64  # matches codec.py's pixel/audio codebook size


class _Conv1dVQVAE(nn.Module):
    """Shared shape for EEGVQVAE/CommandVQVAE -- only in_channels differs
    (21 EEG electrodes vs. 6 motor DOF)."""

    def __init__(self, in_channels: int, hidden: int = 16, embed_dim: int = 8, n_codes: int = N_CODES):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, embed_dim, 3, padding=1),
        )
        self.vq = VectorQuantizer(n_codes, embed_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, in_channels, 3, padding=1),
        )

    def forward(self, x):
        ze = self.enc(x)
        zq, idx, vq_loss = self.vq(ze)
        recon = self.dec(zq)
        return recon, idx, vq_loss


class EEGVQVAE(_Conv1dVQVAE):
    def __init__(self, n_codes: int = N_CODES):
        super().__init__(in_channels=21, n_codes=n_codes)


class CommandVQVAE(_Conv1dVQVAE):
    def __init__(self, n_codes: int = N_CODES):
        super().__init__(in_channels=6, n_codes=n_codes)


def _train_one_codec(model: nn.Module, x: torch.Tensor, steps: int, n_codes: int, tag: str):
    """x: (B, C, T). Same dead-code-revival trick as codec.py's train_codec
    (without it one codebook entry wins early and collapses everything
    onto it) -- generalized to B>1 batches of trials instead of one image.
    """
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(steps):
        opt.zero_grad()
        recon, idx, vq_loss = model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        loss.backward()
        opt.step()

        if step % 25 == 0:
            with torch.no_grad():
                ze = model.enc(x)
                flat = ze.permute(0, *range(2, ze.dim()), 1).reshape(-1, ze.shape[1])
                used = idx.unique()
                dead = torch.tensor([c for c in range(n_codes) if c not in used])
                if len(dead) > 0:
                    replacements = flat[torch.randint(0, flat.size(0), (len(dead),))]
                    replacements = replacements + 0.02 * torch.randn_like(replacements)
                    model.vq.codebook.weight.data[dead] = replacements

        if step % 100 == 0 or step == steps - 1:
            print(
                f"  [{tag} codec] step {step}: recon={recon_loss.item():.4f} "
                f"vq={vq_loss.item():.4f} unique_codes_used={idx.unique().numel()}/{n_codes}"
            )
    return model


def train_codecs(eeg: torch.Tensor, cmd: torch.Tensor, steps: int = 300, n_codes: int = N_CODES):
    """eeg: (N, T, 21), cmd: (N, T, 6) -- as returned by
    run_track2_shared_core.build_eeg_data. Returns
    (eeg_model, cmd_model, eeg_codes, cmd_codes), where *_codes are
    (N, T//4) long tensors, 1:1 aligned in time.
    """
    eeg_c = eeg.permute(0, 2, 1)  # (N, 21, T)
    cmd_c = cmd.permute(0, 2, 1)  # (N, 6, T)

    eeg_model = EEGVQVAE(n_codes=n_codes)
    _train_one_codec(eeg_model, eeg_c, steps, n_codes, "eeg")
    cmd_model = CommandVQVAE(n_codes=n_codes)
    _train_one_codec(cmd_model, cmd_c, steps, n_codes, "cmd")

    with torch.no_grad():
        _, eeg_idx, _ = eeg_model(eeg_c)
        _, cmd_idx, _ = cmd_model(cmd_c)
    return eeg_model, cmd_model, eeg_idx, cmd_idx
