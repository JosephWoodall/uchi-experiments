"""Tiny from-scratch VQ-VAE codecs (van den Oord et al. 2017, arXiv:1711.00937)
turning the single image / single audio clip into discrete code sequences --
the same role EnCodec/VQGAN play in core_principle.md's unified-token-space
plan, but small enough to train in seconds on CPU instead of downloading a
pretrained multi-hundred-MB model. Each codec's codebook is the token vocab
for that modality (see train.py's PIXEL/AUDIO branches) -- a smaller, honest
scope than fully merging into the text/code vocab today (see gen_multimodal_data.py
docstring and the conversation record for why that merge is a follow-up).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
N_CODES = 64  # shared codebook size for both modalities -- the vocab size train.py uses


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, embed_dim)
        self.codebook.weight.data.uniform_(-1 / n_codes, 1 / n_codes)
        self.beta = beta

    def forward(self, ze):
        # ze: (B, embed_dim, *spatial) -> flatten spatial dims
        shape = ze.shape
        flat = ze.permute(0, *range(2, ze.dim()), 1).reshape(-1, shape[1])
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(1)
        )
        idx = dist.argmin(dim=1)
        zq_flat = self.codebook(idx)

        codebook_loss = F.mse_loss(zq_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, zq_flat.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        zq_flat = flat + (zq_flat - flat).detach()  # straight-through
        zq = zq_flat.reshape(shape[0], *shape[2:], shape[1]).movedim(-1, 1)
        idx = idx.reshape(shape[0], *shape[2:])
        return zq, idx, vq_loss


class PixelVQVAE(nn.Module):
    def __init__(self, hidden: int = 32, embed_dim: int = 16, n_codes: int = N_CODES):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, embed_dim, 3, padding=1),
        )
        self.vq = VectorQuantizer(n_codes, embed_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        ze = self.enc(x)
        zq, idx, vq_loss = self.vq(ze)
        recon = self.dec(zq)
        return recon, idx, vq_loss


class AudioVQVAE(nn.Module):
    def __init__(self, hidden: int = 32, embed_dim: int = 16, n_codes: int = N_CODES):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, embed_dim, 3, padding=1),
        )
        self.vq = VectorQuantizer(n_codes, embed_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, 1, 3, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        ze = self.enc(x)
        zq, idx, vq_loss = self.vq(ze)
        recon = self.dec(zq)
        return recon, idx, vq_loss


def _load_image(size: int = 256) -> torch.Tensor:
    img = Image.open(ROOT / "data" / "pixel" / "image.png").convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def _load_audio() -> torch.Tensor:
    sr, wave = wavfile.read(ROOT / "data" / "audio" / "clip.wav")
    arr = wave.astype(np.float32) / 32767.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,L)


def train_codec(modality: str, steps: int = 500, n_codes: int = N_CODES) -> torch.Tensor:
    """Trains the codec on the single image/clip and returns its flattened
    code-index sequence (raster order for pixel, temporal order for audio).
    Cached to disk -- codec training is a one-time cost, not per-experiment.
    """
    cache_path = CACHE_DIR / f"{modality}_codes.pt"
    if cache_path.exists():
        return torch.load(cache_path)

    if modality == "pixel":
        x = _load_image()
        model = PixelVQVAE(n_codes=n_codes)
    elif modality == "audio":
        x = _load_audio()
        model = AudioVQVAE(n_codes=n_codes)
    else:
        raise ValueError(modality)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(steps):
        opt.zero_grad()
        recon, idx, vq_loss = model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss
        loss.backward()
        opt.step()

        # Dead-code revival: without this, one codebook entry wins early
        # and gradient reinforcement collapses everything onto it (observed
        # first pass: pixel codec fell to 1/64 codes). Reseed unused codes
        # from real encoder outputs every 25 steps so they can compete again.
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
            print(f"  [{modality} codec] step {step}: recon={recon_loss.item():.4f} vq={vq_loss.item():.4f} "
                  f"unique_codes_used={idx.unique().numel()}/{n_codes}")

    with torch.no_grad():
        _, idx, _ = model(x)
    codes = idx.reshape(-1)  # flatten spatial/temporal dims into one sequence

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(codes, cache_path)
    return codes


if __name__ == "__main__":
    pixel_codes = train_codec("pixel")
    audio_codes = train_codec("audio")
    print(f"pixel: {len(pixel_codes)} tokens, audio: {len(audio_codes)} tokens")
