"""Synthetic single-image and single-audio-clip inputs -- same "one
deliberate input" philosophy as romeo_and_juliet.txt and the stdlib corpus,
not scraped data. Structured (not noise) so the codecs have something
learnable: concentric shapes for the image, a varying-frequency melody for
audio.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent


def make_image(path: Path, size: int = 256, block: int = 4):
    """Shaded block grid, not flat regions -- a mostly-uniform image (the
    original design: thin outlines on a flat background) starves the VQ-VAE
    encoder of anything to distinguish and collapses the codebook to one
    code (observed first pass). Every block gets a distinct, deterministic
    shade so there's genuine local texture everywhere.
    """
    arr = np.zeros((size, size), dtype=np.uint8)
    for by in range(0, size, block):
        for bx in range(0, size, block):
            row, col = by // block, bx // block
            val = int(128 + 100 * np.sin(row * 0.7) * np.cos(col * 0.5) + 20 * ((row + col) % 3))
            arr[by : by + block, bx : bx + block] = np.clip(val, 0, 255)
    img = Image.fromarray(arr, mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_audio(path: Path, sample_rate: int = 8000, seconds: float = 4.0):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    notes_hz = [220, 261, 293, 329, 349, 392, 440, 392, 349, 329, 293, 261]
    note_len = len(t) // len(notes_hz)
    wave = np.zeros_like(t)
    for i, hz in enumerate(notes_hz):
        seg = slice(i * note_len, (i + 1) * note_len)
        wave[seg] = 0.5 * np.sin(2 * np.pi * hz * t[seg])
    wave_i16 = (wave * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, wave_i16)


if __name__ == "__main__":
    make_image(ROOT / "data" / "pixel" / "image.png")
    make_audio(ROOT / "data" / "audio" / "clip.wav")
    print("wrote data/pixel/image.png and data/audio/clip.wav")
