"""Generate synthetic data and train the FusionNet.

We don't have a labelled real-world multimodal dataset, so we synthesise
one that is *faithful to the problem*: each sample represents a moment in
which a face and a piece of text may or may not agree.

Generation process for one sample
---------------------------------
1.  Pick a "true" emotion `e` uniformly from the seven canonical classes.
2.  Build the *visual* probability vector `v` as a Dirichlet-style spike
    around `e`  (the face usually leaks the true affective state).
3.  With probability `p_mismatch = 0.35`, replace the *textual* vector
    `t` with a spike around a DIFFERENT polarity emotion -- this is
    exactly the "I'm fine while looking sad" situation.
4.  Otherwise spike `t` around `e` as well.

The targets are
    * emotion label = the visually shown emotion (face wins on conflicts),
    * alignment label = 1 if both modalities point to the same polarity,
      else 0.

Run with:
    python train_fusion.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from modules.fusion import FusionNet
from modules.utils import CANONICAL_EMOTIONS, EMOTION_POLARITY


def soft_vector(target_idx: int, sharpness: float = 6.0, rng=None) -> np.ndarray:
    """Build a 7-d probability vector spiked at `target_idx`."""
    rng = rng or np.random
    base = rng.dirichlet(np.ones(7) * 0.5)
    base[target_idx] += sharpness
    base = base / base.sum()
    return base.astype(np.float32)


def make_dataset(
    n: int = 8000,
    p_mismatch: float = 0.35,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    polarities = [EMOTION_POLARITY[e] for e in CANONICAL_EMOTIONS]

    X, Y_emo, Y_align = [], [], []
    for _ in range(n):
        true_idx = int(rng.integers(0, 7))
        v = soft_vector(true_idx, rng=rng)
        if rng.random() < p_mismatch:
            # Choose any emotion whose polarity differs from the true one.
            candidates = [
                i for i, p in enumerate(polarities) if p != polarities[true_idx]
            ]
            other_idx = int(rng.choice(candidates))
            t = soft_vector(other_idx, rng=rng)
            aligned = 0
        else:
            t = soft_vector(true_idx, rng=rng)
            aligned = 1
        X.append(np.concatenate([v, t]))
        Y_emo.append(true_idx)
        Y_align.append(aligned)

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(Y_emo, dtype=torch.long),
        torch.tensor(Y_align, dtype=torch.float32),
    )


def train() -> None:
    X, y_emo, y_al = make_dataset()
    ds = TensorDataset(X, y_emo, y_al)
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    net = FusionNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    n_epochs = 30
    for ep in range(n_epochs):
        total, n_emo_ok, n_align_ok, n_seen = 0.0, 0, 0, 0
        for x, ye, ya in loader:
            opt.zero_grad()
            emo_logits, align_logit = net(x)
            loss = ce(emo_logits, ye) + bce(align_logit, ya)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            n_emo_ok += int((emo_logits.argmax(-1) == ye).sum())
            n_align_ok += int(
                ((torch.sigmoid(align_logit) > 0.5).float() == ya).sum()
            )
            n_seen += len(x)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"epoch {ep + 1:02d}/{n_epochs}  "
                f"loss={total / n_seen:.4f}  "
                f"emo_acc={n_emo_ok / n_seen:.3f}  "
                f"align_acc={n_align_ok / n_seen:.3f}"
            )

    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fusion_net.pt"
    torch.save(net.state_dict(), out_path)
    print(f"Saved fusion weights to {out_path}")


if __name__ == "__main__":
    train()
