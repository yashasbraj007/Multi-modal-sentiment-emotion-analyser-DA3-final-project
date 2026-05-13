"""Multimodal fusion layer.

The fusion layer is the heart of MoodSyncAI.  It takes two 7-dimensional
emotion probability vectors -- one from vision, one from text -- and produces

    * a unified emotion label,
    * an alignment score in [0, 1],
    * a binary "mismatch detected" flag,
    * the fused probability distribution itself.

Two scoring paths are computed and blended:

1. **Rule-based agreement** -- cosine similarity between the two probability
   vectors combined with a polarity-match indicator.  Cheap, transparent,
   and always available.

2. **Learned `FusionNet`** -- a small MLP (14 -> 32 -> 16 -> {7 emotions,
   1 alignment scalar}) trained on a synthetic dataset that explicitly
   models incongruent affect (e.g. saying "I'm fine" while looking sad).
   See `train_fusion.py`.

If the trained weights are not present on disk we transparently fall back
to the rule-based score only, so the application still runs end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modules.utils import (
    CANONICAL_EMOTIONS,
    EMOTION_POLARITY,
    TEXT_TO_CANONICAL,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FUSION_WEIGHTS = MODELS_DIR / "fusion_net.pt"


class FusionNet(nn.Module):
    """Tiny MLP that fuses two 7-dim emotion distributions.

    Input  : 14-d vector  (concat of vision + text probabilities)
    Output : (emotion_logits[7], alignment_logit[1])
    """

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.emotion_head = nn.Linear(16, 7)
        self.alignment_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.emotion_head(h), self.alignment_head(h).squeeze(-1)


class FusionModule:
    def __init__(self) -> None:
        self.net = FusionNet()
        if FUSION_WEIGHTS.exists():
            self.net.load_state_dict(torch.load(FUSION_WEIGHTS, map_location="cpu"))
            self.trained = True
        else:
            self.trained = False
        self.net.eval()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_canonical_text(probs: dict) -> dict:
        """Re-key the text model's probs to the canonical vocabulary."""
        out = {e: 0.0 for e in CANONICAL_EMOTIONS}
        for k, v in probs.items():
            out[TEXT_TO_CANONICAL[k]] += v
        return out

    @staticmethod
    def _vector(canon_probs: dict) -> np.ndarray:
        return np.array([canon_probs[e] for e in CANONICAL_EMOTIONS], dtype=np.float32)

    # ------------------------------------------------------------------ core
    def fuse(self, vision_out: dict, text_out: dict) -> dict:
        text_canon = self._to_canonical_text(text_out["probs"])
        v_vec = self._vector(vision_out["probs"])
        t_vec = self._vector(text_canon)

        # --- Rule-based alignment ----------------------------------------
        cos = float(
            np.dot(v_vec, t_vec)
            / (np.linalg.norm(v_vec) * np.linalg.norm(t_vec) + 1e-9)
        )
        v_polarity = EMOTION_POLARITY[vision_out["label"]]
        t_polarity = text_out["sentiment"]
        polarity_match = 1.0 if v_polarity == t_polarity else 0.0
        rule_alignment = 0.5 * cos + 0.5 * polarity_match

        # --- Learned head -------------------------------------------------
        if self.trained:
            with torch.no_grad():
                x = torch.from_numpy(np.concatenate([v_vec, t_vec])).unsqueeze(0)
                emo_logits, align_logit = self.net(x)
                learned_probs = torch.softmax(emo_logits[0], dim=-1).numpy()
                learned_alignment = float(torch.sigmoid(align_logit[0]))
            fused_probs = 0.5 * learned_probs + 0.25 * v_vec + 0.25 * t_vec
            alignment = 0.5 * learned_alignment + 0.5 * rule_alignment
        else:
            fused_probs = 0.5 * v_vec + 0.5 * t_vec
            alignment = rule_alignment

        fused_idx = int(fused_probs.argmax())
        fused_label = CANONICAL_EMOTIONS[fused_idx]
        fused_confidence = float(fused_probs[fused_idx])
        mismatch = alignment < 0.5

        return {
            "fused_label": fused_label,
            "fused_confidence": fused_confidence,
            "fused_probs": {
                e: float(p) for e, p in zip(CANONICAL_EMOTIONS, fused_probs)
            },
            "alignment": float(alignment),
            "rule_alignment": float(rule_alignment),
            "mismatch": bool(mismatch),
            "trained_fusion_used": self.trained,
            "vision_polarity": v_polarity,
            "text_polarity": t_polarity,
        }
