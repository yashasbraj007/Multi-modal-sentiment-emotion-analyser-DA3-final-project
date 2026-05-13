"""Shared constants, label mappings and colour palette.

The vision and text models we use have *slightly* different label spaces
(`happy` vs `joy`, `sad` vs `sadness`, `angry` vs `anger`).  We define a
single canonical vocabulary here and provide forward maps.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Canonical seven-class emotion vocabulary
# ----------------------------------------------------------------------------
CANONICAL_EMOTIONS: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# Vision model labels are already canonical.
VISION_TO_CANONICAL: dict[str, str] = {e: e for e in CANONICAL_EMOTIONS}

# Text model (j-hartmann) uses different surface forms.
TEXT_TO_CANONICAL: dict[str, str] = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}

# Coarse polarity bucket used for the rule-based fusion score.
EMOTION_POLARITY: dict[str, str] = {
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fear": "negative",
    "disgust": "negative",
}

# Colour palette — covers BOTH naming conventions so bar charts work
# regardless of which model produced the dict.
EMOTION_COLORS: dict[str, str] = {
    "angry": "#e74c3c",
    "anger": "#e74c3c",
    "disgust": "#27ae60",
    "fear": "#8e44ad",
    "happy": "#f1c40f",
    "joy": "#f1c40f",
    "neutral": "#95a5a6",
    "sad": "#3498db",
    "sadness": "#3498db",
    "surprise": "#e67e22",
}
