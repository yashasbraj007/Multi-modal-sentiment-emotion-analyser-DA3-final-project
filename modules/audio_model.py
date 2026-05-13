"""Whisper-based speech-to-text (optional third modality)."""
from __future__ import annotations

import torch
from transformers import pipeline

WHISPER_MODEL_ID = "openai/whisper-tiny"


class AudioTranscriber:
    def __init__(self, device: str | None = None) -> None:
        device_idx = 0 if (device == "cuda" or torch.cuda.is_available()) else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_ID,
            device=device_idx,
        )

    def transcribe(self, audio_path: str) -> str:
        result = self.pipe(audio_path, return_timestamps=False)
        return str(result["text"]).strip()
