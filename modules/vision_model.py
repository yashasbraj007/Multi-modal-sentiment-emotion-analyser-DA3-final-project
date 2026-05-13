"""Vision Transformer wrapper for facial-emotion classification.

Uses the publicly available `trpakov/vit-face-expression` checkpoint
(ViT-base fine-tuned on FER-2013-style data) which classifies a face image
into seven Ekman-style categories: angry, disgust, fear, happy, neutral,
sad, surprise.

The class exposes:
    .predict(image)              -> probability dict + top label
    .forward_with_attention(...) -> raw outputs incl. attention tensors
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

VIT_MODEL_ID = "trpakov/vit-face-expression"


class VisionEmotionModel:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID)
        # output_attentions=True so we can build Grad-CAM-style heat-maps later.
        self.model = (
            AutoModelForImageClassification.from_pretrained(
                VIT_MODEL_ID, output_attentions=True
            )
            .to(self.device)
            .eval()
        )
        self.id2label: dict[int, str] = self.model.config.id2label

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Return top label, confidence and full probability distribution."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        idx = int(probs.argmax())
        prob_dict = {self.id2label[i].lower(): float(p) for i, p in enumerate(probs)}
        return {
            "label": self.id2label[idx].lower(),
            "confidence": float(probs[idx]),
            "probs": prob_dict,
        }

    @torch.no_grad()
    def forward_with_attention(self, image: Image.Image):
        """Forward pass returning the full ModelOutput including attentions."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs, inputs
