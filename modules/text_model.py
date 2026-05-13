"""Text emotion + polarity sentiment classifier.

Wraps `j-hartmann/emotion-english-distilroberta-base`, a DistilRoBERTa model
fine-tuned on six emotion datasets that outputs seven labels:
    anger, disgust, fear, joy, neutral, sadness, surprise.

In addition to the fine-grained distribution we derive a coarse polarity
bucket {positive, negative, neutral} by summing the probabilities of all
emotions that share the same polarity (see `EMOTION_POLARITY`).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# Map fine-grained emotion -> polarity bucket.
EMOTION_POLARITY = {
    "joy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "disgust": "negative",
}


class TextEmotionModel:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                EMOTION_MODEL_ID, output_attentions=True
            )
            .to(self.device)
            .eval()
        )
        self.id2label: dict[int, str] = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1).cpu().numpy()
        idx = int(probs.argmax())
        prob_dict = {self.id2label[i].lower(): float(p) for i, p in enumerate(probs)}

        # Aggregate to polarity buckets.
        polarity_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for emo, p in prob_dict.items():
            polarity_scores[EMOTION_POLARITY.get(emo, "neutral")] += p
        sentiment = max(polarity_scores, key=polarity_scores.get)

        return {
            "label": self.id2label[idx].lower(),
            "confidence": float(probs[idx]),
            "probs": prob_dict,
            "sentiment": sentiment,
            "sentiment_confidence": float(polarity_scores[sentiment]),
            "sentiment_scores": polarity_scores,
        }

    @torch.no_grad()
    def forward_with_attention(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs, inputs
