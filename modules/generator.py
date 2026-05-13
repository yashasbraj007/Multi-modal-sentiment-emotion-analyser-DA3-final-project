"""Generative natural-language summary.

We use Google's instruction-tuned FLAN-T5-base, which is small enough to run
comfortably on CPU but already understands instructions such as "summarise
the speaker's emotional state in two sentences".

Robustness: if the model cannot be loaded (offline / disk full) we fall back
to a deterministic template -- the demo therefore never crashes.  We also
post-check the generated length and append the template if FLAN gave us a
suspiciously short answer.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GEN_MODEL_ID = "google/flan-t5-base"


def _template(vision_out: dict, text_out: dict, fusion_out: dict, text: str) -> str:
    v_label = vision_out["label"]
    v_conf = vision_out["confidence"] * 100
    t_label = text_out["sentiment"]
    t_conf = text_out["sentiment_confidence"] * 100
    if fusion_out["mismatch"]:
        return (
            f"Despite expressing a {t_label} sentiment verbally "
            f"({t_conf:.0f}% confidence) — '{text.strip()}' — the speaker's "
            f"facial cues read as {v_label} ({v_conf:.0f}% confidence). "
            f"This incongruence between word and expression suggests "
            f"underlying discomfort worth acknowledging in the conversation."
        )
    return (
        f"The speaker's words and facial expression are consistent: a "
        f"{t_label} verbal tone matches a {v_label} expression. "
        f"There is no detectable affective mismatch."
    )


class SummaryGenerator:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
            self.model = (
                AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID)
                .to(self.device)
                .eval()
            )
            self.available = True
        except Exception as exc:  # noqa: BLE001
            print(f"[SummaryGenerator] Falling back to template only: {exc}")
            self.available = False

    @torch.no_grad()
    def generate(
        self,
        vision_out: dict,
        text_out: dict,
        fusion_out: dict,
        text: str,
    ) -> str:
        if not self.available:
            return _template(vision_out, text_out, fusion_out, text)

        prompt = (
            "You are an empathetic communication coach. In two short "
            "sentences, summarise the speaker's emotional state and "
            "explicitly note any mismatch between their words and face. "
            f'They said: "{text.strip()}". '
            f"Verbal sentiment: {text_out['sentiment']} "
            f"({text_out['sentiment_confidence'] * 100:.0f}%). "
            f"Facial expression: {vision_out['label']} "
            f"({vision_out['confidence'] * 100:.0f}%). "
            f"Alignment score: {fusion_out['alignment']:.2f}. "
            f"Mismatch detected: "
            f"{'yes' if fusion_out['mismatch'] else 'no'}."
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        generated = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # Safety net: if the generation collapsed to a stub, append the
        # deterministic template so the user always sees a useful summary.
        if len(generated.split()) < 12:
            generated = (
                f"{generated.rstrip('.')} — " + _template(
                    vision_out, text_out, fusion_out, text
                )
            )
        return generated
