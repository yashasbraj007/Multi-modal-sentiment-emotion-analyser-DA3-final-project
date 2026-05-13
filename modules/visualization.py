"""Attention-based visualisations for the vision and text Transformers.

For the ViT we implement **attention rollout** (Abnar & Zuidema, 2020):
the attention matrices of every layer are multiplied together, with an
identity term added to account for the residual stream.  The first row of
the resulting matrix gives the contribution of every patch to the [CLS]
token, which we reshape back to a 14x14 grid and overlay on the image.

For the text RoBERTa we simply take the last-layer attention from the
[CLS] token to all other tokens and shade each token by that weight.
This is a faithful approximation of "what did the classifier look at?"
without the cost of full integrated gradients.
"""
from __future__ import annotations

import html
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# ViT attention rollout                                                       #
# --------------------------------------------------------------------------- #
def _attention_rollout(attentions) -> np.ndarray:
    """Return the CLS-row attention rollout as a 1-D numpy array (one weight
    per spatial patch, CLS itself excluded)."""
    # attentions is a tuple of (1, num_heads, T, T) tensors, one per layer.
    result = torch.eye(attentions[0].size(-1))
    for attn in attentions:
        attn_heads = attn.mean(dim=1)[0]  # avg over heads -> (T, T)
        # Add identity for residual stream and re-normalise rows.
        attn_heads = attn_heads + torch.eye(attn_heads.size(0))
        attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
        result = attn_heads @ result
    mask = result[0, 1:]  # CLS -> patches
    return mask.cpu().numpy()


def visualise_vision_attention(vision_model, image: Image.Image) -> Image.Image:
    outputs, _ = vision_model.forward_with_attention(image)
    if not outputs.attentions:
        return image

    mask = _attention_rollout(outputs.attentions)
    side = int(np.sqrt(mask.shape[0]))
    mask = mask.reshape(side, side)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)

    img_np = np.asarray(image.resize((224, 224))).astype(np.float32) / 255.0
    mask_up = (
        np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
        )
        / 255.0
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_np)
    ax.imshow(mask_up, cmap="jet", alpha=0.45)
    ax.axis("off")
    ax.set_title("Attention rollout (ViT)")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# --------------------------------------------------------------------------- #
# Text token-level attention                                                  #
# --------------------------------------------------------------------------- #
def visualise_text_attention(text_model, text: str) -> str:
    """Return an HTML string in which each token is shaded by the CLS token's
    last-layer attention weight."""
    outputs, inputs = text_model.forward_with_attention(text)
    if not outputs.attentions:
        return html.escape(text)

    last = outputs.attentions[-1][0].mean(0)  # avg heads -> (T, T)
    # CLS row, drop CLS and SEP.
    cls_attn = last[0, 1:-1].cpu().numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-9)

    token_ids = inputs["input_ids"][0][1:-1]
    tokens = text_model.tokenizer.convert_ids_to_tokens(token_ids)

    spans = []
    for tok, w in zip(tokens, cls_attn):
        alpha = float(w)
        clean = tok.replace("Ġ", " ").replace("▁", " ")
        spans.append(
            f"<span style='background:rgba(231,76,60,{alpha:.2f});"
            f"padding:2px 1px;border-radius:3px;'>{html.escape(clean)}</span>"
        )
    return (
        "<div style='line-height:2.0;font-family:monospace;font-size:15px;'>"
        + "".join(spans)
        + "</div>"
    )
