# Architecture & Design Decisions

This document is the **"technical depth"** companion to the README

## 1. Model choices

### 1.1 Vision — `trpakov/vit-face-expression`
- **Family:** Vision Transformer (ViT-base, ~86 M parameters).
- **Training data:** FER-2013-style facial expression dataset.
- **Output:** seven Ekman categories (angry, disgust, fear, happy, neutral, sad, surprise).
- **Why ViT, not a CNN?** The assignment allowed either, and ViT offers
  *attention maps for free*, which we exploit in `visualization.py` to
  build the Grad-CAM-style explainability view. A ResNet-18 emotion
  classifier would have been smaller but would have required adding
  Grad-CAM machinery on top.

### 1.2 Text — `j-hartmann/emotion-english-distilroberta-base`
- **Family:** DistilRoBERTa fine-tuned on six emotion datasets (~67 M parameters).
- **Output:** seven labels that *almost* match the vision label space
  (`joy ↔ happy`, `anger ↔ angry`, `sadness ↔ sad`). Reconciled via
  `TEXT_TO_CANONICAL` in `modules/utils.py`.
- **Why not just a positive/negative sentiment model?** We need the full
  emotion distribution so that (a) the fusion vector is informative and
  (b) the user can read the bar chart with fine-grained categories.

### 1.3 Audio — `openai/whisper-tiny`
- 39 M parameters; runs in real time on CPU; English-capable.
- Audio is **not a parallel modality** — it feeds the text channel via
  transcription. This is the simplest faithful integration of audio
  given the time budget.

### 1.4 Generative — `google/flan-t5-base`
- 250 M parameters, instruction-tuned encoder–decoder.
- Chosen over GPT-2 because FLAN-T5 follows explicit instructions
  ("Summarise in two sentences…") much more reliably than a
  free-running causal LM. The course "Train your own GPT-2" content
  is therefore extended by demonstrating the *next* generation of
  instruction-tuned LMs.

---

## 2. The fusion layer

We adopt a **hybrid** scheme: a transparent rule-based score *plus* a
small learned MLP.

### 2.1 Rule-based agreement
```
align_rule = 0.5 * cos(v_vec, t_vec) + 0.5 * 1[polarity(v) == polarity(t)]
```
- Cosine similarity captures geometric agreement (e.g. both vectors
  spiked on the same class).
- Polarity match is a categorical sanity check — *"do positive words
  match a positive face?"*

### 2.2 Learned `FusionNet`
A tiny MLP with two heads:

```
input  : 14-d (concat of vision + text 7-d distributions)
shared : 14 -> 32 -> 16 (ReLU)
heads  : emotion_logits[7], alignment_logit[1]
```

### 2.3 Synthetic training data
We cannot train on a labelled real multimodal dataset within the
time budget, so we synthesise 8 000 samples that *faithfully model the
problem*:

1. Sample a "true emotion" `e` uniformly.
2. Build the visual probability vector `v` as a Dirichlet-style spike
   around `e`. The face usually leaks the underlying affect.
3. With probability `0.35`, replace the *textual* vector `t` with a
   spike around a **different polarity** emotion — this is precisely
   the "I'm fine while looking sad" case the brief asks us to detect.
4. Otherwise spike `t` around `e` as well.

Targets:
- `emotion_label = e` (the face usually wins on conflicts).
- `alignment_label = 1` if both vectors share the same polarity, else 0.

Training converges in ~30 s on CPU; final synthetic alignment accuracy
is typically ≥ 0.90.

### 2.4 Blending
```
fused_probs = 0.5 * learned + 0.25 * v + 0.25 * t
alignment   = 0.5 * learned_alignment + 0.5 * rule_alignment
mismatch    = alignment < 0.5
```
The blend keeps the system robust if the learned net is ever badly
calibrated.

### 2.5 Graceful degradation
If `models/fusion_net.pt` is missing the module silently falls back to
the rule-based score, so a freshly-cloned repo without running
`train_fusion.py` still produces a sensible answer.

---

## 3. Explainability

| Modality | Technique | Why |
|---|---|---|
| Vision | **Attention rollout** (Abnar & Zuidema, 2020) | Combines attention from every layer with a residual-aware identity term — much more faithful than raw last-layer attention. |
| Text | Last-layer CLS attention | Cheap, intuitive ("which words did the classifier look at?") and renders neatly as token shading. |

Both are wired into the Streamlit UI behind an *Attention* expander to
keep the main view clean.

---

## 4. Challenges & solutions

| Challenge | Solution |
|---|---|
| Vision and text models use slightly different labels (`joy` vs `happy`, `anger` vs `angry`). | Canonical mapping in `utils.py`, applied inside the fusion module. |
| No labelled multimodal dataset for the fusion-net. | Synthetic data generator in `train_fusion.py` that *encodes the problem* (polarity-swapped pairs). |
| FLAN-T5 occasionally produces a stub like *"yes."*. | Length check → fall back to a deterministic template appended after. |
| Streamlit reloads models on every interaction. | `@st.cache_resource` for all five models. |
| Raw last-layer attention is misleading for ViT. | **Attention rollout** across every layer with residual identity term. |
| First-run model downloads (~1 GB) feel slow. | Documented in the README; spinners with descriptive messages in `app.py`. |
| Whisper needs FFmpeg for some audio formats. | We stick to `.wav` / `.mp3` / `.m4a` which `librosa` handles directly; the requirements pin `librosa` + `soundfile`. |

---

## 5. Extensibility

- **Adding a fourth modality** (e.g. body pose) becomes a matter of
  appending another 7-d emotion vector to the fusion network's input
  and re-training the MLP — the rest of the pipeline is agnostic.
- **Real-time video** can be plugged in by feeding successive webcam
  frames through `vision_model.predict` and plotting the timeline of
  emotion probabilities — `app.py` is the only file that needs to
  change.
- **Better fusion** could use cross-attention over per-modality
  embeddings, but for our 7-d categorical setting an MLP is sufficient
  and far more interpretable.
