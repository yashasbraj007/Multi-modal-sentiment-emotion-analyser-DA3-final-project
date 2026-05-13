#  MoodSyncAI — Multi-Modal Sentiment & Emotion Analyser

> **Course:** Data Analytics 3 — Deep Learning & GenAI
> **Module:** Final Project
> **Instructor:** Prof. Dr. Gayan de Silva

MoodSyncAI combines facial-expression analysis, verbal-sentiment analysis,
a learned fusion layer and a generative language model to produce a unified
emotional assessment of a person in a given moment — including detection of
**mismatches** between what someone *says* and what their face *shows*.

---

##  Features

| Stage | Model | Library |
|---|---|---|
| Facial emotion | `trpakov/vit-face-expression` (Vision Transformer) |  transformers |
| Verbal emotion + sentiment | `j-hartmann/emotion-english-distilroberta-base` |  transformers |
| Audio (optional) | `openai/whisper-tiny` |  transformers |
| Fusion | Rule-based + trained `FusionNet` MLP | PyTorch |
| Generative summary | `google/flan-t5-base` (with template fallback) |  transformers |
| UI | Streamlit (with built-in webcam) | streamlit |

### Extended (bonus) features implemented
-  **Audio input** via Whisper — third modality, auto-transcribed into the text channel.
-  **Learned fusion network** — not just a weighted average. Small MLP trained on a synthetic dataset that explicitly models incongruent affect.
-  **Attention visualisation** — attention-rollout heat-map for the ViT and token-level attention shading for the text Transformer.
-  **Webcam capture** — face image straight from the camera, no upload needed.
-  **Bar charts across all categories** with colour-coded emotions.



# 1. Clone or unzip the project
cd moodsync-ai

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# If torch is slow to install on a CPU-only machine, you can pre-install:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Train the small fusion network (~30 seconds on CPU)
python train_fusion.py

# 5. Run the app
streamlit run app.py
```

The app opens at <http://localhost:8501>. On first launch the
HuggingFace models (~1 GB total) are downloaded and cached locally; this
takes 3–10 minutes. After that, everything runs offline.

### VSCode tips
- Install the **Python** extension.
- Select the `.venv` interpreter via `Ctrl+Shift+P` → *Python: Select Interpreter*.
- Run with the terminal command above (Streamlit picks up edits automatically).

---

##  Demo script (5-minute live demo)

1. Open the app at `http://localhost:8501`.
2. Upload the photo of a colleague (or use the webcam).
3. Type the sentence:
   *"No, I think the project is going really well."*
4. Click **Analyse**. Walk through:
   - Visual emotion bar-chart → top emotion *Sad* / *Fearful* ~ 68 %
   - Text emotion bar-chart → top sentiment *Positive* ~ 81 %
   -  **MISMATCH DETECTED** badge
   - Generative summary explaining the incongruence
5. Open the **Attention heat-map** expander to show explainability.
6. (Optional) Toggle audio input and upload a short `.wav` clip — Whisper transcribes it and the same pipeline runs.


##  Project layout

```
moodsync-ai/
├── app.py                      # Streamlit UI
├── train_fusion.py             # Trains the fusion MLP (synthetic data)
├── modules/
│   ├── __init__.py
│   ├── vision_model.py         # ViT wrapper (+ attention export)
│   ├── text_model.py           # DistilRoBERTa wrapper (+ attention)
│   ├── audio_model.py          # Whisper wrapper
│   ├── fusion.py               # Rule + learned fusion
│   ├── generator.py            # FLAN-T5 summary + safe fallback
│   ├── visualization.py        # Attention rollout & token heat-map
│   └── utils.py                # Canonical mapping & colour palette
├── models/
│   └── fusion_net.pt           # Created by train_fusion.py
├── requirements.txt
├── README.md
└── ARCHITECTURE.md

