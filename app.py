"""
MoodSyncAI -- Multi-Modal Sentiment & Emotion Analyser
=====================================================
Streamlit entry point.

Pipeline
--------
1. Vision  : Vision Transformer (ViT) -> 7-class facial emotion distribution.
2. Text    : DistilRoBERTa -> 7-class verbal emotion + polarity sentiment.
3. Audio   : (optional) Whisper -> transcript -> feeds the text channel.
4. Fusion  : Rule-based agreement score + trained `FusionNet` MLP -> unified
             emotion, alignment in [0, 1] and a binary mismatch flag.
5. Summary : FLAN-T5 generates a short, sensitive natural-language report.
"""
from __future__ import annotations

import tempfile

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from modules.audio_model import AudioTranscriber
from modules.fusion import FusionModule
from modules.generator import SummaryGenerator
from modules.text_model import TextEmotionModel
from modules.utils import EMOTION_COLORS
from modules.vision_model import VisionEmotionModel
from modules.visualization import (
    visualise_text_attention,
    visualise_vision_attention,
)

st.set_page_config(page_title="MoodSyncAI", page_icon="🎭", layout="wide")


# --------------------------------------------------------------------------- #
# Cached model loaders -- prevent re-instantiation on every interaction       #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading vision model …")
def load_vision_model() -> VisionEmotionModel:
    return VisionEmotionModel()


@st.cache_resource(show_spinner="Loading text model …")
def load_text_model() -> TextEmotionModel:
    return TextEmotionModel()


@st.cache_resource(show_spinner="Loading audio model …")
def load_audio_model() -> AudioTranscriber:
    return AudioTranscriber()


@st.cache_resource(show_spinner="Loading fusion module …")
def load_fusion_module() -> FusionModule:
    return FusionModule()


@st.cache_resource(show_spinner="Loading summary generator …")
def load_generator() -> SummaryGenerator:
    return SummaryGenerator()


def bar_chart(probs: dict, title: str) -> go.Figure:
    """Plotly bar chart from a {label: prob} dict, colour-coded per emotion."""
    labels = list(probs.keys())
    values = [probs[k] * 100 for k in labels]
    colors = [EMOTION_COLORS.get(k.lower(), "#888") for k in labels]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title="Confidence (%)",
        height=350,
        yaxis_range=[0, 110],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --------------------------------------------------------------------------- #
# UI                                                                          #
# --------------------------------------------------------------------------- #
st.title("🎭 MoodSyncAI")
st.markdown(
    "**Multi-modal sentiment & emotion analyser** — combines a Vision "
    "Transformer for facial emotion, a RoBERTa-based text model for verbal "
    "sentiment, a learned fusion layer for mismatch detection, and a "
    "generative model for plain-language explanation."
)

with st.sidebar:
    st.header("⚙️ Settings")
    use_audio = st.checkbox("Use audio input (Whisper)", value=False)
    show_attention = st.checkbox("Show attention visualisations", value=True)
    st.divider()
    st.markdown(
        "**Pipeline**\n"
        "1. Vision · ViT face expression\n"
        "2. Text · DistilRoBERTa emotion\n"
        "3. Audio · Whisper (optional)\n"
        "4. Fusion · Rule + learned MLP\n"
        "5. Generation · FLAN-T5\n"
    )
    st.divider()
    st.caption("DA3 Final Project · MoodSyncAI")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1️⃣  Face image")
    img_source = st.radio("Source", ["Upload", "Webcam"], horizontal=True)
    image: Image.Image | None = None
    if img_source == "Upload":
        upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if upload is not None:
            image = Image.open(upload).convert("RGB")
    else:
        snap = st.camera_input("Take a photo")
        if snap is not None:
            image = Image.open(snap).convert("RGB")
    if image is not None:
        st.image(image, caption="Input image", use_container_width=True)

with col2:
    st.subheader("2️⃣  What the person said")
    if use_audio:
        audio_file = st.file_uploader(
            "Upload a short audio clip (wav/mp3/m4a)", type=["wav", "mp3", "m4a"]
        )
        transcribed = ""
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix="_" + audio_file.name
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            with st.spinner("Transcribing audio …"):
                transcribed = load_audio_model().transcribe(tmp_path)
            st.success("Transcript ready.")
        text = st.text_area(
            "Transcript (auto-filled, editable)", value=transcribed, height=140
        )
    else:
        text = st.text_area(
            "Type or paste what the person said",
            value="No, I think the project is going really well.",
            height=140,
        )

st.divider()
analyse = st.button("🔍  Analyse", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Inference + results                                                         #
# --------------------------------------------------------------------------- #
if analyse:
    if image is None or not text.strip():
        st.error("Please provide both an image and text.")
        st.stop()

    vision = load_vision_model()
    text_m = load_text_model()
    fusion = load_fusion_module()
    gen = load_generator()

    with st.spinner("Running multi-modal analysis …"):
        v_out = vision.predict(image)
        t_out = text_m.predict(text)
        f_out = fusion.fuse(v_out, t_out)
        summary = gen.generate(v_out, t_out, f_out, text)

    # ------------------------------------------------------------------ vision
    st.subheader("👁️  Visual emotion")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            bar_chart(v_out["probs"], "Facial-emotion confidences"),
            use_container_width=True,
        )
    with c2:
        st.metric(
            "Top emotion",
            v_out["label"].capitalize(),
            f"{v_out['confidence'] * 100:.1f}%",
        )
        if show_attention:
            with st.expander("🔥  Attention heat-map (rollout)"):
                heat = visualise_vision_attention(vision, image)
                st.image(heat, use_container_width=True)

    # -------------------------------------------------------------------- text
    st.subheader("💬  Textual sentiment")
    c3, c4 = st.columns([2, 1])
    with c3:
        st.plotly_chart(
            bar_chart(t_out["probs"], "Text-emotion confidences"),
            use_container_width=True,
        )
    with c4:
        st.metric(
            "Sentiment",
            t_out["sentiment"].capitalize(),
            f"{t_out['sentiment_confidence'] * 100:.1f}%",
        )
        st.metric(
            "Top emotion",
            t_out["label"].capitalize(),
            f"{t_out['confidence'] * 100:.1f}%",
        )
        if show_attention:
            with st.expander("🔍  Token attention"):
                attn_html = visualise_text_attention(text_m, text)
                st.markdown(attn_html, unsafe_allow_html=True)

    # ------------------------------------------------------------------ fusion
    st.subheader("  Fusion result")
    if f_out["mismatch"]:
        badge = "🟠  MISMATCH DETECTED"
        badge_color = "orange"
    else:
        badge = "🟢  ALIGNED"
        badge_color = "green"

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(
        "Fused emotion",
        f_out["fused_label"].capitalize(),
        f"{f_out['fused_confidence'] * 100:.1f}%",
    )
    cc2.metric("Alignment score", f"{f_out['alignment'] * 100:.1f}%")
    cc3.markdown(
        f"<h3 style='color:{badge_color};margin-top:0.5rem'>{badge}</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Vision polarity: **{f_out['vision_polarity']}** · "
        f"Text polarity: **{f_out['text_polarity']}** · "
        f"Trained fusion network used: "
        f"**{'yes' if f_out['trained_fusion_used'] else 'no (rule-only)'}**"
    )

    # ----------------------------------------------------------- generative
    st.subheader("📝  Generative summary")
    st.info(summary)

    with st.expander("🛠️  Raw outputs (debug)"):
        st.json({"vision": v_out, "text": t_out, "fusion": f_out})
