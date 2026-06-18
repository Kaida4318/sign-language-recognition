import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sign Recognition",
    page_icon="🤟",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load model once (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model("sign_language_cnn.h5")

model = get_model()
labels = [chr(i) for i in range(65, 91) if chr(i) not in ["J", "Z"]]

# ----------------------------------------------------------------------
# CSS — warm, human-centered product design
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #fafaf8;
        --card: #ffffff;
        --border: #eeeef2;
        --ink: #1a1a2e;
        --slate: #6b7280;
        --indigo: #6366f1;
        --indigo-soft: #eef0fe;
        --indigo-deep: #4338ca;
        --coral: #ff8b6b;
        --coral-soft: #fff1ec;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
        background-color: var(--bg) !important;
    }
    .stApp, .stApp p, .stApp span, .stApp label {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1180px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
    [data-testid="stToolbar"] { display: none; }

    /* ---------- Hero ---------- */
    .hero-wrap {
        background: linear-gradient(135deg, var(--indigo-soft) 0%, var(--coral-soft) 100%);
        border-radius: 32px;
        padding: 3.4rem 3rem;
        margin-bottom: 2.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.7); color: var(--indigo-deep);
        font-size: 0.8rem; font-weight: 600; padding: 0.45rem 1rem;
        border-radius: 99px; margin-bottom: 1.3rem;
    }
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.9rem; font-weight: 800;
        line-height: 1.12; margin-bottom: 1rem; color: var(--ink); max-width: 680px;
    }
    .hero-sub {
        font-size: 1.08rem; color: var(--slate); max-width: 560px; line-height: 1.65; margin-bottom: 1.8rem;
    }
    .hero-cta {
        display: inline-flex; align-items: center; gap: 8px;
        background: var(--indigo); color: #ffffff; font-weight: 600; font-size: 0.95rem;
        padding: 0.8rem 1.6rem; border-radius: 14px;
    }
    .hero-illustration {
        position: absolute; right: 2.5rem; top: 50%; transform: translateY(-50%);
        font-size: 6rem; opacity: 0.9;
    }

    /* ---------- Section labels ---------- */
    .section-eyebrow {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.78rem; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--indigo); font-weight: 700; margin-bottom: 0.5rem;
    }
    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.5rem; font-weight: 700;
        color: var(--ink); margin-bottom: 1.6rem;
    }

    /* ---------- Cards via real containers ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card) !important;
        border-radius: 24px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 24px rgba(20, 20, 50, 0.04);
    }

    .card-kicker {
        display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;
    }
    .card-icon {
        width: 40px; height: 40px; border-radius: 12px; background: var(--indigo-soft);
        display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;
    }
    .card-kicker-title { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.05rem; }

    /* Uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: #fbfbfd !important;
        border: 1.5px dashed #d8d9f0 !important;
        border-radius: 18px !important;
        padding: 0.6rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--indigo) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--ink) !important; font-weight: 500 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--slate) !important; }
    [data-testid="stFileUploaderFile"] button { display: none !important; }
    [data-testid="stFileUploaderDeleteBtn"] { display: inline-flex !important; }
    [data-testid="stImage"] img { border-radius: 16px; margin-top: 0.9rem; }

    /* ---------- Result ---------- */
    .result-empty {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        color: var(--slate); font-size: 0.95rem; text-align: center; gap: 0.9rem; padding: 2.6rem 1rem;
    }
    .result-empty-icon {
        width: 60px; height: 60px; border-radius: 50%; background: var(--indigo-soft);
        display: flex; align-items: center; justify-content: center; font-size: 26px;
    }

    .result-top { display: flex; align-items: center; gap: 1.8rem; padding: 0.4rem 0 1rem 0; }
    .letter-ring-wrap { position: relative; width: 128px; height: 128px; flex-shrink: 0; }
    .letter-ring-bg {
        position: absolute; inset: 0; border-radius: 50%;
        background: conic-gradient(var(--indigo) calc(var(--pct) * 1%), var(--border) 0);
    }
    .letter-ring-inner {
        position: absolute; inset: 8px; border-radius: 50%; background: var(--card);
        display: flex; align-items: center; justify-content: center;
    }
    .letter-text {
        font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 3rem; color: var(--indigo-deep);
    }
    .result-meta-label { font-size: 0.85rem; color: var(--slate); margin-bottom: 0.3rem; }
    .result-conf { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--ink); margin-bottom: 0.5rem; }
    .result-explain {
        background: var(--indigo-soft); border-radius: 14px; padding: 0.9rem 1.1rem;
        font-size: 0.88rem; color: var(--indigo-deep); line-height: 1.5;
    }

    /* ---------- Stats grid ---------- */
    .stat-number { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--indigo-deep); }
    .stat-label { font-size: 0.82rem; color: var(--slate); margin-top: 0.3rem; line-height: 1.4; }

    /* ---------- Footer ---------- */
    .footer-wrap {
        display: flex; justify-content: space-between; align-items: center;
        padding: 1.8rem 0 0.4rem 0; border-top: 1px solid var(--border); margin-top: 1rem;
        font-size: 0.85rem; color: var(--slate);
    }
    .footer-link { color: var(--indigo-deep); font-weight: 600; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Hero section
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">🤟 AI for accessibility</div>
        <div class="hero-title">Bridging the gap between sign language and everyone else</div>
        <div class="hero-sub">Upload a photo of a hand sign and watch a neural network read it in real time —
        trained on 27,000+ images to recognize the ASL alphabet with 95.7% accuracy.</div>
        <div class="hero-cta">Try it below ↓</div>
        <div class="hero-illustration">🤚</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Recognition workspace
# ----------------------------------------------------------------------
st.markdown('<div class="section-eyebrow">Recognition workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload a hand sign to get started</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown(
            '<div class="card-kicker"><div class="card-icon">📷</div>'
            '<div class="card-kicker-title">Your photo</div></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drag and drop a hand sign photo, or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        else:
            st.markdown(
                '<div class="result-empty"><div class="result-empty-icon">🖼️</div>'
                "Your photo will appear here once uploaded</div>",
                unsafe_allow_html=True,
            )

with col_right:
    with st.container(border=True):
        st.markdown(
            '<div class="card-kicker"><div class="card-icon">✨</div>'
            '<div class="card-kicker-title">Prediction</div></div>',
            unsafe_allow_html=True,
        )

        if uploaded_file is not None:
            gray = image.convert("L").resize((28, 28))
            img_array = np.array(gray) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            with st.spinner("Analyzing hand shape..."):
                prediction = model.predict(img_array, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class] * 100)
            letter = labels[predicted_class]

            if confidence >= 80:
                explain = f"Strong match. The model is highly confident this shows the letter {letter}."
            elif confidence >= 50:
                explain = f"Likely a match for {letter}, though the hand shape shares some features with similar letters."
            else:
                explain = f"Low confidence. This may not be a clear ASL hand sign, or the photo could be unclear."

            st.markdown(
                f"""
                <div class="result-top">
                    <div class="letter-ring-wrap">
                        <div class="letter-ring-bg" style="--pct:{confidence:.0f};"></div>
                        <div class="letter-ring-inner"><div class="letter-text">{letter}</div></div>
                    </div>
                    <div style="flex:1;">
                        <div class="result-meta-label">Predicted letter</div>
                        <div class="result-conf">{confidence:.1f}% confident</div>
                    </div>
                </div>
                <div class="result-explain">💡 {explain}</div>
                """,
                unsafe_allow_html=True,
            )

            # Probability chart for top candidates
            order = np.argsort(prediction)[::-1][:6]
            chart_df = pd.DataFrame(
                {"Letter": [labels[i] for i in order], "Probability": [float(prediction[i] * 100) for i in order]}
            ).set_index("Letter")
            st.markdown('<div style="margin-top:1.4rem; font-size:0.85rem; color:#6b7280; font-weight:600;">Top candidates</div>', unsafe_allow_html=True)
            st.bar_chart(chart_df, color="#6366F1", height=180)
        else:
            st.markdown(
                '<div class="result-empty"><div class="result-empty-icon">🔮</div>'
                "Upload a photo on the left to see the predicted letter</div>",
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# About the model
# ----------------------------------------------------------------------
st.markdown('<div class="section-eyebrow" style="margin-top:2.6rem;">About the model</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">A convolutional neural network, trained from scratch</div>', unsafe_allow_html=True)

stat_cols = st.columns(4)
stats = [
    ("95.7%", "Test accuracy (CNN)"),
    ("72.7%", "Baseline dense network accuracy"),
    ("24", "Letters recognized — excludes J & Z"),
    ("27,455", "Training images"),
]
for col, (number, label) in zip(stat_cols, stats):
    with col:
        with st.container(border=True):
            st.markdown(
                f'<div class="stat-number">{number}</div><div class="stat-label">{label}</div>',
                unsafe_allow_html=True,
            )

with st.container(border=True):
    st.markdown(
        '<div class="card-kicker"><div class="card-icon">🧠</div>'
        '<div class="card-kicker-title">Architecture summary</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style="color:#6b7280; line-height:1.7; font-size:0.92rem;">
        Two convolutional layers (32 and 64 filters) extract shape patterns from each image, each followed
        by max-pooling to reduce size while keeping the strongest signal. A dense layer with dropout combines
        these features before a final 24-way softmax layer outputs a probability for each letter. Compared to
        a plain feedforward network, this architecture generalizes far better to new images — visible directly
        in the accuracy gap above.
        </p>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-wrap">
        <div>Built with TensorFlow &amp; Streamlit · Muhammad Zubairu Rabiu</div>
        <a class="footer-link" href="https://github.com/Kaida4318" target="_blank">View on GitHub →</a>
    </div>
    """,
    unsafe_allow_html=True,
)