import streamlit as st
import numpy as np
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
# CSS — light warm background, rounded teal cards
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #f6f5f1;
        --bg-soft: #eef2f0;
        --teal: #1d9e75;
        --teal-dark: #085041;
        --teal-light: #e1f5ee;
        --teal-mid: #5dcaa5;
        --text: #1f2420;
        --text-dim: #6b716c;
        --border: #e3e1da;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
        background-color: var(--bg) !important;
    }

    .stApp, .stApp p, .stApp span, .stApp label {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }
    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1100px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
    [data-testid="stToolbar"] { display: none; }

    /* Real Streamlit containers, styled to look like rounded cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 28px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 2px 14px rgba(0,0,0,0.03);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.4rem 0.5rem;
    }

    .hero-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--teal-light); color: var(--teal-dark);
        font-size: 0.78rem; font-weight: 600; padding: 0.4rem 0.9rem;
        border-radius: 99px; margin-bottom: 1.1rem;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif; font-size: 2.6rem; font-weight: 600;
        line-height: 1.15; margin-bottom: 0.8rem; color: var(--text);
    }
    .hero-sub { font-size: 1.02rem; color: var(--text-dim); max-width: 600px; line-height: 1.6; margin-bottom: 2.2rem; }

    .card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 0.9rem; }
    .card-icon {
        width: 36px; height: 36px; border-radius: 50%; background: var(--teal-light);
        display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 16px;
    }
    .card-title { font-family: 'Poppins', sans-serif; font-size: 1.02rem; font-weight: 600; }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-soft) !important;
        border: 1.5px dashed #c9d6cf !important;
        border-radius: 22px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--teal) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text) !important; font-weight: 500 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--text-dim) !important; }

    [data-testid="stFileUploaderFile"] button {
        display: none !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] {
        display: inline-flex !important;
    }
    [data-testid="stFileUploaderFile"] {
        gap: 0 !important;
    }

    [data-testid="stImage"] img { border-radius: 16px; margin-top: 0.8rem; }

    .pred-empty {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        color: var(--text-dim); font-size: 0.92rem; text-align: center; gap: 0.8rem;
        padding: 2.4rem 1rem;
    }
    .pred-empty-icon {
        width: 56px; height: 56px; border-radius: 50%; background: var(--bg-soft);
        display: flex; align-items: center; justify-content: center; font-size: 24px;
    }

    .pred-active { display: flex; flex-direction: column; align-items: center; padding: 0.6rem 0 0.4rem; }
    .letter-badge {
        width: 140px; height: 140px; border-radius: 50%;
        background: var(--teal-light);
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 1.1rem;
    }
    .letter-text { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 4.3rem; color: var(--teal-dark); line-height: 1; }
    .conf-text { font-size: 1.05rem; font-weight: 600; color: var(--text); margin-bottom: 0.8rem; }
    .confidence-track { background: var(--bg-soft); border-radius: 99px; height: 10px; width: 80%; overflow: hidden; margin: 0 auto; }
    .confidence-fill { background: var(--teal-mid); height: 100%; border-radius: 99px; }
    .rank-strip { display: flex; gap: 10px; margin-top: 1.4rem; justify-content: center; }
    .rank-chip { background: var(--bg-soft); border-radius: 16px; padding: 0.55rem 0.95rem; text-align: center; min-width: 60px; }
    .rank-chip-letter { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 1.05rem; }
    .rank-chip-pct { font-size: 0.74rem; color: var(--text-dim); margin-top: 2px; }

    .stat-number { font-family: 'Poppins', sans-serif; font-size: 1.8rem; font-weight: 600; color: var(--teal-dark); }
    .stat-label { font-size: 0.8rem; color: var(--text-dim); margin-top: 0.25rem; line-height: 1.4; }

    .section-label {
        font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-dim);
        font-weight: 600; margin: 2.4rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-pill">🤟 Computer vision · ASL fingerspelling</div>
    <div class="hero-title">Show me a sign, I'll tell you the letter</div>
    <div class="hero-sub">A convolutional neural network trained on 27,455 hand-sign images, recognizing
    24 static letters of the ASL alphabet from a single photo. Upload one below to try it.</div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Work area — real st.container so content actually nests inside the card
# ----------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown(
            '<div class="card-header"><div class="card-icon">📷</div>'
            '<div class="card-title">Your photo</div></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop a hand sign photo here, or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        else:
            st.markdown(
                '<div class="pred-empty"><div class="pred-empty-icon">🖼️</div>'
                "Your uploaded photo will appear here</div>",
                unsafe_allow_html=True,
            )

with col_right:
    with st.container(border=True):
        st.markdown(
            '<div class="card-header"><div class="card-icon">✨</div>'
            '<div class="card-title">Result</div></div>',
            unsafe_allow_html=True,
        )

        if uploaded_file is not None:
            gray = image.convert("L").resize((28, 28))
            img_array = np.array(gray) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class] * 100)
            letter = labels[predicted_class]

            top3_idx = np.argsort(prediction)[::-1][:3]
            chips_html = ""
            for idx in top3_idx:
                pct = float(prediction[idx] * 100)
                chips_html += (
                    f'<div class="rank-chip"><div class="rank-chip-letter">{labels[idx]}</div>'
                    f'<div class="rank-chip-pct">{pct:.0f}%</div></div>'
                )

            st.markdown(
                f"""
                <div class="pred-active">
                    <div class="letter-badge"><div class="letter-text">{letter}</div></div>
                    <div class="conf-text">{confidence:.1f}% confident</div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width:{confidence:.1f}%;"></div>
                    </div>
                    <div class="rank-strip">{chips_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pred-empty"><div class="pred-empty-icon">🤚</div>'
                "Upload a photo on the left to see the predicted letter</div>",
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# Model story
# ----------------------------------------------------------------------
st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)

stat_cols = st.columns(4)
stats = [
    ("95.7%", "Test accuracy (CNN)"),
    ("72.7%", "Test accuracy (baseline dense network)"),
    ("24", "Letters recognized — excludes J and Z, which require motion"),
    ("27,455", "Training images"),
]
for col, (number, label) in zip(stat_cols, stats):
    with col:
        with st.container(border=True):
            st.markdown(
                f'<div class="stat-number">{number}</div><div class="stat-label">{label}</div>',
                unsafe_allow_html=True,
            )