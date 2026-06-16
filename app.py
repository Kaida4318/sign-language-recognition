import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# ----------------------------------------------------------------------
# Page config — wide layout, custom title/icon shown in the browser tab
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sign Recognition",
    page_icon="🤟",
    layout="wide",
)

# ----------------------------------------------------------------------
# Load model once (cached so it doesn't reload on every interaction)
# ----------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model("sign_language_cnn.h5")

model = get_model()
labels = [chr(i) for i in range(65, 91) if chr(i) not in ["J", "Z"]]

# ----------------------------------------------------------------------
# Custom CSS — design system for this page
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #161616;
        --bg-panel: #1d1d1d;
        --border: #2a2a2a;
        --text: #f2efe9;
        --text-dim: #9a958c;
        --accent: #ff8a3d;
        --accent-dim: #5c3a22;
        --success: #2dd4bf;
    }

    html, body, [class*="css"] {
        background-color: var(--bg) !important;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 1100px;
    }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu, footer, header {visibility: hidden;}

    .hero-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.1rem;
        font-weight: 700;
        line-height: 1.05;
        color: var(--text);
        margin-bottom: 0.8rem;
        letter-spacing: -0.01em;
    }

    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: var(--text-dim);
        max-width: 620px;
        line-height: 1.55;
        margin-bottom: 2.2rem;
    }

    .panel {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.8rem;
    }

    .panel-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-dim);
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .result-letter {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 7.5rem;
        font-weight: 700;
        line-height: 1;
        color: var(--accent);
        text-align: center;
        margin: 0.4rem 0;
    }

    .result-caption {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: var(--text-dim);
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }

    .confidence-track {
        background: var(--border);
        border-radius: 99px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin-top: 0.6rem;
    }

    .confidence-fill {
        background: linear-gradient(90deg, var(--accent-dim), var(--accent));
        height: 100%;
        border-radius: 99px;
    }

    .stat-row {
        display: flex;
        gap: 1.2rem;
        margin-top: 2.4rem;
    }

    .stat-box {
        flex: 1;
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
    }

    .stat-number {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--success);
    }

    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: var(--text-dim);
        margin-top: 0.2rem;
    }

    .divider-line {
        border-top: 1px solid var(--border);
        margin: 3rem 0 2rem 0;
    }

    [data-testid="stFileUploader"] {
        background: var(--bg-panel);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown('<div class="hero-eyebrow">Computer Vision · ASL Fingerspelling</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Sign Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A convolutional neural network trained on 27,455 hand-sign '
    'images, recognizing 24 static letters of the ASL alphabet from a single photo. '
    'Upload a hand sign below to see it in action.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Main two-column layout: upload (left) / result (right)
# ----------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Input image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        " ", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Prediction</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        gray = image.convert("L").resize((28, 28))
        img_array = np.array(gray) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        with st.spinner(""):
            prediction = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        letter = labels[predicted_class]

        st.markdown(f'<div class="result-letter">{letter}</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-caption">Predicted letter</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{confidence:.1f}%;"></div>
            </div>
            <div class="result-caption" style="margin-top:0.5rem;">{confidence:.1f}% confidence</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center; color:#9a958c; padding: 3.5rem 0; font-size:0.95rem;">'
            "Upload an image to see a prediction"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Model story — stat row for credibility
# ----------------------------------------------------------------------
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
st.markdown('<div class="panel-label">Model performance</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-number">95.7%</div>
            <div class="stat-label">Test accuracy (CNN)</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">72.7%</div>
            <div class="stat-label">Test accuracy (baseline dense network)</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">24</div>
            <div class="stat-label">Letters recognized (excludes J, Z — motion signs)</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">27,455</div>
            <div class="stat-label">Training images</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)