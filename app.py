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
# CSS — technical signal-display direction
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #0d0f0e;
        --bg-panel: #131613;
        --border: #232823;
        --border-bright: #2f3a2f;
        --text: #e6ebe6;
        --text-dim: #6f7a6f;
        --signal: #5fffaa;
        --signal-dim: #1f5c43;
        --signal-faint: rgba(95, 255, 170, 0.08);
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
        background-color: var(--bg) !important;
    }

    .stApp, .stApp p, .stApp span, .stApp label {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }
    .mono { font-family: 'JetBrains Mono', monospace; }

    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
    [data-testid="stToolbar"] { display: none; }

    /* ---------- Header strip ---------- */
    .topline {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid var(--border); padding-bottom: 1.1rem; margin-bottom: 2rem;
    }
    .topline-id {
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--signal);
        letter-spacing: 0.05em;
    }
    .topline-status {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dim);
        display: flex; align-items: center; gap: 6px;
    }
    .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--signal); display: inline-block; }

    .hero-title {
        font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 700;
        line-height: 1.25; margin-bottom: 0.9rem; color: var(--text); letter-spacing: -0.01em;
    }
    .hero-sub { font-size: 0.98rem; color: var(--text-dim); max-width: 640px; line-height: 1.65; margin-bottom: 2.2rem; }

    /* ---------- Containers as panels ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-panel) !important;
        border-radius: 4px !important;
        border: 1px solid var(--border) !important;
    }

    .panel-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--text-dim); font-weight: 500; margin-bottom: 1rem;
        display: flex; align-items: center; gap: 8px;
    }
    .panel-label::before { content: "//"; color: var(--signal-dim); }

    /* Uploader: keep native button, restyle to fit terminal aesthetic */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg) !important;
        border: 1px dashed var(--border-bright) !important;
        border-radius: 2px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: transparent !important;
        color: var(--signal) !important;
        border: 1px solid var(--signal-dim) !important;
        border-radius: 2px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--text-dim) !important; }
    [data-testid="stFileUploaderFile"] button { display: none !important; }
    [data-testid="stFileUploaderDeleteBtn"] { display: inline-flex !important; }

    [data-testid="stImage"] img { border-radius: 2px; border: 1px solid var(--border); margin-top: 0.8rem; }

    /* ---------- Result readout ---------- */
    .readout-empty {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        color: var(--text-dim); font-size: 0.85rem; text-align: center; gap: 0.7rem; padding: 2.6rem 1rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .readout-main { display: flex; align-items: center; gap: 1.8rem; padding: 0.4rem 0 1.4rem 0; }
    .readout-letter {
        font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 5.4rem; line-height: 1;
        color: var(--signal); flex-shrink: 0;
        text-shadow: 0 0 24px var(--signal-faint);
    }
    .readout-meta-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.08em; color: var(--text-dim); margin-bottom: 0.4rem; }
    .readout-conf { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 600; color: var(--text); margin-bottom: 0.6rem; }
    .readout-track { background: var(--border); height: 4px; width: 100%; }
    .readout-fill { background: var(--signal); height: 100%; }

    /* Full 24-class waveform */
    .wave-title { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.08em; color: var(--text-dim); margin: 0.4rem 0 0.8rem 0; }
    .wave-row { display: flex; align-items: center; gap: 8px; padding: 1.5px 0; }
    .wave-label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--text-dim); width: 14px; flex-shrink: 0; }
    .wave-label.active { color: var(--signal); font-weight: 700; }
    .wave-track { flex: 1; background: var(--border); height: 9px; position: relative; overflow: hidden; }
    .wave-fill { background: var(--signal-dim); height: 100%; }
    .wave-fill.active { background: var(--signal); }
    .wave-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--text-dim); width: 38px; text-align: right; flex-shrink: 0; }

    /* ---------- Stats ---------- */
    .section-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--text-dim); font-weight: 500; margin: 2.6rem 0 1rem 0; display: flex; align-items: center; gap: 8px;
    }
    .section-label::before { content: "//"; color: var(--signal-dim); }
    .stat-number { font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; font-weight: 700; color: var(--signal); }
    .stat-label { font-size: 0.78rem; color: var(--text-dim); margin-top: 0.3rem; line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header / hero
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="topline">
        <div class="topline-id">SIGN-RECOGNITION // CNN-CLASSIFIER</div>
        <div class="topline-status"><span class="dot"></span> MODEL LOADED</div>
    </div>
    <div class="hero-title">ASL fingerspelling classifier</div>
    <div class="hero-sub">A convolutional neural network trained on 27,455 hand-sign images, recognizing
    24 static letters of the ASL alphabet from a single photo. Upload one below to run inference.</div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Work area
# ----------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    with st.container(border=True):
        st.markdown('<div class="panel-label">INPUT</div>', unsafe_allow_html=True)
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
                '<div class="readout-empty">[ NO INPUT IMAGE ]<br>awaiting upload</div>',
                unsafe_allow_html=True,
            )

with col_right:
    with st.container(border=True):
        st.markdown('<div class="panel-label">PREDICTION</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            gray = image.convert("L").resize((28, 28))
            img_array = np.array(gray) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class] * 100)
            letter = labels[predicted_class]

            st.markdown(
                f"""
                <div class="readout-main">
                    <div class="readout-letter">{letter}</div>
                    <div style="flex:1;">
                        <div class="readout-meta-label">CONFIDENCE</div>
                        <div class="readout-conf">{confidence:.1f}%</div>
                        <div class="readout-track"><div class="readout-fill" style="width:{confidence:.1f}%;"></div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Full 24-class probability waveform, sorted by probability descending
            order = np.argsort(prediction)[::-1]
            st.markdown('<div class="wave-title">FULL DISTRIBUTION — ALL 24 CLASSES</div>', unsafe_allow_html=True)
            rows_html = ""
            for idx in order:
                pct = float(prediction[idx] * 100)
                is_top = idx == predicted_class
                rows_html += f"""
                <div class="wave-row">
                    <div class="wave-label{' active' if is_top else ''}">{labels[idx]}</div>
                    <div class="wave-track"><div class="wave-fill{' active' if is_top else ''}" style="width:{max(pct, 0.5):.1f}%;"></div></div>
                    <div class="wave-pct">{pct:.1f}%</div>
                </div>
                """
            st.markdown(f'<div>{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="readout-empty">[ STANDBY ]<br>upload an image to run inference</div>',
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# Model stats
# ----------------------------------------------------------------------
st.markdown('<div class="section-label">MODEL METRICS</div>', unsafe_allow_html=True)

stat_cols = st.columns(4)
stats = [
    ("95.7%", "Test accuracy (CNN)"),
    ("72.7%", "Test accuracy, baseline dense network"),
    ("24", "Classes — excludes J, Z (motion signs)"),
    ("27,455", "Training images"),
]
for col, (number, label) in zip(stat_cols, stats):
    with col:
        with st.container(border=True):
            st.markdown(
                f'<div class="stat-number">{number}</div><div class="stat-label">{label}</div>',
                unsafe_allow_html=True,
            )