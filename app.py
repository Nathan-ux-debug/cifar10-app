import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io

# Page config
st.set_page_config(
    page_title="CIFAR-10 Neural Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=Space+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .main { background-color: #0a0a0f; color: #e8e8f0; }

    .stApp { background-color: #0a0a0f; }

    h1, h2, h3 { color: #e8e8f0 !important; }

    .title-block {
        padding: 2rem 0 1rem;
        border-bottom: 1px solid #2a2a3d;
        margin-bottom: 2rem;
    }

    .tag {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #7c6aff;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .main-title {
        font-size: 48px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #7c6aff, #ff6a9b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #7a7a9a;
        font-size: 15px;
        margin-top: 8px;
    }

    .pred-class {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #6affd4, #7c6aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: capitalize;
    }

    .conf-text {
        font-family: 'Space Mono', monospace;
        font-size: 14px;
        color: #7a7a9a;
        margin-top: 4px;
    }

    .conf-value { color: #6affd4; font-weight: 700; }

    .stat-box {
        background: #12121a;
        border: 1px solid #2a2a3d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .stat-value {
        font-family: 'Space Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #7c6aff, #ff6a9b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        font-size: 11px;
        color: #7a7a9a;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
        font-family: 'Space Mono', monospace;
    }

    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #7a7a9a;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/cifar10_model.h5', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

def preprocess(img):
    img = img.convert('RGB').resize((32, 32))
    arr = np.array(img).astype('float32') / 255.0
    return arr.reshape(1, 32, 32, 3)

def predict(img):
    arr = preprocess(img)
    preds = model.predict(arr, verbose=0)[0]
    top = int(np.argmax(preds))
    return CLASS_NAMES[top], float(preds[top]), preds

# Header
st.markdown("""
<div class="title-block">
    <div class="tag">— Deep Learning · Computer Vision</div>
    <div class="main-title">CIFAR-10 Neural Classifier</div>
    <div class="subtitle">CNN trained on 60,000 images across 10 object classes</div>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🖼️ Upload an Image")
    uploaded = st.file_uploader("Choose any image", type=['png', 'jpg', 'jpeg', 'webp'])

    st.markdown("### Or pick a sample")
    sample_cols = st.columns(5)
    selected_sample = None

    for i, cls in enumerate(CLASS_NAMES):
        with sample_cols[i % 5]:
            try:
                sample_img = Image.open(f'static/sample_images/{cls}.png')
                st.image(sample_img, caption=cls, use_container_width=True)
                if st.button(cls, key=f"btn_{cls}", use_container_width=True):
                    selected_sample = cls
            except:
                if st.button(cls, key=f"btn_{cls}", use_container_width=True):
                    selected_sample = cls

with col2:
    st.markdown("### 🎯 Prediction Result")

    img_to_predict = None

    if uploaded:
        img_to_predict = Image.open(uploaded)
        st.image(img_to_predict, caption="Uploaded image", width=160)
    elif selected_sample:
        try:
            img_to_predict = Image.open(f'static/sample_images/{selected_sample}.png')
            st.image(img_to_predict, caption=selected_sample, width=160)
        except:
            st.error("Sample image not found.")

    if img_to_predict:
        with st.spinner("Predicting..."):
            pred_class, confidence, all_scores = predict(img_to_predict)

        st.markdown(f'<div class="pred-class">{pred_class}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="conf-text">Confidence: <span class="conf-value">{confidence*100:.1f}%</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Class probability distribution</div>', unsafe_allow_html=True)

        sorted_scores = sorted(zip(CLASS_NAMES, all_scores), key=lambda x: x[1], reverse=True)
        for cls, score in sorted_scores:
            cols = st.columns([2, 6, 1])
            with cols[0]:
                st.markdown(f"<small style='color:#7a7a9a; font-family:monospace'>{cls}</small>", unsafe_allow_html=True)
            with cols[1]:
                color = "#6affd4" if cls == pred_class else "#7c6aff"
                st.markdown(f"""
                <div style="background:#1a1a26; border-radius:99px; height:8px; margin-top:6px;">
                    <div style="background:{color}; width:{score*100}%; height:8px; border-radius:99px;"></div>
                </div>""", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<small style='color:#7a7a9a; font-family:monospace'>{score*100:.1f}%</small>", unsafe_allow_html=True)
    else:
        st.info("Upload an image or pick a sample on the left to get a prediction.")

# Evaluation section
st.markdown("---")
st.markdown("### 📊 Model Evaluation")
st.markdown("Runs 10 validation passes on CIFAR-10 test data and generates a full confusion matrix.")
st.caption("⚠️ This takes about 60–90 seconds")

if st.button("▶ Run Evaluation + Confusion Matrix", type="primary"):
    with st.spinner("Loading CIFAR-10 test data and running validation..."):
        from tensorflow.keras.datasets import cifar10
        (_, _), (X_test, y_test) = cifar10.load_data()
        X_test = X_test.astype('float32') / 255.0
        y_true = y_test.flatten()

        results = []
        progress = st.progress(0, text="Running validation passes...")
        for run in range(10):
            idx = np.random.choice(len(X_test), 200, replace=False)
            preds = model.predict(X_test[idx], verbose=0)
            y_pred = np.argmax(preds, axis=1)
            acc = np.mean(y_pred == y_true[idx])
            results.append(round(float(acc) * 100, 2))
            progress.progress((run + 1) / 10, text=f"Run {run+1}/10 — {acc*100:.1f}%")

        progress.empty()

        all_preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
        overall_acc = float(np.mean(all_preds == y_true)) * 100
        mean_acc = sum(results) / 10

    # Stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{overall_acc:.1f}%</div><div class="stat-label">Test Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{mean_acc:.1f}%</div><div class="stat-label">Mean Val Acc</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{max(results)}%</div><div class="stat-label">Best Run</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">10 Validation Run Accuracies</div>', unsafe_allow_html=True)
    run_cols = st.columns(10)
    for i, acc in enumerate(results):
        with run_cols[i]:
            st.markdown(f"<div style='background:#1a1a26; border:1px solid #2a2a3d; border-radius:8px; padding:8px; text-align:center; font-family:monospace; font-size:12px; color:#6affd4'><span style='color:#7a7a9a; font-size:10px'>Run {i+1}</span><br>{acc}%</div>", unsafe_allow_html=True)

    # Confusion matrix
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y_true, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#12121a')
    ax.set_facecolor('#12121a')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted Label', color='#7a7a9a')
    ax.set_ylabel('True Label', color='#7a7a9a')
    ax.set_title('Confusion Matrix — CIFAR-10', color='#e8e8f0', pad=15)
    ax.tick_params(colors='#7a7a9a')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()