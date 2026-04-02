import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# -------------------------------------------------
# 1) PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Targeted Object Recognition",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Targeted Object Recognition System")
st.caption(
    "Detects: **Mobile Phone · Charger · Pen** — anything else is flagged as Unknown"
)

# -------------------------------------------------
# 2) MODEL LOADING (cached)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> MobileNetV2:
    """Load MobileNetV2 pre-trained on ImageNet with top classifier head."""
    return MobileNetV2(weights="imagenet", include_top=True)


model = load_model()

# -------------------------------------------------
# 3) TARGET KEYWORD MAP
# -------------------------------------------------
TARGET_KEYWORDS = {
    "Mobile Phone": ["cellular_telephone", "cell_phone", "dial_telephone", "pay-phone"],
    "Charger": ["power_strip", "wall_socket", "electric_plug", "extension_cord"],
    "Pen": ["ballpoint", "fountain_pen", "quill", "pencil_box"],
}

# Sensible threshold for targeted detection; below this we report Unknown.
TARGET_CONFIDENCE_THRESHOLD = 0.18


# -------------------------------------------------
# 4) PREPROCESSING
# -------------------------------------------------
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to MobileNetV2-ready batch tensor."""
    rgb_image = pil_image.convert("RGB")
    img_resized = rgb_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_batch)


# -------------------------------------------------
# 5) PREDICTION FILTERING
# -------------------------------------------------
def get_target_scores(predictions: np.ndarray, top_k: int = 20):
    """
    Aggregate probabilities per target using top-k ImageNet predictions.

    Returns:
      top_preds: decoded top-k list
      target_scores: dict with accumulated probability per target
    """
    top_preds = decode_predictions(predictions, top=top_k)[0]

    target_scores = {target: 0.0 for target in TARGET_KEYWORDS}

    for _, label, prob in top_preds:
        label_lower = label.lower()
        for target, keywords in TARGET_KEYWORDS.items():
            if any(kw in label_lower for kw in keywords):
                target_scores[target] += float(prob)

    return top_preds, target_scores


def choose_final_label(predictions: np.ndarray, top_k: int = 20):
    """Return a robust target decision and confidence."""
    top_preds, target_scores = get_target_scores(predictions, top_k=top_k)

    best_target = max(target_scores, key=target_scores.get)
    best_target_conf = float(target_scores[best_target])

    # If no target signal is strong enough, treat as Unknown.
    if best_target_conf < TARGET_CONFIDENCE_THRESHOLD:
        best_label = top_preds[0][1]
        best_conf = float(top_preds[0][2])
        return {
            "result": "Unknown",
            "confidence": best_conf,
            "best_imagenet_label": best_label,
            "target_scores": target_scores,
            "top_preds": top_preds,
        }

    return {
        "result": best_target,
        "confidence": best_target_conf,
        "best_imagenet_label": top_preds[0][1],
        "target_scores": target_scores,
        "top_preds": top_preds,
    }


# -------------------------------------------------
# 6) UI INPUT
# -------------------------------------------------
st.markdown("---")
input_method = st.radio(
    "Choose input method:", ["📷 Camera", "🖼️ Upload Image"], horizontal=True
)

image = None
if input_method == "📷 Camera":
    camera_photo = st.camera_input("Point camera at: Mobile Phone, Charger, or Pen")
    if camera_photo is not None:
        image = Image.open(camera_photo)
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)


# -------------------------------------------------
# 7) INFERENCE + OUTPUT
# -------------------------------------------------
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Running CNN inference…"):
        tensor = preprocess_image(image)
        predictions = model.predict(tensor, verbose=0)
        decision = choose_final_label(predictions, top_k=20)

    result = decision["result"]
    confidence = decision["confidence"]

    st.markdown("### 🧠 Model Output")

    if result == "Unknown":
        st.error("⚠️ Unknown Object — Not a Mobile Phone, Charger, or Pen")
        st.metric(label="Top ImageNet Confidence", value=f"{confidence * 100:.2f}%")
        st.caption(f"Top ImageNet label: `{decision['best_imagenet_label']}`")
        st.warning(
            "Out-of-Distribution (OOD) detected. "
            "No target class reached the confidence threshold."
        )
    else:
        st.success(f"✅ Detected: **{result}**")
        st.metric(label="Target Confidence Score", value=f"{confidence * 100:.2f}%")
        st.progress(min(confidence, 1.0))

    with st.expander("🎯 Target score breakdown"):
        for target, score in decision["target_scores"].items():
            st.write(f"- {target}: {score * 100:.2f}%")

    with st.expander("🔬 Raw top-5 ImageNet predictions"):
        for rank, (_, label, prob) in enumerate(decision["top_preds"][:5], 1):
            st.write(f"{rank}. `{label}` — {prob * 100:.2f}%")
