import sys
from pathlib import Path
import hashlib
import io

# ============================================================
# LEAFGUARD AI
# Smart Crop Health & Disease Detection
# ============================================================

# Project root for local execution and Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

from src.config import IMAGE_SIZE
from src.predict import LeafDiseasePredictor
from src.preprocessing import preprocess_single_image


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LeafGuard AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. DISEASE KNOWLEDGE BASE
# ============================================================

DISEASE_INFO = {

    "Apple___Apple_scab": {
        "description": (
            "Apple scab is a fungal infection caused by "
            "Venturia inaequalis affecting apple foliage and fruit."
        ),
        "symptoms": (
            "Olive-green to black velvet-like spots on leaves, "
            "leaf yellowing, and premature leaf drop."
        ),
        "recommendation": (
            "Rake and destroy fallen leaves in autumn, prune tree "
            "canopy to improve airflow, and avoid leaving wet foliage overnight."
        )
    },

    "Apple___Black_rot": {
        "description": (
            "Black rot is a fungal disease caused by "
            "Botryosphaeria obtusa affecting leaves, fruit, and bark."
        ),
        "symptoms": (
            "'Frog-eye' leaf spots with purple margins and tan centers, "
            "black decaying fruit spots, and branch cankers."
        ),
        "recommendation": (
            "Prune out dead wood and cankers during winter dormancy, "
            "remove mummified fruit, and maintain orchard sanitation."
        )
    },

    "Apple___Cedar_apple_rust": {
        "description": (
            "Cedar apple rust is a fungal disease caused by "
            "Gymnosporangium juniperi-virginianae requiring alternate hosts."
        ),
        "symptoms": (
            "Bright yellow-orange spots on the upper leaf surface "
            "with tiny tube-like projections under the leaf."
        ),
        "recommendation": (
            "Remove nearby red cedar or juniper hosts if feasible, "
            "plant rust-resistant apple cultivars, and maintain general tree vigor."
        )
    },

    "Apple___healthy": {
        "description": (
            "No disease detected. Apple foliage exhibits vibrant "
            "green coloration and normal growth."
        ),
        "symptoms": (
            "Clean, smooth green leaves with uniform texture "
            "and no visible lesions or discoloration."
        ),
        "recommendation": (
            "Continue standard orchard management, balanced irrigation, "
            "and periodic crop monitoring."
        )
    },

    "Potato___Early_blight": {
        "description": (
            "Early blight is a common fungal leaf spot disease "
            "caused by Alternaria solani in potatoes."
        ),
        "symptoms": (
            "Dark brown circular spots with concentric target-like "
            "rings surrounded by yellow leaf halos."
        ),
        "recommendation": (
            "Practice crop rotation, mulch soil base to reduce spore splash, "
            "avoid overhead watering, and prune lower infected leaves."
        )
    },

    "Potato___healthy": {
        "description": (
            "No disease detected. Potato plant foliage is healthy and active."
        ),
        "symptoms": (
            "Vibrant green foliage free of spots, blighting, or wilting."
        ),
        "recommendation": (
            "Maintain consistent soil moisture, soil hilling practices, "
            "and routine pest scouting."
        )
    },

    "Potato___Late_blight": {
        "description": (
            "Late blight is a destructive water mold (Oomycete) disease "
            "caused by Phytophthora infestans."
        ),
        "symptoms": (
            "Large, dark water-soaked lesions on leaf tips and edges, "
            "often with delicate white fungal fuzz beneath in humid weather."
        ),
        "recommendation": (
            "Remove and dispose of infected plants promptly to prevent field spread, "
            "ensure good soil drainage, and consult local extension guidelines."
        )
    },

    "Tomato___Bacterial_spot": {
        "description": (
            "Bacterial spot is caused by Xanthomonas species affecting "
            "tomato leaves, stems, and fruit."
        ),
        "symptoms": (
            "Small, dark, water-soaked leaf spots that dry into dark brown "
            "scabbed lesions, causing leaf yellowing and drop."
        ),
        "recommendation": (
            "Avoid working in foliage when wet, use disease-free certified seeds, "
            "practice multi-year rotation, and avoid overhead irrigation."
        )
    },

    "Tomato___Early_blight": {
        "description": (
            "Early blight is a widespread fungal disease caused by "
            "Alternaria solani affecting tomatoes."
        ),
        "symptoms": (
            "Dark brown spots with concentric ring patterns on older lower leaves, "
            "leading to yellowing and leaf loss."
        ),
        "recommendation": (
            "Stake plants for upright growth, mulch soil base, prune affected "
            "lower leaves, and maintain plant spacing for air circulation."
        )
    },

    "Tomato___healthy": {
        "description": (
            "No disease detected. Tomato plant exhibits healthy, vigorous growth."
        ),
        "symptoms": (
            "Deep green leaves with normal morphology and no signs of spotting or necrosis."
        ),
        "recommendation": (
            "Maintain regular watering at plant base, proper trellising, "
            "and routine visual crop inspections."
        )
    },

    "Tomato___Late_blight": {
        "description": (
            "Late blight is a serious infection caused by Phytophthora infestans "
            "capable of rapidly affecting tomato crops."
        ),
        "symptoms": (
            "Irregular dark brown water-soaked leaf spots, white downy growth "
            "on lower leaf surfaces during humid weather, and dark stem lesions."
        ),
        "recommendation": (
            "Promptly remove and destroy infected plant material, avoid wet foliage, "
            "and practice multi-year crop rotation."
        )
    },

    "Tomato___Leaf_Mold": {
        "description": (
            "Leaf mold is a fungal disease caused by Passalora fulva "
            "(Cladosporium fulvum), prevalent in high humidity."
        ),
        "symptoms": (
            "Pale yellow spots on upper leaf surfaces corresponding to velvety "
            "olive-green to light brown mold beneath."
        ),
        "recommendation": (
            "Increase airflow and ventilation around plants, reduce humidity, "
            "space plants adequately, and prune lower dense foliage."
        )
    },

    "Tomato___Septoria_leaf_spot": {
        "description": (
            "Septoria leaf spot is a foliage disease caused by the fungus "
            "Septoria lycopersici."
        ),
        "symptoms": (
            "Numerous small circular spots with dark brown margins and "
            "tan/grey centers containing tiny black fruiting bodies."
        ),
        "recommendation": (
            "Remove lower infected foliage, apply ground mulch to prevent soil splash, "
            "and avoid overhead sprinkler watering."
        )
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": (
            "Damage caused by Two-Spotted Spider Mites (Tetranychus urticae), "
            "microscopic sap-sucking arachnids."
        ),
        "symptoms": (
            "Fine yellow or white stippling on leaf surfaces, bronze-yellow foliage "
            "discoloration, and fine silk webbing underneath."
        ),
        "recommendation": (
            "Rinse leaf undersides with water sprays, apply neem oil or insecticidal "
            "soap if needed, and foster beneficial predatory insects."
        )
    },

    "Tomato___Target_Spot": {
        "description": (
            "Target spot is a fungal foliage disease caused by "
            "Corynespora cassiicola."
        ),
        "symptoms": (
            "Small pinpoint spots that expand into brown circular lesions "
            "with light tan centers and concentric rings."
        ),
        "recommendation": (
            "Maintain row spacing to facilitate leaf drying, prune lower "
            "infected leaves, and practice crop rotation."
        )
    },

    "Tomato___Tomato_mosaic_virus": {
        "description": (
            "Tomato Mosaic Virus (ToMV) is a persistent viral pathogen "
            "transmitted mechanically by contact."
        ),
        "symptoms": (
            "Mottled light and dark green mosaic leaf patterns, leaf distortion, "
            "stunting, and puckered growth."
        ),
        "recommendation": (
            "Remove and destroy infected plants (viruses cannot be cured chemically), "
            "sanitize tools thoroughly, and wash hands before handling plants."
        )
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": (
            "Tomato Yellow Leaf Curl Virus (TYLCV) is a viral disease "
            "transmitted primarily by silverleaf whiteflies."
        ),
        "symptoms": (
            "Severe upward leaf curling and yellowing along leaf margins, "
            "leaf size reduction, and stunted plant growth."
        ),
        "recommendation": (
            "Control whitefly vectors using insect netting or reflective mulches, "
            "remove infected reservoir plants, and use resistant varieties."
        )
    }
}


# ============================================================
# 3. IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(pil_image):
    """
    Checks image resolution, darkness, overexposure,
    and image detail/sharpness.
    """

    width, height = pil_image.size

    if width < 100 or height < 100:
        return (
            False,
            "The uploaded image resolution is too small "
            "(< 100 × 100 px). Please upload a higher "
            "resolution crop leaf photo."
        )

    img_np = np.array(
        pil_image.convert("RGB"),
        dtype=np.float32
    )

    gray = np.mean(
        img_np,
        axis=2
    )

    # Darkness check
    dark_pixel_ratio = float(
        np.mean(gray < 40.0)
    )

    p75_brightness = float(
        np.percentile(gray, 75)
    )

    if (
        dark_pixel_ratio > 0.65
        or p75_brightness < 45.0
    ):
        return (
            False,
            "The uploaded image is too dark or underexposed. "
            "Please upload a well-lit photo of the crop leaf."
        )

    # Overexposure check
    mean_brightness = float(
        np.mean(img_np)
    )

    if mean_brightness > 225.0:
        return (
            False,
            "The uploaded image is overexposed / too bright. "
            "Please upload a photo taken under balanced lighting."
        )

    # Detail / sharpness check
    gy, gx = np.gradient(gray)

    detail_score = float(
        np.mean(
            np.sqrt(
                gx ** 2 +
                gy ** 2
            )
        )
    )

    if detail_score < 3.5:
        return (
            False,
            "The uploaded image appears blurry or lacks fine detail. "
            "Please upload a clear, focused close-up of the crop leaf."
        )

    return True, None


# ============================================================
# 4. GRAD-CAM
# ============================================================

def generate_gradcam_overlay(
    model,
    input_data,
    class_index,
    original_image
):
    """
    Generates Grad-CAM using the confirmed 'out_relu' layer.
    """

    grad_layer = model.get_layer(
        "out_relu"
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            grad_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_data,
            training=False
        )

        class_score = predictions[
            :,
            class_index
        ]

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # Improve visual contrast
    heatmap = np.power(
        np.clip(
            heatmap,
            0.0,
            1.0
        ),
        0.85
    )

    # Resize original
    original = (
        original_image
        .convert("RGB")
        .resize(
            (448, 448),
            Image.Resampling.LANCZOS
        )
    )

    # Resize heatmap
    heat_img = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    ).resize(
        original.size,
        Image.Resampling.BILINEAR
    )

    heat_array = (
        np.asarray(
            heat_img,
            dtype=np.float32
        ) / 255.0
    )

    # Red/yellow activation overlay
    heat_rgba = np.zeros(
        (
            original.height,
            original.width,
            4
        ),
        dtype=np.uint8
    )

    heat_rgba[..., 0] = 255

    heat_rgba[..., 1] = np.uint8(
        255 * (1.0 - heat_array)
    )

    heat_rgba[..., 2] = 0

    heat_rgba[..., 3] = np.uint8(
        205 * heat_array
    )

    overlay = Image.fromarray(
        heat_rgba,
        mode="RGBA"
    )

    result = Image.alpha_composite(
        original.convert("RGBA"),
        overlay
    )

    return result.convert("RGB")


# ============================================================
# 5. CACHED MODEL LOADING
# ============================================================

@st.cache_resource
def load_leaf_predictor():

    predictor = LeafDiseasePredictor()

    predictor.load_model()

    return predictor


# ============================================================
# 6. SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_file_hash" not in st.session_state:
    st.session_state.analysis_file_hash = None

if "analysis_image_bytes" not in st.session_state:
    st.session_state.analysis_image_bytes = None

if "gradcam_image" not in st.session_state:
    st.session_state.gradcam_image = None


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌱 LeafGuard AI")

    st.caption(
        "Smart Crop Health & Disease Detection"
    )

    st.divider()

    st.markdown("### 🌾 Supported Crops")

    st.markdown(
        """
        🍎 **Apple**

        Scab • Black Rot • Cedar Apple Rust • Healthy

        🥔 **Potato**

        Early Blight • Late Blight • Healthy

        🍅 **Tomato**

        Bacterial Spot • Early Blight • Late Blight  
        Leaf Mold • Septoria • Spider Mites  
        Target Spot • Mosaic Virus • TYLCV • Healthy
        """
    )

    st.divider()

    st.markdown("### 🧭 How It Works")

    st.markdown(
        """
        **01** Upload a clear leaf photo

        **02** Run AI analysis

        **03** Review the result

        **04** Explain the prediction
        """
    )

    st.divider()

    st.markdown("### 🛡️ Safety")

    st.caption(
        "LeafGuard checks image quality before inference "
        "and flags uncertain predictions."
    )

    st.divider()

    st.caption(
        "LeafGuard AI v1"
    )


# ============================================================
# 8. TOP BRAND HEADER
# ============================================================

logo_col, title_col, status_col = st.columns(
    [0.7, 5, 1.4],
    vertical_alignment="center"
)

with logo_col:

    st.markdown(
        "# 🌱"
    )

with title_col:

    st.title(
        "LeafGuard AI"
    )

    st.caption(
        "Smart Crop Health & Disease Detection"
    )

with status_col:
    st.caption("🤖 MobileNetV2 • 17 Classes")
    


st.write(
    "Analyze Apple, Potato, and Tomato leaves using "
    "MobileNetV2 with image-quality protection, "
    "confidence-aware safety checks, and Grad-CAM explainability."
)

st.divider()


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

st.subheader(
    "📊 LeafGuard AI at a Glance"
)

metric_1, metric_2, metric_3 = st.columns(
    3
)

with metric_1:

    st.metric(
        label="Supported Classes",
        value="17"
    )

    st.caption(
        "Apple • Potato • Tomato"
    )

with metric_2:

    st.metric(
        label="Test Accuracy",
        value="87.70%"
    )

    st.caption(
        "Held-out benchmark"
    )

with metric_3:

    st.metric(
        label="Explainability",
        value="Grad-CAM"
    )

    st.caption(
        "Visual model attention"
    )


st.divider()


# ============================================================
# 10. STEP 1 - UPLOAD
# ============================================================

st.subheader(
    "📷 Step 1 — Upload & Prepare Your Leaf"
)

st.progress(
    0.25,
    text="Step 1 of 4 — Upload your leaf image"
)

st.info(
    "Use a clear, focused, well-lit close-up image "
    "from Apple, Potato, or Tomato."
)

uploaded_file = st.file_uploader(
    "Choose a crop leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key="leaf_uploader"
)


if uploaded_file is not None:

    try:

        file_bytes = uploaded_file.getvalue()

        current_file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        # If this is a new image, remove the previous result.
        if (
            st.session_state.analysis_file_hash
            != current_file_hash
        ):

            st.session_state.analysis_file_hash = (
                current_file_hash
            )

            st.session_state.analysis_result = None
            st.session_state.analysis_image_bytes = None
            st.session_state.gradcam_image = None

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        st.image(
            image,
            caption=f"Uploaded Leaf • {uploaded_file.name}",
            width="stretch"
        )

        # Image quality check
        is_quality_ok, quality_msg = (
            check_image_quality(image)
        )

        quality_col1, quality_col2 = st.columns(
            2
        )

        with quality_col1:

            st.write(
                f"**Resolution**  \n"
                f"{image.width} × {image.height}px"
            )

        with quality_col2:

            if is_quality_ok:

                st.success(
                    "✅ Image quality passed"
                )

            else:

                st.warning(
                    "⚠️ Image quality issue"
                )

        if not is_quality_ok:

            st.warning(
                quality_msg
            )

        st.write("")

        analyze_btn = st.button(
            "🧪 Analyze Leaf",
            type="primary",
            width="stretch",
            key="analyze_button"
        )

        # ----------------------------------------------------
        # RUN INFERENCE
        # ----------------------------------------------------

        if analyze_btn:

            if not is_quality_ok:

                st.session_state.analysis_result = None
                st.session_state.analysis_image_bytes = None
                st.session_state.gradcam_image = None

                st.error(
                    "Analysis stopped because the image "
                    "did not pass the quality checks."
                )

            else:

                with st.spinner(
                    "🔬 Running LeafGuard AI analysis..."
                ):

                    try:

                        predictor = (
                            load_leaf_predictor()
                        )

                        uploaded_file.seek(0)

                        result = (
                            predictor.predict(
                                uploaded_file
                            )
                        )

                        st.session_state.analysis_result = (
                            result
                        )

                        st.session_state.analysis_image_bytes = (
                            file_bytes
                        )

                        st.session_state.gradcam_image = None

                        st.success(
                            "✅ Step 1 completed. "
                            "Your image has been analyzed."
                        )

                    except Exception as err:

                        st.session_state.analysis_result = None
                        st.session_state.analysis_image_bytes = None
                        st.session_state.gradcam_image = None

                        st.error(
                            f"❌ **Prediction Error:** {err}"
                        )

    except Exception as err:

        st.error(
            f"❌ **Invalid Image:** {err}"
        )


else:

    st.warning(
        "Upload a leaf image to begin Step 1."
    )


# ============================================================
# 11. STEP 2 - AI DIAGNOSIS
#     Appears only after Step 1 is completed.
# ============================================================

if st.session_state.analysis_result is not None:

    st.divider()

    st.subheader(
        "🔬 Step 2 — AI Diagnosis"
    )

    st.progress(
        0.50,
        text="Step 2 of 4 — Reviewing AI diagnosis"
    )

    result = (
        st.session_state.analysis_result
    )

    analysis_image = Image.open(
        io.BytesIO(
            st.session_state.analysis_image_bytes
        )
    ).convert("RGB")

    raw_class_name = (
        result["predicted_class_name"]
    )

    confidence_val = float(
        result["confidence"]
    )

    confidence_pct = (
        confidence_val * 100.0
    )

    # --------------------------------------------------------
    # Probability analysis
    # --------------------------------------------------------

    all_probs = sorted(
        [
            float(p)
            for p in result[
                "all_probabilities"
            ]
        ],
        reverse=True
    )

    top1_prob = (
        all_probs[0]
        if len(all_probs) > 0
        else confidence_val
    )

    top2_prob = (
        all_probs[1]
        if len(all_probs) > 1
        else 0.0
    )

    margin = (
        top1_prob -
        top2_prob
    )

    # Existing Phase 21 safeguard
    is_unrecognized = (
        top1_prob < 0.50
        or margin < 0.20
    )

    formatted_name = (
        raw_class_name
        .replace("___", " - ")
        .replace("_", " ")
    )

    is_healthy = (
        "healthy"
        in raw_class_name.lower()
    )


    # --------------------------------------------------------
    # Diagnosis result
    # --------------------------------------------------------

    if is_unrecognized:

        st.warning(
            "⚠️ **Unrecognized Crop or Foliage**"
        )

        st.markdown(
            "The model cannot confidently match this "
            "image to the supported Apple, Potato, "
            "or Tomato conditions."
        )

        st.info(
            f"**Nearest Model Match (Uncertain):** "
            f"{formatted_name}"
        )

    elif confidence_pct < 70.0:

        st.warning(
            "⚠️ **Low-Confidence Prediction**"
        )

        st.markdown(
            f"### [Low Confidence] {formatted_name}"
        )

        st.write(
            f"The model certainty is "
            f"**{confidence_pct:.2f}%**, so this result "
            "should be treated as an estimated prediction."
        )

    else:

        if is_healthy:

            st.success(
                "🌱 **Healthy Crop Tissue**"
            )

        else:

            st.error(
                "⚠️ **Crop Disease Detected**"
            )

        st.markdown(
            f"### {formatted_name}"
        )

        st.caption(
            "Prediction generated by the trained MobileNetV2 model."
        )


    # --------------------------------------------------------
    # Confidence section
    # --------------------------------------------------------

    st.markdown(
        "#### 🎯 Model Confidence"
    )

    confidence_col1, confidence_col2 = (
        st.columns([1, 2])
    )

    with confidence_col1:

        st.metric(
            label="Model Certainty",
            value=f"{confidence_pct:.2f}%"
        )

    with confidence_col2:

        st.progress(
            min(
                max(
                    confidence_val,
                    0.0
                ),
                1.0
            )
        )

        st.caption(
            f"Top-1 vs Top-2 probability margin: "
            f"{margin:.2f}"
        )


    # --------------------------------------------------------
    # Scope note
    # --------------------------------------------------------

    st.info(
        "ℹ️ **Scope:** LeafGuard AI v1 is trained for "
        "Apple, Potato, and Tomato leaf conditions."
    )


    # --------------------------------------------------------
    # Warning
    # --------------------------------------------------------

    if is_unrecognized:

        st.warning(
            "⚠️ **Unrecognized Crop or Foliage Notice:** "
            "Please upload a clear close-up leaf image "
            "from a supported crop."
        )

    elif confidence_pct < 70.0:

        st.warning(
            "⚠️ **Low-Confidence Warning:** "
            "This is an estimated prediction. Try another "
            "clear, well-lit close-up image for re-evaluation."
        )


    # ========================================================
    # STEP 3 - UNDERSTAND RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "📖 Step 3 — Understand the Result"
    )

    st.progress(
        0.75,
        text="Step 3 of 4 — Condition information"
    )

    if is_unrecognized:

        st.info(
            "Detailed disease information is not shown "
            "because the image was not confidently matched "
            "to a supported condition."
        )

    else:

        info = DISEASE_INFO.get(
            raw_class_name,
            {
                "description":
                    "No detailed description available.",

                "symptoms":
                    "N/A",

                "recommendation":
                    "Consult a local agricultural "
                    "extension specialist for guidance."
            }
        )

        info_col1, info_col2 = st.columns(
            2
        )

        with info_col1:

            with st.container(border=True):

                st.markdown(
                    "### 📖 Condition Overview"
                )

                st.write(
                    info["description"]
                )

        with info_col2:

            with st.container(border=True):

                st.markdown(
                    "### 🔍 Common Symptoms"
                )

                st.write(
                    info["symptoms"]
                )

        with st.container(border=True):

            st.markdown(
                "### 🛡️ Recommended General Action"
            )

            st.write(
                info["recommendation"]
            )


    # ========================================================
    # STEP 4 - GRAD-CAM
    # ========================================================

    st.divider()

    st.subheader(
        "🔬 Step 4 — Explain This Prediction"
    )

    st.progress(
        1.0,
        text="Step 4 of 4 — Explainable AI"
    )

    st.write(
        "Grad-CAM highlights regions of the image "
        "that influenced the model's prediction."
    )

    st.caption(
        "This visualization explains model behavior. "
        "It does not prove that a highlighted region "
        "contains the disease."
    )


    # --------------------------------------------------------
    # Generate Grad-CAM button
    # --------------------------------------------------------

    if not is_unrecognized:

        explain_btn = st.button(
            "🔬 Generate Grad-CAM Explanation",
            type="secondary",
            width="stretch",
            key="gradcam_button"
        )

        if explain_btn:

            with st.spinner(
                "🧠 Generating AI attention map..."
            ):

                try:

                    # Use the exact preprocessing pipeline
                    gradcam_input = (
                        preprocess_single_image(
                            io.BytesIO(
                                st.session_state.analysis_image_bytes
                            ),
                            target_size=IMAGE_SIZE
                        )
                    )

                    predictor = (
                        load_leaf_predictor()
                    )

                    gradcam_image = (
                        generate_gradcam_overlay(
                            predictor.model,
                            gradcam_input,
                            result[
                                "predicted_class_index"
                            ],
                            analysis_image
                        )
                    )

                    st.session_state.gradcam_image = (
                        gradcam_image
                    )

                except Exception as exc:

                    st.session_state.gradcam_image = None

                    st.warning(
                        "Grad-CAM visualization could not "
                        f"be generated: {exc}"
                    )

    else:

        st.info(
            "Grad-CAM is unavailable because this prediction "
            "is currently classified as unrecognized/uncertain."
        )


    # --------------------------------------------------------
    # Display Grad-CAM
    # --------------------------------------------------------

    if (
        st.session_state.gradcam_image
        is not None
    ):

        st.markdown(
            "### 🧠 Model Attention Visualization"
        )

        st.caption(
            "Highlighted regions show areas that contributed "
            "more strongly to the model's prediction."
        )

        original_col, gradcam_col = (
            st.columns(2)
        )

        with original_col:

            st.image(
                analysis_image,
                caption="Original Leaf",
                width="stretch"
            )

        with gradcam_col:

            st.image(
                st.session_state.gradcam_image,
                caption="Grad-CAM Explanation",
                width="stretch"
            )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 LeafGuard AI v1 • MobileNetV2 • "
    "17 Apple, Potato & Tomato conditions • "
    "Confidence-aware AI • Grad-CAM Explainability"
)