import sys
from pathlib import Path
import hashlib
import io

# Add project root directory to sys.path for module resolution on Streamlit Cloud
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


# 1. Page Configuration
st.set_page_config(
    page_title="LeafGuard AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. Knowledge Base for 17 LeafGuard AI v1 Supported Classes
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal infection caused by Venturia inaequalis affecting apple foliage and fruit.",
        "symptoms": "Olive-green to black velvet-like spots on leaves, leaf yellowing, and premature leaf drop.",
        "recommendation": "Rake and destroy fallen leaves in autumn, prune tree canopy to improve airflow, and avoid leaving wet foliage overnight."
    },

    "Apple___Black_rot": {
        "description": "Black rot is a fungal disease caused by Botryosphaeria obtusa affecting leaves, fruit, and bark.",
        "symptoms": "'Frog-eye' leaf spots with purple margins and tan centers, black decaying fruit spots, and branch cankers.",
        "recommendation": "Prune out dead wood and cankers during winter dormancy, remove mummified fruit, and maintain orchard sanitation."
    },

    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae requiring alternate hosts.",
        "symptoms": "Bright yellow-orange spots on the upper leaf surface with tiny tube-like projections under the leaf.",
        "recommendation": "Remove nearby red cedar or juniper hosts if feasible, plant rust-resistant apple cultivars, and maintain general tree vigor."
    },

    "Apple___healthy": {
        "description": "No disease detected. Apple foliage exhibits vibrant green coloration and normal growth.",
        "symptoms": "Clean, smooth green leaves with uniform texture and no visible lesions or discoloration.",
        "recommendation": "Continue standard orchard management, balanced irrigation, and periodic crop monitoring."
    },

    "Potato___Early_blight": {
        "description": "Early blight is a common fungal leaf spot disease caused by Alternaria solani in potatoes.",
        "symptoms": "Dark brown circular spots with concentric target-like rings surrounded by yellow leaf halos.",
        "recommendation": "Practice crop rotation, mulch soil base to reduce spore splash, avoid overhead watering, and prune lower infected leaves."
    },

    "Potato___healthy": {
        "description": "No disease detected. Potato plant foliage is healthy and active.",
        "symptoms": "Vibrant green foliage free of spots, blighting, or wilting.",
        "recommendation": "Maintain consistent soil moisture, soil hilling practices, and routine pest scouting."
    },

    "Potato___Late_blight": {
        "description": "Late blight is a destructive water mold (Oomycete) disease caused by Phytophthora infestans.",
        "symptoms": "Large, dark water-soaked lesions on leaf tips and edges, often with delicate white fungal fuzz beneath in humid weather.",
        "recommendation": "Remove and dispose of infected plants promptly to prevent field spread, ensure good soil drainage, and consult local extension guidelines."
    },

    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas species affecting tomato leaves, stems, and fruit.",
        "symptoms": "Small, dark, water-soaked leaf spots that dry into dark brown scabbed lesions, causing leaf yellowing and drop.",
        "recommendation": "Avoid working in foliage when wet, use disease-free certified seeds, practice multi-year rotation, and avoid overhead irrigation."
    },

    "Tomato___Early_blight": {
        "description": "Early blight is a widespread fungal disease caused by Alternaria solani affecting tomatoes.",
        "symptoms": "Dark brown spots with concentric ring patterns on older lower leaves, leading to yellowing and leaf loss.",
        "recommendation": "Stake plants for upright growth, mulch soil base, prune affected lower leaves, and maintain plant spacing for air circulation."
    },

    "Tomato___healthy": {
        "description": "No disease detected. Tomato plant exhibits healthy, vigorous growth.",
        "symptoms": "Deep green leaves with normal morphology and no signs of spotting or necrosis.",
        "recommendation": "Maintain regular watering at plant base, proper trellising, and routine visual crop inspections."
    },

    "Tomato___Late_blight": {
        "description": "Late blight is a serious infection caused by Phytophthora infestans capable of rapidly affecting tomato crops.",
        "symptoms": "Irregular dark brown water-soaked leaf spots, white downy growth on lower leaf surfaces during humid weather, and dark stem lesions.",
        "recommendation": "Promptly remove and destroy infected plant material, avoid wet foliage, and practice multi-year crop rotation."
    },

    "Tomato___Leaf_Mold": {
        "description": "Leaf mold is a fungal disease caused by Passalora fulva (Cladosporium fulvum), prevalent in high humidity.",
        "symptoms": "Pale yellow spots on upper leaf surfaces corresponding to velvety olive-green to light brown mold beneath.",
        "recommendation": "Increase airflow and ventilation around plants, reduce humidity, space plants adequately, and prune lower dense foliage."
    },

    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot is a foliage disease caused by the fungus Septoria lycopersici.",
        "symptoms": "Numerous small circular spots with dark brown margins and tan/grey centers containing tiny black fruiting bodies.",
        "recommendation": "Remove lower infected foliage, apply ground mulch to prevent soil splash, and avoid overhead sprinkler watering."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Damage caused by Two-Spotted Spider Mites (Tetranychus urticae), microscopic sap-sucking arachnids.",
        "symptoms": "Fine yellow or white stippling on leaf surfaces, bronze-yellow foliage discoloration, and fine silk webbing underneath.",
        "recommendation": "Rinse leaf undersides with water sprays, apply neem oil or insecticidal soap if needed, and foster beneficial predatory insects."
    },

    "Tomato___Target_Spot": {
        "description": "Target spot is a fungal foliage disease caused by Corynespora cassiicola.",
        "symptoms": "Small pinpoint spots that expand into brown circular lesions with light tan centers and concentric rings.",
        "recommendation": "Maintain row spacing to facilitate leaf drying, prune lower infected leaves, and practice crop rotation."
    },

    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato Mosaic Virus (ToMV) is a persistent viral pathogen transmitted mechanically by contact.",
        "symptoms": "Mottled light and dark green mosaic leaf patterns, leaf distortion, stunting, and puckered growth.",
        "recommendation": "Remove and destroy infected plants (viruses cannot be cured chemically), sanitize tools thoroughly, and wash hands before handling plants."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is a viral disease transmitted primarily by silverleaf whiteflies.",
        "symptoms": "Severe upward leaf curling and yellowing along leaf margins, leaf size reduction, and stunted plant growth.",
        "recommendation": "Control whitefly vectors using insect netting or reflective mulches, remove infected reservoir plants, and use resistant varieties."
    }
}


# 3. Image Quality Assessment Function
def check_image_quality(pil_image):
    """
    Evaluates resolution, darkness, overexposure,
    and sharpness of uploaded leaf image.

    Returns:
        (is_valid: bool, warning_message: str or None)
    """

    w, h = pil_image.size

    # Resolution check
    if w < 100 or h < 100:
        return (
            False,
            "The uploaded image resolution is too small (< 100 × 100 px). "
            "Please upload a higher resolution crop leaf photo."
        )

    img_np = np.array(
        pil_image.convert("RGB"),
        dtype=np.float32
    )

    gray = np.mean(img_np, axis=2)

    # 1. Darkness check using dark-pixel ratio and 75th percentile
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

    # 2. Overexposure check
    mean_brightness = float(
        np.mean(img_np)
    )

    if mean_brightness > 225.0:
        return (
            False,
            "The uploaded image is overexposed / too bright. "
            "Please upload a photo taken under balanced lighting."
        )

    # 3. Sharpness / detail check
    gy, gx = np.gradient(gray)

    detail_score = float(
        np.mean(
            np.sqrt(gx ** 2 + gy ** 2)
        )
    )

    if detail_score < 3.5:
        return (
            False,
            "The uploaded image appears blurry or lacks fine detail. "
            "Please upload a clear, focused close-up of the crop leaf."
        )

    return True, None


# 4. Grad-CAM Generation
def generate_gradcam_overlay(
    model,
    input_data,
    class_index,
    original_image
):
    """
    Generates a Grad-CAM visualization using the trained
    MobileNetV2 model without changing model weights.
    """

    # Confirmed final convolutional feature layer
    grad_layer = model.get_layer("out_relu")

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

    # Global average of gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # Weighted activation map
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # Keep positive activations
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # Resize original image
    original = original_image.convert(
        "RGB"
    ).resize(
        (448, 448),
        Image.Resampling.LANCZOS
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

    heat_array = np.asarray(
        heat_img,
        dtype=np.float32
    ) / 255.0

    # Build red/yellow activation overlay
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
        190 * heat_array
    )

    heat_overlay = Image.fromarray(
        heat_rgba,
        mode="RGBA"
    )

    result = Image.alpha_composite(
        original.convert("RGBA"),
        heat_overlay
    )

    return result.convert("RGB")


# 5. Cached Model Loading Function
@st.cache_resource
def load_leaf_predictor():
    predictor = LeafDiseasePredictor()
    predictor.load_model()
    return predictor


# 6. Initialize Session State
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_file_hash" not in st.session_state:
    st.session_state.analysis_file_hash = None

if "analysis_image_bytes" not in st.session_state:
    st.session_state.analysis_image_bytes = None

if "gradcam_image" not in st.session_state:
    st.session_state.gradcam_image = None


# 7. Sidebar Setup
with st.sidebar:
    st.title("🌱 LeafGuard AI")
    st.caption("Intelligent Agricultural Diagnostics")
    st.markdown("---")

    st.subheader("📊 System Specs")

    st.markdown("""
    - **Model Architecture:** MobileNetV2
    - **Target Classes:** 17 Supported
    - **Test Accuracy:** 87.70%
    - **Weighted F1-Score:** 87.36%
    """)

    st.markdown("---")

    st.subheader("🌾 Supported Crops")

    st.markdown("""
    - 🍎 **Apple** *(Scab, Black Rot, Rust, Healthy)*
    - 🥔 **Potato** *(Early Blight, Late Blight, Healthy)*
    - 🍅 **Tomato** *(Bacterial Spot, Blight, Mold, Mites, Viruses, Healthy)*
    """)

    st.markdown("---")

    st.subheader("💡 3-Step Guide")

    st.markdown("""
    1. **Upload** a leaf image (JPG/PNG).
    2. Click **Analyze Leaf**.
    3. Review **Diagnosis & Action Plan**.
    4. Use **Grad-CAM** to see regions influencing the prediction.
    """)


# 8. Main Dashboard Header
st.title("🌱 LeafGuard AI Diagnostic Dashboard")

st.subheader(
    "Intelligent Crop Health & Disease Detection System"
)

st.write(
    "Upload a crop leaf image to analyze plant health, "
    "identify disease conditions, and receive real-time "
    "confidence scores generated by our MobileNetV2 deep learning model."
)

st.divider()


# 9. Two-Column Dashboard Layout
col_left, col_right = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with col_left:

    st.markdown(
        "### 📷 1. Image Upload & Preview"
    )

    uploaded_file = st.file_uploader(
        "Choose a crop leaf image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        key="leaf_uploader"
    )

    if uploaded_file is not None:

        try:

            # Read uploaded bytes once
            file_bytes = uploaded_file.getvalue()

            # Create fingerprint so results persist correctly
            # when the Grad-CAM button causes Streamlit reruns.
            current_file_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            # If user uploads a different image,
            # clear previous analysis.
            if (
                st.session_state.analysis_file_hash
                != current_file_hash
            ):

                st.session_state.analysis_result = None
                st.session_state.analysis_image_bytes = None
                st.session_state.gradcam_image = None

                st.session_state.analysis_file_hash = (
                    current_file_hash
                )

            image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Crop Leaf Image",
                width="stretch"
            )

            # Pre-inference quality check
            is_quality_ok, quality_msg = (
                check_image_quality(image)
            )

            # Display image specifications
            if is_quality_ok:

                st.info(
                    f"📁 **Filename:** `{uploaded_file.name}`  \n"
                    f"📐 **Resolution:** {image.width} × {image.height} px  \n"
                    f"🎨 **Quality Check:** Pass "
                    f"(Balanced Lighting & Detail)"
                )

            else:

                st.warning(
                    f"📁 **Filename:** `{uploaded_file.name}` "
                    f"({image.width}×{image.height} px)  \n"
                    f"⚠️ **Quality Notice:** {quality_msg}"
                )

            st.markdown("---")

            analyze_btn = st.button(
                "🧪 Analyze Leaf",
                type="primary",
                width="stretch"
            )

            # ------------------------------------------------
            # EXECUTE ANALYSIS
            # ------------------------------------------------

            if analyze_btn:

                if not is_quality_ok:

                    # Clear any previous analysis
                    st.session_state.analysis_result = None
                    st.session_state.analysis_image_bytes = None
                    st.session_state.gradcam_image = None

                else:

                    with st.spinner(
                        "Executing MobileNetV2 deep learning inference..."
                    ):

                        try:

                            predictor = load_leaf_predictor()

                            # Reset file pointer
                            uploaded_file.seek(0)

                            result = predictor.predict(
                                uploaded_file
                            )

                            # Store analysis so that it survives
                            # later Streamlit reruns.
                            st.session_state.analysis_result = result

                            st.session_state.analysis_image_bytes = (
                                file_bytes
                            )

                            # New prediction means new Grad-CAM.
                            st.session_state.gradcam_image = None

                        except Exception as err:

                            st.error(
                                f"❌ **Prediction Error:** "
                                f"An error occurred while running "
                                f"inference ({err}). "
                                f"Please verify model file "
                                f"`models/leafguard_mobilenetv2.h5` exists."
                            )

                            st.session_state.analysis_result = None
                            st.session_state.analysis_image_bytes = None
                            st.session_state.gradcam_image = None

        except Exception as err:

            st.error(
                f"❌ **Invalid Image File:** "
                f"Could not read uploaded file. "
                f"Detail: {err}"
            )

            analyze_btn = False

    else:

        st.info(
            "Please upload a leaf image above to enable analysis."
        )

        analyze_btn = False

        # Clear previous state when nothing is uploaded.
        st.session_state.analysis_result = None
        st.session_state.analysis_file_hash = None
        st.session_state.analysis_image_bytes = None
        st.session_state.gradcam_image = None


# ============================================================
# RIGHT COLUMN
# ============================================================

with col_right:

    st.markdown(
        "### 🔬 2. Diagnostic Analysis & Advisory"
    )

    # No image uploaded
    if uploaded_file is None:

        st.info(
            "👈 Upload a crop leaf image on the left panel "
            "to begin diagnostic analysis."
        )

    # Image uploaded, but no completed analysis yet
    elif st.session_state.analysis_result is None:

        st.info(
            "👆 Click **Analyze Leaf** on the left panel "
            "to execute model inference."
        )

    # ========================================================
    # DISPLAY STORED ANALYSIS
    # ========================================================
    else:

        result = st.session_state.analysis_result

        # Reconstruct analyzed image from session state
        analysis_image = Image.open(
            io.BytesIO(
                st.session_state.analysis_image_bytes
            )
        ).convert("RGB")

        raw_class_name = result[
            "predicted_class_name"
        ]

        confidence_val = float(
            result["confidence"]
        )

        confidence_pct = (
            confidence_val * 100
        )

        # ----------------------------------------------------
        # Top-1 and Top-2 probability analysis
        # ----------------------------------------------------

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
            top1_prob - top2_prob
        )

        # ----------------------------------------------------
        # Uncertainty safeguard
        # ----------------------------------------------------

        is_unrecognized = (
            top1_prob < 0.50
            or margin < 0.20
        )

        # ----------------------------------------------------
        # Clean display name
        # ----------------------------------------------------

        formatted_name = (
            raw_class_name
            .replace("___", " - ")
            .replace("_", " ")
        )

        is_healthy = (
            "healthy"
            in raw_class_name.lower()
        )

        # ----------------------------------------------------
        # Inference complete
        # ----------------------------------------------------

        st.success(
            "🔬 **Inference Complete!**"
        )

        st.caption(
            "ℹ️ **Scope Note:** LeafGuard AI v1 is trained "
            "for Apple, Potato, and Tomato leaf conditions."
        )

        st.markdown("---")

        # ====================================================
        # HEALTH STATUS CARD
        # ====================================================

        if is_unrecognized:

            st.warning(
                "### Status: ⚠️ "
                "Unrecognized Crop or Foliage"
            )

            st.markdown(
                "**The model cannot confidently match this image "
                "to the 17 supported Apple, Potato, or Tomato "
                "conditions. Please upload a clear close-up leaf "
                "image from a supported crop.**"
            )

            st.markdown(
                f"#### Nearest Model Match (Uncertain):\n"
                f"### **[Uncertain] {formatted_name}**"
            )

        elif confidence_pct < 70.0:

            st.warning(
                "### Status: ⚠️ "
                "Low-Confidence Prediction"
            )

            st.markdown(
                f"#### Estimated Condition:\n"
                f"### **[Low Confidence] {formatted_name}**"
            )

        else:

            if is_healthy:

                st.success(
                    "### Status: 🌱 Healthy Crop Tissue"
                )

            else:

                st.error(
                    "### Status: ⚠️ Crop Disease Detected"
                )

            st.markdown(
                f"#### Diagnosed Condition:\n"
                f"### **{formatted_name}**"
            )

        st.markdown("---")

        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.markdown(
            "#### Model Confidence Score"
        )

        st.metric(
            label="Model Certainty",
            value=f"{confidence_pct:.2f}%"
        )

        st.progress(
            min(
                max(
                    confidence_val,
                    0.0
                ),
                1.0
            )
        )

        # ====================================================
        # WARNING BANNERS
        # ====================================================

        if is_unrecognized:

            st.warning(
                "⚠️ **Unrecognized Crop or Foliage Notice:** "
                "The model cannot confidently match this image "
                "to the 17 supported Apple, Potato, or Tomato "
                "conditions. Please upload a clear close-up leaf "
                "image from a supported crop."
            )

        elif confidence_pct < 70.0:

            st.warning(
                "⚠️ **Low-Confidence Warning (< 70%):** "
                "The model certainty is below 70%. The result "
                "is presented as an estimated prediction. "
                "The image might have glare, blur, unusual "
                "angles, or an unrepresented leaf condition. "
                "Please consider uploading another clear, "
                "well-lit close-up photo for re-evaluation."
            )

        st.markdown("---")

        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

        info = DISEASE_INFO.get(
            raw_class_name,
            {
                "description":
                    "No detailed description available.",

                "symptoms":
                    "N/A",

                "recommendation":
                    "Consult a local agricultural extension "
                    "specialist for guidance."
            }
        )

        st.markdown(
            "#### 📖 Condition Overview"
        )

        st.write(
            info["description"]
        )

        st.markdown(
            "#### 🔍 Common Symptoms"
        )

        st.write(
            info["symptoms"]
        )

        st.markdown(
            "#### 🛡️ Recommended General Action"
        )

        st.write(
            info["recommendation"]
        )

        # ====================================================
        # GRAD-CAM
        # ====================================================

        st.markdown("---")

        st.markdown(
            "#### 🔬 Explain This Prediction"
        )

        st.caption(
            "Highlighted regions indicate areas that influenced "
            "the model's prediction. This visualization explains "
            "the model output; it does not prove that a specific "
            "region contains the disease."
        )

        if not is_unrecognized:

            explain_btn = st.button(
                "🔬 Generate Grad-CAM Explanation",
                width="stretch",
                key="gradcam_button"
            )

            if explain_btn:

                try:

                    # Re-create preprocessing input using
                    # the exact same preprocessing pipeline.
                    gradcam_input = (
                        preprocess_single_image(
                            io.BytesIO(
                                st.session_state.analysis_image_bytes
                            ),
                            target_size=IMAGE_SIZE
                        )
                    )

                    predictor = load_leaf_predictor()

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

                    # Persist image across Streamlit reruns.
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
                "Grad-CAM is unavailable for an "
                "unrecognized/uncertain image because the "
                "model could not confidently match it to "
                "a supported condition."
            )

        # ====================================================
        # DISPLAY GRAD-CAM RESULT
        # ====================================================

        if (
            st.session_state.gradcam_image
            is not None
        ):

            st.markdown("#### 🧠 Model Attention Visualization")

            st.info(
                "The visualization highlights regions that "
                "contributed to the model's prediction. "
                "Use it as an explanation aid, not as proof "
                "of a disease diagnosis."
            )

            col_original, col_gradcam = (
                st.columns(2)
            )

            with col_original:

                st.image(
                    analysis_image,
                    caption="Original Leaf",
                    use_container_width=True
                )

            with col_gradcam:

                st.image(
                    st.session_state.gradcam_image,
                    caption="Grad-CAM Explanation",
                    use_container_width=True
                )