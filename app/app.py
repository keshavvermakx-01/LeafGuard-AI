import sys
from pathlib import Path

# Add project root directory to sys.path for module resolution on Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import streamlit as st
from PIL import Image
from src.predict import LeafDiseasePredictor

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


# 3. Image Quality Assessment Function (Resolution, Darkness, Overexposure & Detail)
def check_image_quality(pil_image):
    """
    Evaluates resolution, darkness (via dark_pixel_ratio & p75_brightness), overexposure,
    and sharpness of uploaded leaf image using PIL and NumPy.
    Returns (is_valid: bool, warning_message: str or None)
    """
    w, h = pil_image.size
    if w < 100 or h < 100:
        return False, "The uploaded image resolution is too small (< 100 × 100 px). Please upload a higher resolution crop leaf photo."

    img_np = np.array(pil_image.convert("RGB"), dtype=np.float32)
    gray = np.mean(img_np, axis=2)

    # 1. Advanced Darkness Check (Dark Pixel Ratio & 75th Percentile Intensity)
    dark_pixel_ratio = float(np.mean(gray < 40.0))
    p75_brightness = float(np.percentile(gray, 75))

    if dark_pixel_ratio > 0.65 or p75_brightness < 45.0:
        return False, "The uploaded image is too dark or underexposed. Please upload a well-lit photo of the crop leaf."

    # 2. Overexposure Check
    mean_brightness = float(np.mean(img_np))
    if mean_brightness > 225.0:
        return False, "The uploaded image is overexposed / too bright. Please upload a photo taken under balanced lighting."

    # 3. Sharpness / detail check via image spatial gradients
    gy, gx = np.gradient(gray)
    detail_score = float(np.mean(np.sqrt(gx**2 + gy**2)))
    if detail_score < 3.5:
        return False, "The uploaded image appears blurry or lacks fine detail. Please upload a clear, focused close-up of the crop leaf."

    return True, None


# 4. Cached Model Loading Function
@st.cache_resource
def load_leaf_predictor():
    predictor = LeafDiseasePredictor()
    predictor.load_model()
    return predictor


# 5. Sidebar Setup
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
    """)


# 6. Main Dashboard Header
st.title("🌱 LeafGuard AI Diagnostic Dashboard")
st.subheader("Intelligent Crop Health & Disease Detection System")
st.write(
    "Upload a crop leaf image to analyze plant health, identify disease conditions, "
    "and receive real-time confidence scores generated by our MobileNetV2 deep learning model."
)
st.divider()

# 7. Two-Column Dashboard Layout
col_left, col_right = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: Input & Upload ---
with col_left:
    st.markdown("### 📷 1. Image Upload & Preview")
    
    uploaded_file = st.file_uploader(
        "Choose a crop leaf image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        key="leaf_uploader"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Crop Leaf Image", width="stretch")

            # Run pre-inference image quality check
            is_quality_ok, quality_msg = check_image_quality(image)

            # Display image specs & quality status
            if is_quality_ok:
                st.info(
                    f"📁 **Filename:** `{uploaded_file.name}`  \n"
                    f"📐 **Resolution:** {image.width} × {image.height} px  \n"
                    f"🎨 **Quality Check:** Pass (Balanced Lighting & Detail)"
                )
            else:
                st.warning(
                    f"📁 **Filename:** `{uploaded_file.name}` ({image.width}×{image.height} px)  \n"
                    f"⚠️ **Quality Notice:** {quality_msg}"
                )

            st.markdown("---")
            analyze_btn = st.button("🧪 Analyze Leaf", type="primary", width="stretch")
        except Exception as err:
            st.error(f"❌ **Invalid Image File:** Could not read uploaded file. Detail: {err}")
            analyze_btn = False
    else:
        st.info("Please upload a leaf image above to enable analysis.")
        analyze_btn = False

# --- RIGHT COLUMN: Diagnostic Results & Guidance ---
with col_right:
    st.markdown("### 🔬 2. Diagnostic Analysis & Advisory")

    if uploaded_file is None:
        st.info("👈 Upload a crop leaf image on the left panel to begin diagnostic analysis.")
    elif not analyze_btn:
        st.info("👆 Click **Analyze Leaf** on the left panel to execute model inference.")
    else:
        # Check image quality before executing model inference
        is_quality_ok, quality_msg = check_image_quality(image)
        if not is_quality_ok:
            st.warning(
                f"📷 **Image Quality Check Notice:**\n\n{quality_msg}\n\n"
                "Please re-upload a clearer, well-lit close-up leaf photo for reliable diagnostics."
            )
        else:
            with st.spinner("Executing MobileNetV2 deep learning inference..."):
                try:
                    predictor = load_leaf_predictor()
                    result = predictor.predict(uploaded_file)
                except Exception as err:
                    st.error(
                        f"❌ **Prediction Error:** An error occurred while running inference ({err}). "
                        "Please verify model file `models/leafguard_mobilenetv2.h5` exists."
                    )
                    st.stop()

                raw_class_name = result["predicted_class_name"]
                confidence_val = float(result["confidence"])
                confidence_pct = confidence_val * 100

                # Analyze top-1 and top-2 probabilities for probability margin / diffusion check
                all_probs = sorted([float(p) for p in result["all_probabilities"]], reverse=True)
                top1_prob = all_probs[0] if len(all_probs) > 0 else confidence_val
                top2_prob = all_probs[1] if len(all_probs) > 1 else 0.0
                margin = top1_prob - top2_prob

                # Flag prediction as unrecognized foliage if top-1 < 0.50 OR margin < 0.20
                is_unrecognized = (top1_prob < 0.50) or (margin < 0.20)

                # Format raw label for clean presentation
                formatted_name = raw_class_name.replace("___", " - ").replace("_", " ")
                is_healthy = "healthy" in raw_class_name.lower()

                st.success("🔬 **Inference Complete!**")
                st.caption("ℹ️ **Scope Note:** LeafGuard AI v1 is trained for Apple, Potato, and Tomato leaf conditions.")
                st.markdown("---")

                # Health Status Card & Uncertainty Safeguard
                if is_unrecognized:
                    st.warning("### Status: ⚠️ Unrecognized Crop or Foliage")
                    st.markdown(
                        "**The model cannot confidently match this image to the 17 supported Apple, Potato, or Tomato conditions. "
                        "Please upload a clear close-up leaf image from a supported crop.**"
                    )
                    st.markdown(f"#### Nearest Model Match (Uncertain):\n### **[Uncertain] {formatted_name}**")
                elif confidence_pct < 70.0:
                    st.warning("### Status: ⚠️ Low-Confidence Prediction")
                    st.markdown(f"#### Estimated Condition:\n### **[Low Confidence] {formatted_name}**")
                else:
                    if is_healthy:
                        st.success("### Status: 🌱 Healthy Crop Tissue")
                    else:
                        st.error("### Status: ⚠️ Crop Disease Detected")

                    st.markdown(f"#### Diagnosed Condition:\n### **{formatted_name}**")

                st.markdown("---")

                # Confidence Metric & Visual Progress Bar
                st.markdown("#### Model Confidence Score")
                st.metric(label="Model Certainty", value=f"{confidence_pct:.2f}%")
                st.progress(min(max(confidence_val, 0.0), 1.0))

                # Warning Banners by Priority (1. is_unrecognized -> 2. confidence_pct < 70.0 -> 3. normal)
                if is_unrecognized:
                    st.warning(
                        "⚠️ **Unrecognized Crop or Foliage Notice:** "
                        "The model cannot confidently match this image to the 17 supported Apple, Potato, or Tomato conditions. "
                        "Please upload a clear close-up leaf image from a supported crop."
                    )
                elif confidence_pct < 70.0:
                    st.warning(
                        "⚠️ **Low-Confidence Warning (< 70%):** "
                        "The model certainty is below 70%. The result is presented as an estimated prediction. "
                        "The image might have glare, blur, unusual angles, or an unrepresented leaf condition. "
                        "Please consider uploading another clear, well-lit close-up photo for re-evaluation."
                    )

                st.markdown("---")

                # Structured Disease Information & Advisory Guidance
                info = DISEASE_INFO.get(raw_class_name, {
                    "description": "No detailed description available.",
                    "symptoms": "N/A",
                    "recommendation": "Consult a local agricultural extension specialist for guidance."
                })

                st.markdown("#### 📖 Condition Overview")
                st.write(info["description"])

                st.markdown("#### 🔍 Common Symptoms")
                st.write(info["symptoms"])

                st.markdown("#### 🛡️ Recommended General Action")
                st.write(info["recommendation"])
