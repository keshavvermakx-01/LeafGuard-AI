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

# PDF report generation
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape
from datetime import datetime

from src.config import IMAGE_SIZE
from src.predict import LeafDiseasePredictor
from src.preprocessing import preprocess_single_image

@st.cache_resource(show_spinner=False)
def load_leaf_predictor():
    return LeafDiseasePredictor()
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
# STREAMLIT SESSION STATE
# ============================================================

# Initialize every state key before any later code reads it.
# This prevents AttributeError/KeyError on a fresh Streamlit session.
if "analysis_file_hash" not in st.session_state:
    st.session_state.analysis_file_hash = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_image_bytes" not in st.session_state:
    st.session_state.analysis_image_bytes = None

if "gradcam_image" not in st.session_state:
    st.session_state.gradcam_image = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if "batch_image_bytes" not in st.session_state:
    st.session_state.batch_image_bytes = {}

if "batch_file_signature" not in st.session_state:
    st.session_state.batch_file_signature = None

if "analysis_stage" not in st.session_state:
    st.session_state.analysis_stage = 1

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = None


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

def generate_gradcam_explanation(
    model,
    input_data,
    class_index,
    original_image
):
    """
    Enhanced Grad-CAM:
    - Generates normalized activation heatmap
    - Creates a colored heatmap image
    - Creates an overlay
    - Produces a simple natural-language explanation
    """

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

        class_score = predictions[:, class_index]

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

    # Slight contrast enhancement
    heatmap = np.power(
        np.clip(
            heatmap,
            0.0,
            1.0
        ),
        0.80
    )

    # --------------------------------------------------------
    # Convert original image
    # --------------------------------------------------------

    original = (
        original_image
        .convert("RGB")
        .resize(
            (448, 448),
            Image.Resampling.LANCZOS
        )
    )

    # --------------------------------------------------------
    # Resize 7x7 heatmap
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create blue -> cyan -> yellow -> red heatmap
    # --------------------------------------------------------

    heat_rgb = np.zeros(
        (
            original.height,
            original.width,
            3
        ),
        dtype=np.uint8
    )

    # Low activation: blue
    # Medium: cyan/yellow
    # High: red
    r = np.clip(
        255 * (heat_array * 1.8 - 0.25),
        0,
        255
    )

    g = np.clip(
        255 * (1.4 - np.abs(heat_array - 0.5) * 2.2),
        0,
        255
    )

    b = np.clip(
        255 * (1.0 - heat_array * 1.8),
        0,
        255
    )

    heat_rgb[..., 0] = np.uint8(r)
    heat_rgb[..., 1] = np.uint8(g)
    heat_rgb[..., 2] = np.uint8(b)

    heatmap_image = Image.fromarray(
        heat_rgb,
        mode="RGB"
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    alpha = np.uint8(
        210 * heat_array
    )

    heat_rgba = np.dstack(
        [
            heat_rgb,
            alpha
        ]
    )

    heat_overlay = Image.fromarray(
        heat_rgba,
        mode="RGBA"
    )

    overlay_image = Image.alpha_composite(
        original.convert("RGBA"),
        heat_overlay
    ).convert("RGB")

    # --------------------------------------------------------
    # Analyze where the model focused
    # --------------------------------------------------------

    max_activation = float(
        np.max(heatmap)
    )

    threshold = max_activation * 0.60

    active_pixels = np.argwhere(
        heatmap >= threshold
    )

    if len(active_pixels) == 0:

        focus_y = heatmap.shape[0] / 2
        focus_x = heatmap.shape[1] / 2

    else:

        focus_y = float(
            np.mean(
                active_pixels[:, 0]
            )
        )

        focus_x = float(
            np.mean(
                active_pixels[:, 1]
            )
        )

    h, w = heatmap.shape

    # Horizontal region
    if focus_x < w / 3:
        horizontal = "left"

    elif focus_x > (2 * w / 3):
        horizontal = "right"

    else:
        horizontal = "central"

    # Vertical region
    if focus_y < h / 3:
        vertical = "upper"

    elif focus_y > (2 * h / 3):
        vertical = "lower"

    else:
        vertical = "middle"

    if vertical == "middle" and horizontal == "central":
        focus_region = "central portion"

    elif horizontal == "central":
        focus_region = f"{vertical}-central portion"

    elif vertical == "middle":
        focus_region = f"middle-{horizontal} portion"

    else:
        focus_region = f"{vertical}-{horizontal} portion"

    # --------------------------------------------------------
    # Attention coverage / explanation strength
    # --------------------------------------------------------

    # The heatmap is normalized, so its maximum is approximately 1.0.
    # Coverage above a fixed threshold is more informative than peak value.
    attention_coverage = float(
        np.mean(heatmap >= 0.60) * 100.0
    )

    if attention_coverage <= 5.0:

        strength_text = (
            "The model's attention is relatively concentrated in a small area."
        )

    elif attention_coverage <= 15.0:

        strength_text = (
            "The model shows a moderate concentration of attention across the leaf."
        )

    else:

        strength_text = (
            "The model's attention is distributed across a broader portion of the image."
        )

    explanation = (
        f"The model's strongest activation is concentrated "
        f"in the {focus_region} of the leaf. "
        f"{strength_text} "
        "These highlighted regions contributed more strongly "
        "to the selected prediction."
    )

    return {
        "original": original,
        "heatmap": heatmap_image,
        "overlay": overlay_image,
        "heatmap_array": heatmap,
        "focus_region": focus_region,
        "attention_coverage": attention_coverage,
        "explanation": explanation
    }

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
    "AI-powered leaf health analysis for Apple, Potato, and Tomato, "
    "with confidence checks and visual explanations."
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
# PDF REPORT HELPERS / BUILDERS
# Defined before the UI code that generates reports.
# ============================================================

def _pdf_safe_text(value):
    """Normalize common Unicode punctuation for standard PDF fonts."""
    text = str(value if value is not None else "")
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00d7": "x",
        "\u2022": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _pil_png_stream(pil_image):
    stream = io.BytesIO()
    pil_image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def _reportlab_image(pil_image, max_width, max_height):
    stream = _pil_png_stream(pil_image)
    width, height = pil_image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions for PDF report.")
    scale = min(max_width / width, max_height / height)
    return RLImage(stream, width=width * scale, height=height * scale)


def build_diagnosis_pdf(result, analysis_image_bytes, gradcam_data=None):
    """Create a downloadable single-image LeafGuard diagnosis report."""

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="LeafGuard AI Diagnosis Report",
        author="LeafGuard AI",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "LeafGuardTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "LeafGuardSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#5B6573"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "LeafGuardHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "LeafGuardBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "LeafGuardSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#5B6573"),
    )

    def p(text, style=body_style):
        return Paragraph(escape(_pdf_safe_text(text)), style)

    raw_class = result.get("predicted_class_name", "Unknown")
    formatted = _pdf_safe_text(raw_class).replace("___", " - ").replace("_", " ")
    if " - " in formatted:
        crop = formatted.split(" - ", 1)[0]
        condition = formatted.split(" - ", 1)[1]
    else:
        crop = "Unknown crop"
        condition = formatted

    confidence = float(result.get("confidence", 0.0)) * 100.0
    all_probs = sorted(
        [float(x) for x in result.get("all_probabilities", [])],
        reverse=True,
    )
    top2 = all_probs[1] if len(all_probs) > 1 else 0.0
    margin = (all_probs[0] if all_probs else confidence / 100.0) - top2
    is_unrecognized = (
        (all_probs[0] if all_probs else confidence / 100.0) < 0.50
        or margin < 0.20
    )
    is_healthy = "healthy" in raw_class.lower()

    info = DISEASE_INFO.get(raw_class, {})
    description = info.get("description", "No detailed condition description is available.")
    symptoms = info.get("symptoms", "No detailed symptom information is available.")
    recommendation = info.get("recommendation", "Consult a local agricultural extension specialist for guidance.")

    status = (
        "Unrecognized / uncertain image"
        if is_unrecognized
        else "Healthy - no disease detected"
        if is_healthy
        else "Supported crop disease detected"
    )

    story = []
    story.append(Paragraph("LEAFGUARD AI", title_style))
    story.append(Paragraph("Plant Health Analysis Report", subtitle_style))

    try:
        analysis_image = Image.open(io.BytesIO(analysis_image_bytes)).convert("RGB")
        story.append(
            _reportlab_image(analysis_image, max_width=4.8 * inch, max_height=3.55 * inch)
        )
        story.append(Spacer(1, 8))
    except Exception:
        story.append(p("The analyzed leaf image could not be embedded in the report.", small_style))

    summary_data = [
        [p("Analysis date", small_style), p(datetime.now().strftime("%d %B %Y, %H:%M"), body_style)],
        [p("Crop", small_style), p(crop, body_style)],
        [p("Prediction", small_style), p(condition, body_style)],
        [p("Model confidence", small_style), p(f"{confidence:.2f}%", body_style)],
        [p("Top-1 vs Top-2 margin", small_style), p(f"{margin:.2f}", body_style)],
        [p("Analysis status", small_style), p(status, body_style)],
        [p("Image quality", small_style), p("Passed", body_style)],
    ]

    summary_table = Table(summary_data, colWidths=[1.65 * inch, 5.15 * inch])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F6F8")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D7DEE5")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E3E8ED")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(summary_table)

    story.append(Paragraph("Condition Overview", heading_style))
    story.append(p(description))

    story.append(Paragraph("Common Symptoms", heading_style))
    story.append(p(symptoms))

    story.append(Paragraph("Recommended General Action", heading_style))
    story.append(p(recommendation))

    if isinstance(gradcam_data, dict):
        story.append(PageBreak())
        story.append(Paragraph("Explainability - Grad-CAM", heading_style))

        # Put the three visualizations in equal-width cells with their captions underneath.
        visual_cells = []
        for key, label in [
            ("original", "Original Leaf"),
            ("heatmap", "AI Attention Heatmap"),
            ("overlay", "Grad-CAM Overlay"),
        ]:
            image = gradcam_data.get(key)
            if image is not None:
                try:
                    cell_image = _reportlab_image(image, 2.02 * inch, 2.0 * inch)
                    visual_cells.append(
                        Table(
                            [
                                [cell_image],
                                [p(label, small_style)],
                            ],
                            colWidths=[2.08 * inch],
                        )
                    )
                except Exception:
                    visual_cells.append(
                        Table(
                            [
                                [p("Image unavailable", small_style)],
                                [p(label, small_style)],
                            ],
                            colWidths=[2.08 * inch],
                        )
                    )
            else:
                visual_cells.append(
                    Table(
                        [
                            [p("Image unavailable", small_style)],
                            [p(label, small_style)],
                        ],
                        colWidths=[2.08 * inch],
                    )
                )

        gradcam_table = Table(
            [visual_cells],
            colWidths=[2.18 * inch, 2.18 * inch, 2.18 * inch],
        )
        gradcam_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7DEE5")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E3E8ED")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ])
        )
        story.append(gradcam_table)
        story.append(Spacer(1, 8))
        story.append(
            p(
                f"Primary focus region: {gradcam_data.get('focus_region', 'not available')} - "
                f"Attention coverage: {float(gradcam_data.get('attention_coverage', 0.0)):.1f}%"
            )
        )
        story.append(
            p(
                gradcam_data.get(
                    "explanation",
                    "Grad-CAM explains model attention for the existing prediction.",
                )
            )
        )
    else:
        story.append(Paragraph("Explainability - Grad-CAM", heading_style))
        story.append(
            p(
                "Grad-CAM was not generated during this session. Return to Step 4 and generate the explanation before creating another report if you want the attention maps included."
            )
        )

    note_data = [[p("Important note", small_style), p("Grad-CAM explains which image regions influenced the model prediction. It does not prove that a highlighted region contains the disease or define an exact disease boundary. Model confidence is a score, not a guarantee of diagnosis.", small_style)]]
    note_table = Table(note_data, colWidths=[1.2 * inch, 5.6 * inch])
    note_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E6")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E8D9A8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.append(Spacer(1, 8))
    story.append(note_table)
    story.append(Spacer(1, 10))
    story.append(p("LeafGuard AI - MobileNetV2 - 17 Apple, Potato & Tomato conditions", small_style))

    doc.build(story)
    return buffer.getvalue()



# --------------------------------------------------------
# Batch-analysis PDF report
# --------------------------------------------------------
def build_batch_pdf(batch_results, batch_image_bytes=None):
    """Create a downloadable PDF report summarizing a batch analysis."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="LeafGuard AI Batch Analysis Report",
        author="LeafGuard AI",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BatchTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=TA_CENTER,
        spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "BatchSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.HexColor("#5B6573"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "BatchHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=15,
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BatchBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12,
        spaceAfter=4,
    )
    small_style = ParagraphStyle(
        "BatchSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#5B6573"),
    )
    table_style = ParagraphStyle(
        "BatchTable",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
    )

    def p(value, style=body_style):
        return Paragraph(escape(_pdf_safe_text(value)), style)

    rows = list(batch_results or [])
    total = len(rows)
    disease_count = sum(1 for r in rows if r.get("Status") == "Crop disease detected")
    healthy_count = sum(1 for r in rows if r.get("Status") == "No disease detected")
    uncertain_count = sum(1 for r in rows if r.get("Status") == "Uncertain")
    error_count = sum(1 for r in rows if r.get("Status") == "Analysis error")

    story = [
        Paragraph("LEAFGUARD AI", title_style),
        Paragraph("Batch Plant Health Analysis Report", subtitle_style),
    ]

    summary_cards = Table(
        [[
            p("Total Images", small_style),
            p("Disease Detected", small_style),
            p("Healthy", small_style),
            p("Uncertain", small_style),
            p("Errors", small_style),
        ], [
            p(str(total), heading_style),
            p(str(disease_count), heading_style),
            p(str(healthy_count), heading_style),
            p(str(uncertain_count), heading_style),
            p(str(error_count), heading_style),
        ]],
        colWidths=[1.32 * inch] * 5,
    )
    summary_cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F6F8")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D7DEE5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E3E8ED")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_cards)
    story.append(Spacer(1, 10))
    story.append(p(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}" , small_style))

    story.append(Paragraph("Batch Results", heading_style))
    table_data = [[
        p("Image", table_style),
        p("Crop", table_style),
        p("Prediction", table_style),
        p("Confidence", table_style),
        p("Status", table_style),
        p("Quality", table_style),
    ]]
    for row in rows:
        table_data.append([
            p(row.get("Image", ""), table_style),
            p(row.get("Crop", ""), table_style),
            p(row.get("Prediction", ""), table_style),
            p(row.get("Confidence", ""), table_style),
            p(row.get("Status", ""), table_style),
            p(row.get("Quality", ""), table_style),
        ])

    results_table = Table(
        table_data,
        colWidths=[2.05 * inch, 0.83 * inch, 1.25 * inch, 0.78 * inch, 1.32 * inch, 1.1 * inch],
        repeatRows=1,
    )
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E2430")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7DEE5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E3E8ED")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(results_table)

    # Add image details when bytes are available. No batch Grad-CAM is generated here.
    image_map = batch_image_bytes or {}
    detail_rows = [
        row for row in rows
        if row.get("Image") in image_map
    ]
    if detail_rows:
        story.append(PageBreak())
        story.append(Paragraph("Image Details", heading_style))
        story.append(
            p(
                "The sections below document each analyzed image. This batch report does not generate or include Grad-CAM visualizations."
            )
        )

        for idx, row in enumerate(detail_rows):
            if idx > 0:
                story.append(Spacer(1, 10))

            image = None
            try:
                image = Image.open(io.BytesIO(image_map[row.get("Image")])).convert("RGB")
            except Exception:
                image = None

            detail_left = []
            if image is not None:
                detail_left.append(_reportlab_image(image, 2.15 * inch, 1.85 * inch))
            else:
                detail_left.append(p("Image unavailable", small_style))

            detail_data = [
                [p("Image", small_style), p(row.get("Image", ""), body_style)],
                [p("Crop", small_style), p(row.get("Crop", ""), body_style)],
                [p("Prediction", small_style), p(row.get("Prediction", ""), body_style)],
                [p("Confidence", small_style), p(row.get("Confidence", ""), body_style)],
                [p("Status", small_style), p(row.get("Status", ""), body_style)],
                [p("Quality", small_style), p(row.get("Quality", ""), body_style)],
            ]
            detail_table = Table(detail_data, colWidths=[1.1 * inch, 3.7 * inch])
            detail_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F6F8")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7DEE5")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E3E8ED")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))

            row_table = Table([[detail_left[0], detail_table]], colWidths=[2.35 * inch, 4.9 * inch])
            row_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(row_table)

    story.append(Spacer(1, 10))
    story.append(p(
        "LeafGuard AI - MobileNetV2 - 17 Apple, Potato & Tomato conditions - Batch analysis summary"
    , small_style))

    doc.build(story)
    return buffer.getvalue()

# ============================================================
# 10. STEP 1 - UPLOAD
# ============================================================

if st.session_state.analysis_stage == 1:

    st.subheader(
        "📷 Step 1 — Upload & Prepare Your Leaf"
    )

    st.progress(
        0.25,
        text="Step 1 of 4 — Upload your leaf image"
    )

    st.info(
        "Choose a single leaf photo or analyze multiple leaf photos at once. "
        "Both options use the same LeafGuard quality checks and MobileNetV2 model."
    )

    single_col, batch_col = st.columns(
        2,
        gap="large"
    )

    # ------------------------------------------------------------
    # SINGLE PHOTO
    # ------------------------------------------------------------

    with single_col:

        with st.container(border=True):

            st.markdown("### 📷 Single Photo")
            st.caption("Analyze one leaf image")

            uploaded_file = st.file_uploader(
                "Choose one crop leaf image",
                type=["jpg", "jpeg", "png"],
                key="leaf_uploader"
            )

            single_quality_ok = False
            single_quality_msg = None
            single_file_bytes = None

            if uploaded_file is not None:

                try:

                    single_file_bytes = uploaded_file.getvalue()
                    current_file_hash = hashlib.sha256(single_file_bytes).hexdigest()

                    if st.session_state.analysis_file_hash != current_file_hash:

                        st.session_state.analysis_file_hash = current_file_hash
                        st.session_state.analysis_result = None
                        st.session_state.analysis_image_bytes = None
                        st.session_state.gradcam_image = None
                        st.session_state.chat_messages = []
                        st.session_state.chat_open = False
                        st.session_state.analysis_mode = None
                        st.session_state.analysis_stage = 1

                    image = Image.open(
                        io.BytesIO(single_file_bytes)
                    ).convert("RGB")

                    st.image(
                        image,
                        caption=f"Uploaded Leaf • {uploaded_file.name}",
                        width="stretch"
                    )

                    single_quality_ok, single_quality_msg = check_image_quality(image)

                    q1, q2 = st.columns(2)

                    with q1:
                        st.write(
                            f"**Resolution**  \n"
                            f"{image.width} × {image.height}px"
                        )

                    with q2:
                        if single_quality_ok:
                            st.success("✅ Quality passed")
                        else:
                            st.warning("⚠️ Quality issue")

                    if not single_quality_ok:
                        st.caption(single_quality_msg)

                except Exception as err:
                    st.error(f"❌ **Invalid Image:** {err}")

            else:
                st.caption("Upload one image to analyze a single leaf.")

    # ------------------------------------------------------------
    # BATCH ANALYSIS
    # ------------------------------------------------------------

    with batch_col:

        with st.container(border=True):

            st.markdown("### 📂 Batch Analysis")
            st.caption("Analyze multiple leaf images in one run")

            batch_files = st.file_uploader(
                "Choose multiple crop leaf images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="batch_leaf_uploader"
            )

            if batch_files:

                batch_signature = tuple(
                    (file.name, len(file.getvalue()))
                    for file in batch_files
                )

                if st.session_state.batch_file_signature != batch_signature:
                    st.session_state.batch_results = []
                    st.session_state.batch_image_bytes = {}
                    st.session_state.batch_file_signature = batch_signature
                    st.session_state.analysis_mode = None
                    st.session_state.analysis_stage = 1

                st.info(f"{len(batch_files)} image(s) selected.")

            else:
                st.session_state.batch_results = []
                st.session_state.batch_image_bytes = {}
                st.session_state.batch_file_signature = None
                st.caption("Select multiple images for batch analysis.")

    # ------------------------------------------------------------
    # ONE COMMON ANALYZE LEAF BUTTON
    # ------------------------------------------------------------

    st.write("")

    if uploaded_file is not None and batch_files:
        st.warning(
            "Please use either Single Photo or Batch Analysis, not both at the same time."
        )

    analyze_leaf_btn = st.button(
        "🧪 Analyze Leaf",
        type="primary",
        width="stretch",
        key="common_analyze_leaf_button"
    )

    if analyze_leaf_btn:

        single_selected = uploaded_file is not None
        batch_selected = bool(batch_files)

        if single_selected and batch_selected:

            st.error(
                "Please select only one analysis option: Single Photo or Batch Analysis."
            )

        elif single_selected:

            if not single_quality_ok:
                st.error(
                    "Analysis stopped because the image did not pass the quality checks."
                )

            else:
                with st.spinner("🔬 Running LeafGuard AI analysis..."):
                    try:
                        predictor = load_leaf_predictor()
                        uploaded_file.seek(0)
                        result = predictor.predict(uploaded_file)

                        st.session_state.analysis_result = result
                        st.session_state.analysis_image_bytes = single_file_bytes
                        st.session_state.gradcam_image = None
                        st.session_state.chat_messages = []
                        st.session_state.chat_open = False
                        st.session_state.analysis_mode = "single"
                        st.session_state.analysis_stage = 2

                        st.rerun()

                    except Exception as err:
                        st.error(f"❌ **Prediction Error:** {err}")

        elif batch_selected:

            predictor = load_leaf_predictor()
            results = []

            with st.spinner("🔬 Analyzing selected leaf images..."):

                for batch_file in batch_files:

                    file_bytes = batch_file.getvalue()
                    row = {
                        "Image": batch_file.name,
                        "Crop": "—",
                        "Prediction": "—",
                        "Confidence": "—",
                        "Status": "Not analyzed",
                        "Quality": "—",
                    }

                    try:
                        batch_image = Image.open(
                            io.BytesIO(file_bytes)
                        ).convert("RGB")

                        st.session_state.batch_image_bytes[batch_file.name] = file_bytes

                        quality_ok, quality_msg = check_image_quality(batch_image)

                        if not quality_ok:
                            row["Status"] = "Quality check failed"
                            row["Quality"] = quality_msg or "Image quality issue"
                            results.append(row)
                            continue

                        batch_file.seek(0)
                        prediction = predictor.predict(batch_file)

                        raw_class = prediction["predicted_class_name"]
                        confidence = float(prediction["confidence"])
                        confidence_pct = confidence * 100.0

                        formatted = (
                            raw_class
                            .replace("___", " - ")
                            .replace("_", " ")
                        )

                        crop = (
                            formatted.split(" - ", 1)[0]
                            if " - " in formatted
                            else "Unknown crop"
                        )

                        disease = (
                            formatted.split(" - ", 1)[-1]
                            if " - " in formatted
                            else formatted
                        )

                        probabilities = sorted(
                            [float(p) for p in prediction["all_probabilities"]],
                            reverse=True
                        )

                        top1 = probabilities[0] if probabilities else confidence
                        top2 = probabilities[1] if len(probabilities) > 1 else 0.0
                        margin = top1 - top2
                        uncertain = top1 < 0.50 or margin < 0.20
                        healthy = "healthy" in raw_class.lower()

                        row["Crop"] = crop
                        row["Prediction"] = disease
                        row["Confidence"] = f"{confidence_pct:.2f}%"
                        row["Quality"] = "Passed"

                        if uncertain:
                            row["Status"] = "Uncertain"
                        elif healthy:
                            row["Status"] = "No disease detected"
                        else:
                            row["Status"] = "Crop disease detected"

                    except Exception as exc:
                        row["Status"] = "Analysis error"
                        row["Quality"] = str(exc)

                    results.append(row)

            st.session_state.batch_results = results
            st.session_state.analysis_mode = "batch"
            st.session_state.analysis_stage = 2
            st.rerun()


# ============================================================
# 11. STEP 2 - AI DIAGNOSIS / BATCH RESULTS
#     Appears on the next screen after Step 1 analysis.
# ============================================================

if st.session_state.analysis_stage == 2:

    if st.button("← New Analysis", key="new_analysis_button"):
        st.session_state.analysis_stage = 1
        st.session_state.analysis_mode = None
        st.session_state.analysis_result = None
        st.session_state.analysis_image_bytes = None
        st.session_state.gradcam_image = None
        st.session_state.batch_results = []
        st.session_state.batch_image_bytes = {}
        st.session_state.batch_file_signature = None
        st.session_state.analysis_file_hash = None
        st.session_state.chat_messages = []
        st.session_state.chat_open = False
        st.rerun()

    if st.session_state.analysis_mode == "batch":

        st.divider()
        st.subheader("📊 Batch Analysis Results")
        st.progress(0.50, text="Step 2 of 4 — Batch analysis results")
        st.caption(
            "The results below were generated using the same image-quality checks, "
            "MobileNetV2 model, and confidence safeguard as the single-image workflow."
        )

        if st.session_state.batch_results:

            st.dataframe(
                st.session_state.batch_results,
                width="stretch",
                hide_index=True,
            )

            import csv
            csv_buffer = io.StringIO()
            fieldnames = ["Image", "Crop", "Prediction", "Confidence", "Status", "Quality"]
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(st.session_state.batch_results)

            st.download_button(
                "⬇️ Download Batch Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="leafguard_batch_results.csv",
                mime="text/csv",
                width="stretch",
                key="batch_csv_download",
            )

            try:
                batch_pdf = build_batch_pdf(
                    st.session_state.batch_results,
                    st.session_state.batch_image_bytes,
                )

                st.download_button(
                    "📄 Download Batch Analysis Report (PDF)",
                    data=batch_pdf,
                    file_name="leafguard_batch_analysis_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="batch_pdf_report_download",
                )
            except Exception as batch_pdf_error:
                st.error(f"❌ Could not create the batch PDF report: {batch_pdf_error}")

        else:
            st.info("No batch results are available yet.")

    elif st.session_state.analysis_mode == "single":

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
            # Clean diagnosis summary
            # --------------------------------------------------------

            st.markdown("### 🌿 Prediction Summary")

            prediction_col, confidence_col = st.columns(
                [1.55, 1],
                vertical_alignment="center"
            )

            with prediction_col:

                crop_name = formatted_name.split(" - ", 1)[0] if " - " in formatted_name else "Unknown crop"

                st.markdown(f"### 🌱 {crop_name}")
                st.caption("Detected crop")

                if is_unrecognized:

                    st.warning("⚠️ Unrecognized or uncertain image")

                    st.markdown(
                        f"## {formatted_name}"
                    )

                    st.caption(
                        "This is the nearest model match, but the model "
                        "does not have enough evidence to make a confident "
                        "supported-condition prediction."
                    )

                elif confidence_pct < 70.0:

                    st.warning("⚠️ Low-confidence prediction")

                    st.markdown(
                        f"## {formatted_name}"
                    )

                    st.caption(
                        "The prediction is an estimate and should be "
                        "re-checked with another clear, well-lit leaf image."
                    )

                elif is_healthy:

                    st.success("🌱 No disease detected")

                    disease_name = (
                        formatted_name.split(" - ", 1)[-1]
                        if " - " in formatted_name
                        else formatted_name
                    )

                    st.markdown(
                        f"## {disease_name}"
                    )

                    st.caption(
                        "The model identified the uploaded leaf as healthy."
                    )

                else:

                    st.error("⚠️ Crop disease detected")

                    disease_name = (
                        formatted_name.split(" - ", 1)[-1]
                        if " - " in formatted_name
                        else formatted_name
                    )

                    st.markdown(
                        f"## {disease_name}"
                    )

                    st.caption(
                        "The model identified a supported crop disease."
                    )

            with confidence_col:

                with st.container(border=True):

                    st.markdown("#### 🎯 Model Confidence")

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

                    st.caption(
                        f"Top-1 vs Top-2 margin: {margin:.2f}"
                    )


            # --------------------------------------------------------
            # Decision note
            # --------------------------------------------------------

            if is_unrecognized:

                st.warning(
                    "Please upload a clear close-up leaf image from a "
                    "supported crop: Apple, Potato, or Tomato."
                )

            elif confidence_pct < 70.0:

                st.info(
                    "For a more reliable result, try another clear, "
                    "well-lit close-up image of the same leaf."
                )

            else:

                st.caption(
                    "Supported scope: Apple, Potato, and Tomato leaf conditions."
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

            # --------------------------------------------------------
            # Make it explicit that Step 4 explains the Step 2 result
            # --------------------------------------------------------

            if not is_unrecognized:

                disease_display_name = (
                    formatted_name.split(" - ", 1)[-1]
                    if " - " in formatted_name
                    else formatted_name
                )

                st.markdown("### 🍃 Prediction Being Explained")

                explain_col1, explain_col2 = st.columns(2)

                with explain_col1:

                    with st.container(border=True):

                        st.caption("Crop")
                        st.markdown(
                            f"### 🌱 {formatted_name.split(' - ', 1)[0]}"
                        )

                with explain_col2:

                    with st.container(border=True):

                        st.caption("Crop Disease")
                        st.markdown(
                            f"### 🦠 {disease_display_name}"
                        )

                st.info(
                    f"ℹ️ This section explains the Step 2 prediction: "
                    f"**{disease_display_name}**. Grad-CAM does not make a "
                    "new prediction or change the diagnosis."
                )

            else:

                st.info(
                    "ℹ️ Grad-CAM is unavailable because this prediction "
                    "is currently classified as unrecognized/uncertain."
                )

            st.write(
                "Grad-CAM highlights the image regions that contributed "
                "more strongly to the prediction already shown in Step 2."
            )

            st.caption(
                "Red indicates stronger model influence, yellow indicates "
                "moderate influence, and blue indicates lower influence."
            )

            # --------------------------------------------------------
            # Generate Grad-CAM button
            # --------------------------------------------------------

            if not is_unrecognized:

                explain_btn = st.button(
                    "🔬 Generate Enhanced AI Explanation",
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

                            gradcam_data = (
                                generate_gradcam_explanation(
                                    predictor.model,
                                    gradcam_input,
                                    result[
                                        "predicted_class_index"
                                    ],
                                    analysis_image
                                )
                            )

                            st.session_state.gradcam_image = (
                                gradcam_data
                            )

                        except Exception as exc:

                            st.session_state.gradcam_image = None

                            st.warning(
                                "Grad-CAM visualization could not "
                                f"be generated: {exc}"
                            )

            # --------------------------------------------------------
            # Display Enhanced Grad-CAM
            # --------------------------------------------------------

            gradcam_data = st.session_state.gradcam_image

            if isinstance(gradcam_data, dict):

                st.markdown(
                    "### 🧠 Model Attention Visualization"
                )

                original_col, heatmap_col, overlay_col = st.columns(3)

                with original_col:

                    st.image(
                        gradcam_data["original"],
                        caption="Original Leaf",
                        width="stretch"
                    )

                with heatmap_col:

                    st.image(
                        gradcam_data["heatmap"],
                        caption="AI Attention Heatmap",
                        width="stretch"
                    )

                with overlay_col:

                    st.image(
                        gradcam_data["overlay"],
                        caption="Grad-CAM Overlay",
                        width="stretch"
                    )

                st.markdown(
                    "### 🧠 Where the Model Focused"
                )

                st.info(
                    gradcam_data["explanation"]
                )

                focus_col1, focus_col2 = st.columns(2)

                with focus_col1:

                    st.metric(
                        "Primary Focus Region",
                        gradcam_data["focus_region"]
                    )

                with focus_col2:

                    st.metric(
                        "Attention Coverage",
                        f"{gradcam_data['attention_coverage']:.1f}%"
                    )

                st.warning(
                    "⚠️ **Interpretation Note:** Grad-CAM shows which "
                    "image regions influenced the model's prediction. "
                    "It does not prove that a highlighted region contains "
                    "the disease or represent an exact disease boundary."
                )




# ============================================================
# 11B. PDF DIAGNOSIS REPORT
# ============================================================



# --------------------------------------------------------
# Single-image PDF report
# --------------------------------------------------------
if (
    st.session_state.analysis_stage == 2
    and st.session_state.analysis_mode == "single"
    and st.session_state.analysis_result is not None
    and st.session_state.analysis_image_bytes is not None
):
    st.divider()
    st.subheader("📄 Diagnosis Report")
    st.caption("Create a downloadable PDF from the existing LeafGuard single-image analysis.")

    try:
        diagnosis_pdf = build_diagnosis_pdf(
            st.session_state.analysis_result,
            st.session_state.analysis_image_bytes,
            st.session_state.gradcam_image,
        )

        st.download_button(
            "📄 Download Diagnosis Report",
            data=diagnosis_pdf,
            file_name="leafguard_diagnosis_report.pdf",
            mime="application/pdf",
            width="stretch",
            key="leafguard_pdf_report_download",
        )
    except Exception as pdf_error:
        st.error(f"❌ Could not create the PDF report: {pdf_error}")

        # ========================================================

# LEAFGUARD AI ASSISTANT (NO API REQUIRED)
    # Floating chat launcher with expandable assistant panel.
    # ========================================================

# The chat assistant is opened from a floating circular button
# fixed to the bottom-right corner of the browser window.
st.markdown(
    """
    <style>
    /* Floating LeafGuard chat launcher */
    .st-key-leafguard_chat_launcher {
        position: fixed !important;
        right: 24px !important;
        bottom: 24px !important;
        z-index: 100000 !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 70px !important;
        height: 70px !important;
    }

    .st-key-leafguard_chat_launcher button {
        width: 70px !important;
        height: 70px !important;
        min-height: 70px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: 40px !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.22) !important;
    }

    .st-key-leafguard_chat_launcher button p {
        font-size: 0 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }

    /* White outline chat-bubble icon; launcher color stays unchanged */
    .st-key-leafguard_chat_launcher button::before {
        content: "";
        width: 27px !important;
        height: 19px !important;
        border: 3px solid #ffffff !important;
        border-radius: 4px !important;
        box-sizing: border-box !important;
        display: block !important;
        position: absolute !important;
        left: 50% !important;
        top: 50% !important;
        transform: translate(-50%, -50%) !important;
    }

    .st-key-leafguard_chat_launcher button::after {
        content: "";
        position: absolute !important;
        width: 8px !important;
        height: 8px !important;
        border-left: 3px solid #ffffff !important;
        border-bottom: 3px solid #ffffff !important;
        left: calc(50% + 5px) !important;
        top: calc(50% + 6px) !important;
        transform: rotate(-12deg) !important;
        background: transparent !important;
    }

    /* Clear, visible close button inside the chat panel */
    .st-key-leafguard_chat_close button {
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 !important;
        border-radius: 10px !important;
        background: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        box-shadow: none !important;
    }

    .st-key-leafguard_chat_close button p {
        color: #ffffff !important;
        font-size: 20px !important;
        margin: 0 !important;
    }

    /* Floating chat panel */
    .st-key-leafguard_chat_panel {
        position: fixed !important;
        right: 24px !important;
        bottom: 98px !important;
        width: min(390px, calc(100vw - 48px)) !important;
        max-height: 72vh !important;
        overflow-y: auto !important;
        z-index: 99999 !important;
        background: #ffffff !important;
        color: #172033 !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        border-radius: 18px !important;
        padding: 14px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.22) !important;
    }

    .st-key-leafguard_chat_panel [data-testid="stChatMessageContent"],
    .st-key-leafguard_chat_panel [data-testid="stChatMessageContent"] p,
    .st-key-leafguard_chat_panel [data-testid="stChatMessageContent"] li,
    .st-key-leafguard_chat_panel label,
    .st-key-leafguard_chat_panel .stMarkdown,
    .st-key-leafguard_chat_panel .stCaption {
        color: #172033 !important;
    }

    .st-key-leafguard_chat_panel input {
        color: #172033 !important;
        background: #f5f7fa !important;
    }

    .st-key-leafguard_chat_panel input::placeholder {
        color: #697386 !important;
    }

    /* Make the chat send button clearly visible on the white panel */
    .st-key-leafguard_chat_panel .stFormSubmitButton button {
        width: 48px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 !important;
        border-radius: 10px !important;
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        box-shadow: none !important;
    }

    .st-key-leafguard_chat_panel .stFormSubmitButton button p {
        color: #ffffff !important;
        font-size: 20px !important;
        margin: 0 !important;
    }

    .leafguard-chat-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .leafguard-chat-context {
        font-size: 0.82rem;
        opacity: 0.75;
        margin-bottom: 8px;
    }

    .leafguard-chat-suggestions {
        font-size: 0.76rem;
        opacity: 0.68;
        margin-top: 3px;
    }

    @media (max-width: 600px) {
        .st-key-leafguard_chat_launcher {
            right: 16px !important;
            bottom: 16px !important;
            width: 64px !important;
            height: 64px !important;
        }

        .st-key-leafguard_chat_launcher button {
            width: 64px !important;
            height: 64px !important;
            min-height: 64px !important;
            font-size: 36px !important;
        }

        .st-key-leafguard_chat_launcher button p {
            font-size: 36px !important;
        }

        .st-key-leafguard_chat_panel {
            right: 16px !important;
            bottom: 88px !important;
            width: calc(100vw - 32px) !important;
            max-height: 68vh !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# Floating launcher
# --------------------------------------------------------

launcher_label = "Chat" if not st.session_state.chat_open else "✕"

if st.button(
    launcher_label,
    key="leafguard_chat_launcher",
    help="Open LeafGuard AI Assistant" if not st.session_state.chat_open else "Close LeafGuard AI Assistant",
):
    st.session_state.chat_open = not st.session_state.chat_open

# --------------------------------------------------------
# Expandable chat panel
# --------------------------------------------------------

if st.session_state.chat_open:

    with st.container(key="leafguard_chat_panel"):

        header_col1, header_col2 = st.columns([5, 1])

        with header_col1:
            st.markdown(
                '<div class="leafguard-chat-title">💬 LeafGuard AI Assistant</div>',
                unsafe_allow_html=True,
            )

        with header_col2:
            if st.button(
                "✕",
                key="leafguard_chat_close",
                help="Close chat",
            ):
                st.session_state.chat_open = False
                st.rerun()

        st.markdown(
            '<div class="leafguard-chat-context">Ask about your current prediction, symptoms, confidence, recommended action, or Grad-CAM.</div>',
            unsafe_allow_html=True,
        )

        # The assistant is available before upload and becomes result-aware after analysis.
        if st.session_state.analysis_result is None:
            st.markdown(
                '<div class="leafguard-chat-context">No leaf has been analyzed yet. Ask about LeafGuard, supported crops, image quality, or Grad-CAM.</div>',
                unsafe_allow_html=True,
            )
        else:
            chat_result = st.session_state.analysis_result
            chat_class = chat_result["predicted_class_name"]
            chat_formatted = chat_class.replace("___", " - ").replace("_", " ")
            chat_crop = chat_formatted.split(" - ", 1)[0] if " - " in chat_formatted else "Unknown crop"
            chat_disease = chat_formatted.split(" - ", 1)[-1] if " - " in chat_formatted else chat_formatted
            chat_confidence = float(chat_result["confidence"]) * 100.0

            with st.container(border=True):
                st.caption("Current analysis")
                context_col1, context_col2, context_col3 = st.columns(3)
                with context_col1:
                    st.markdown(f"**🌱 {chat_crop}**")
                with context_col2:
                    st.markdown(f"**🦠 {chat_disease}**")
                with context_col3:
                    st.markdown(f"**🎯 {chat_confidence:.2f}%**")

        # Initial greeting.
        if len(st.session_state.chat_messages) == 0:
            greeting = "Hello! I’m the LeafGuard AI Assistant."
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": greeting}
            )

        # Display history.
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Reuse the existing local assistant function.
        def local_leafguard_response(question):
            """
            Local, rule-based LeafGuard assistant.
            Uses the current prediction and DISEASE_INFO knowledge base.
            No external API or API key is required.
            """

            q = question.strip().lower()

            if not q:
                return "Please type a question about your LeafGuard result."

            if any(term in q for term in [
                "supported crop",
                "which crops",
                "what crops",
                "supported plant",
                "supported plants"
            ]):
                return (
                    "LeafGuard AI currently supports Apple, Potato, and Tomato "
                    "leaf conditions across 17 trained classes."
                )

            if any(term in q for term in [
                "grad-cam",
                "grad cam",
                "heatmap",
                "attention map",
                "focus"
            ]):
                if isinstance(st.session_state.gradcam_image, dict):
                    focus = st.session_state.gradcam_image.get(
                        "focus_region",
                        "the highlighted region"
                    )
                    coverage = st.session_state.gradcam_image.get(
                        "attention_coverage",
                        0.0
                    )
                    explanation = st.session_state.gradcam_image.get(
                        "explanation",
                        "The highlighted regions contributed more strongly to the prediction."
                    )
                    return (
                        f"Grad-CAM explains the existing Step 2 prediction. "
                        f"The primary focus region is the {focus}, with about "
                        f"{coverage:.1f}% attention coverage. {explanation} "
                        "The heatmap shows model influence; it is not an exact disease boundary."
                    )

                return (
                    "Grad-CAM shows which image regions contributed more strongly "
                    "to the model's existing prediction. Generate the attention map "
                    "in Step 4 to see the visual explanation."
                )

            if any(term in q for term in [
                "image quality",
                "photo quality",
                "blurry",
                "dark image",
                "lighting"
            ]):
                return (
                    "LeafGuard checks image resolution, darkness, overexposure, "
                    "and image detail before running inference. A clear, focused, "
                    "well-lit close-up leaf photo gives the model a better input."
                )

            if st.session_state.analysis_result is None:
                return (
                    "Please upload and analyze a leaf first. After a result is available, "
                    "I can explain the crop, prediction, confidence, symptoms, recommended "
                    "general action, and Grad-CAM."
                )

            current_result = st.session_state.analysis_result
            current_class = current_result["predicted_class_name"]
            current_confidence = float(current_result["confidence"])
            current_confidence_pct = current_confidence * 100.0
            current_formatted = (
                current_class.replace("___", " - ").replace("_", " ")
            )
            current_crop = (
                current_formatted.split(" - ", 1)[0]
                if " - " in current_formatted
                else "Unknown crop"
            )
            current_disease = (
                current_formatted.split(" - ", 1)[-1]
                if " - " in current_formatted
                else current_formatted
            )
            current_healthy = "healthy" in current_class.lower()

            current_info = DISEASE_INFO.get(
                current_class,
                {
                    "description": "No detailed description available.",
                    "symptoms": "N/A",
                    "recommendation": "Consult a local agricultural extension specialist for guidance."
                }
            )

            if any(term in q for term in [
                "what is the crop",
                "which crop",
                "crop name",
                "plant name"
            ]):
                return f"The detected crop is {current_crop}."

            if any(term in q for term in [
                "what disease",
                "which disease",
                "disease name",
                "diagnosis",
                "what did you detect",
                "what is wrong"
            ]):
                if current_healthy:
                    return f"LeafGuard detected {current_crop} as healthy. No disease was detected."
                return f"LeafGuard identified {current_disease} on the {current_crop} leaf."

            if any(term in q for term in [
                "confidence",
                "certainty",
                "sure",
                "probability",
                "how accurate"
            ]):
                sorted_probs = sorted(
                    [float(p) for p in current_result["all_probabilities"]],
                    reverse=True
                )
                top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                current_margin = sorted_probs[0] - top2
                if current_confidence_pct >= 70.0:
                    level = "relatively high"
                elif current_confidence_pct >= 50.0:
                    level = "lower"
                else:
                    level = "low"
                return (
                    f"The model confidence is {current_confidence_pct:.2f}%, which is {level} "
                    f"for this prediction. The Top-1 vs Top-2 probability margin is "
                    f"{current_margin:.2f}. Confidence is a model score, not a guarantee of diagnosis."
                )

            if any(term in q for term in [
                "symptom",
                "signs",
                "look like",
                "appearance"
            ]):
                return f"Common symptoms associated with {current_disease}: {current_info['symptoms']}"

            if any(term in q for term in [
                "what should i do",
                "what do i do",
                "recommend",
                "recommendation",
                "treatment",
                "manage",
                "management",
                "next step",
                "how can i help"
            ]):
                return f"General action for {current_disease}: {current_info['recommendation']}"

            if any(term in q for term in [
                "what is",
                "explain",
                "meaning",
                "tell me about",
                "why"
            ]):
                return f"{current_disease}: {current_info['description']}"

            return (
                "I can help with your current LeafGuard result. Try asking: "
                "'What disease was detected?', 'What are the symptoms?', "
                "'What should I do?', 'Why is the confidence high?', or "
                "'What did Grad-CAM show?'"
            )

        st.markdown(
            '<div class="leafguard-chat-suggestions">Try: What crops are supported? • How does LeafGuard work? • What does Grad-CAM mean?</div>',
            unsafe_allow_html=True,
        )

        with st.form("leafguard_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([6, 1])
            with input_col:
                user_question = st.text_input(
                    "Ask LeafGuard AI",
                    key="leafguard_chat_input",
                    label_visibility="collapsed",
                    placeholder="Ask LeafGuard AI...",
                )
            with send_col:
                submit_chat = st.form_submit_button("➤", use_container_width=True)

        if submit_chat and user_question:
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_question}
            )

            assistant_reply = local_leafguard_response(user_question)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": assistant_reply}
            )

            st.rerun()


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 LeafGuard AI v1 • MobileNetV2 • "
    "17 Apple, Potato & Tomato conditions • "
    "Confidence-aware AI • Grad-CAM Explainability"
)