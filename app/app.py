import sys
from pathlib import Path
import hashlib
import io
import json
from urllib.parse import urlencode
import requests
from requests import RequestException

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

WEATHER_DESCRIPTION_HI = {
    0: "साफ आसमान", 1: "मुख्यतः साफ", 2: "आंशिक बादल", 3: "घने बादल",
    45: "कोहरा", 48: "पाला वाला कोहरा", 51: "हल्की फुहार", 53: "मध्यम फुहार", 55: "घनी फुहार",
    56: "हल्की जमने वाली फुहार", 57: "घनी जमने वाली फुहार", 61: "हल्की बारिश", 63: "मध्यम बारिश", 65: "तेज़ बारिश",
    66: "हल्की जमने वाली बारिश", 67: "तेज़ जमने वाली बारिश", 71: "हल्की बर्फबारी", 73: "मध्यम बर्फबारी", 75: "तेज़ बर्फबारी",
    77: "बर्फ के दाने", 80: "हल्की बारिश की बौछारें", 81: "मध्यम बारिश की बौछारें", 82: "तेज़ बारिश की बौछारें",
    85: "हल्की बर्फ की बौछारें", 86: "तेज़ बर्फ की बौछारें", 95: "गरज के साथ तूफ़ान", 96: "हल्के ओलों के साथ तूफ़ान", 99: "तेज़ ओलों के साथ तूफ़ान",
}

def localized_weather_description(code):
    if st.session_state.get("ui_language") == "hi":
        try:
            return WEATHER_DESCRIPTION_HI.get(int(code), "अज्ञात स्थिति")
        except (TypeError, ValueError):
            return "अज्ञात स्थिति"
    return weather_description(code)


def weather_description(code):
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    try:
        return mapping.get(int(code), "Unknown conditions")
    except (TypeError, ValueError):
        return "Unknown conditions"


def _get_json_with_retry(url, timeout=20, retries=2):
    """Fetch JSON with retries for transient network failures."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "LeafGuard-AI/1.0"},
            )
            response.raise_for_status()
            return response.json()
        except RequestException as exc:
            last_error = exc
            if attempt < retries:
                continue

    raise RuntimeError(
        "Weather service could not be reached from this computer/network. "
        "Check your internet connection, VPN/proxy/firewall settings, then try again."
    ) from last_error


def _choose_best_place(results, requested_name):
    """Choose the most useful city result instead of blindly taking result zero."""
    requested_lower = requested_name.strip().lower()
    candidates = [r for r in results if r.get("latitude") is not None and r.get("longitude") is not None]
    if not candidates:
        raise ValueError("Location not found. Try a city or town name, optionally followed by a country.")

    # Prefer an exact city-name match; otherwise prefer populated city/admin entries.
    exact = [
        r for r in candidates
        if str(r.get("name", "")).strip().lower() == requested_lower
    ]
    if exact:
        candidates = exact

    def score(place):
        feature = str(place.get("feature_code", ""))
        feature_score = 2 if feature.startswith("PPLA") or feature == "PPLC" else 0
        population = float(place.get("population") or 0)
        return (feature_score, population)

    return max(candidates, key=score)


def _next_hour_probability(hourly_times, probabilities, current_time):
    """Return precipitation probability for the first forecast hour after current time."""
    if not hourly_times or not probabilities:
        return None

    try:
        current_dt = datetime.fromisoformat(current_time)
    except (TypeError, ValueError):
        return probabilities[1] if len(probabilities) > 1 else probabilities[0]

    for idx, time_value in enumerate(hourly_times):
        try:
            hourly_dt = datetime.fromisoformat(time_value)
        except (TypeError, ValueError):
            continue
        if hourly_dt > current_dt and idx < len(probabilities):
            return probabilities[idx]

    return probabilities[-1] if probabilities else None


def fetch_weather_for_location(location_query):
    geocode_params = urlencode({
        "name": location_query,
        "count": 10,
        "language": "en",
        "format": "json",
    })
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?{geocode_params}"

    geo = _get_json_with_retry(geocode_url)
    results = geo.get("results") or []
    place = _choose_best_place(results, location_query)

    latitude = place["latitude"]
    longitude = place["longitude"]

    weather_params = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "wind_gusts_10m",
            "cloud_cover",
            "evapotranspiration",
            "vapour_pressure_deficit",
            "dew_point_2m",
            "is_day",
        ]),
        "hourly": "precipitation_probability",
        "forecast_hours": 4,
        "timezone": "auto",
    })
    weather_url = f"https://api.open-meteo.com/v1/forecast?{weather_params}"
    weather = _get_json_with_retry(weather_url)

    current = weather.get("current", {})
    hourly = weather.get("hourly", {})
    probabilities = hourly.get("precipitation_probability") or []
    hourly_times = hourly.get("time") or []
    current_time = current.get("time", "")
    next_prob = _next_hour_probability(hourly_times, probabilities, current_time)

    label_parts = [place.get("name", location_query)]
    if place.get("admin1"):
        label_parts.append(place["admin1"])
    if place.get("country"):
        label_parts.append(place["country"])

    return {
        "location": ", ".join(label_parts),
        "latitude": latitude,
        "longitude": longitude,
        "timezone": weather.get("timezone", place.get("timezone", "")),
        "time": current_time,
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "apparent_temperature": current.get("apparent_temperature"),
        "precipitation": current.get("precipitation"),
        "weather_code": current.get("weather_code"),
        "weather_description": weather_description(current.get("weather_code")),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_gusts": current.get("wind_gusts_10m"),
        "cloud_cover": current.get("cloud_cover"),
        "evapotranspiration": current.get("evapotranspiration"),
        "vpd": current.get("vapour_pressure_deficit"),
        "dew_point": current.get("dew_point_2m"),
        "is_day": current.get("is_day"),
        "precipitation_probability": next_prob,
    }


# ============================================================
# MULTI-LANGUAGE SUPPORT
# English + Hindi. Language selector is inside the custom top-right ⋮ menu.
# ============================================================
UI_TRANSLATIONS = {
    "en": {
        "app_name": "LeafGuard AI",
        "smart_crop": "Smart Crop Health & Disease Detection",
        "supported_crops": "### 🌾 Supported Crops",
        "supported_crops_body": """🍎 **Apple**\n\nScab • Black Rot • Cedar Apple Rust • Healthy\n\n🥔 **Potato**\n\nEarly Blight • Late Blight • Healthy\n\n🍅 **Tomato**\n\nBacterial Spot • Early Blight • Late Blight  \nLeaf Mold • Septoria • Spider Mites  \nTarget Spot • Mosaic Virus • TYLCV • Healthy""",
        "how_it_works": "### 🧭 How It Works",
        "how_it_works_body": "**01** Upload a clear leaf photo\n\n**02** Run AI analysis\n\n**03** Review the result\n\n**04** Explain the prediction",
        "features": "### ✨ Features",
        "features_body": "• Single Photo Analysis  \n• Batch Analysis  \n• Image Quality Check  \n• Confidence-Aware Prediction  \n• Disease Information  \n• Grad-CAM Explainability  \n• AI Assistant  \n• PDF Diagnosis Reports  \n• Scan History  \n• Weather & Environment",
        "safety": "### 🛡️ Safety",
        "safety_caption": "LeafGuard checks image quality before inference and flags uncertain predictions.",
        "weather": "🌦️ Weather",
        "weather_heading": "### 🌦️ Weather & Environment",
        "weather_caption": "Open the full weather report for a city or town. Weather values provide environmental context and are not a disease diagnosis.",
        "location": "Location",
        "get_weather": "🌤️ Get Weather",
        "enter_city": "Please enter a city or town.",
        "fetching_weather": "Fetching weather...",
        "weather_service_error": "Weather service could not be reached from this computer/network. Check your internet connection, VPN/proxy/firewall settings, then try again.",
        "weather_lookup_failed": "Weather lookup failed:",
        "unknown_location": "Unknown location",
        "unknown_conditions": "Unknown conditions",
        "current_condition": "**Current condition:**",
        "temperature": "Temperature",
        "feels_like": "Feels Like",
        "humidity": "Humidity",
        "rain_next_hour": "Rain Next Hour",
        "detailed_conditions": "#### Detailed Conditions",
        "precipitation": "🌧️ Precipitation",
        "wind": "💨 Wind",
        "wind_gusts": "💨 Wind Gusts",
        "cloud_cover": "☁️ Cloud Cover",
        "dew_point": "💧 Dew Point",
        "evapotranspiration": "🌿 Evapotranspiration",
        "vpd": "📈 Vapour Pressure Deficit",
        "local_weather_time": "Local weather time",
        "timezone": "Timezone",
        "weather_info": "Enter a location and choose Get Weather to open the full report.",
        "step1": "📷 Step 1 — Upload & Prepare Your Leaf",
        "step1_progress": "Step 1 of 4 — Upload your leaf image",
        "upload_info": "Choose a single leaf photo or analyze multiple leaf photos at once. Both options use the same LeafGuard quality checks and MobileNetV2 model.",
        "single_photo": "### 📷 Single Photo",
        "single_caption": "Analyze one leaf image",
        "single_uploader": "Choose one crop leaf image",
        "uploaded_leaf": "Uploaded Leaf",
        "resolution": "Resolution",
        "quality_passed": "✅ Quality passed",
        "quality_issue": "⚠️ Quality issue",
        "single_empty": "Upload one image to analyze a single leaf.",
        "batch": "### 📂 Batch Analysis",
        "batch_caption": "Analyze multiple leaf images in one run",
        "batch_uploader": "Choose multiple crop leaf images",
        "selected_images": "{} image(s) selected.",
        "batch_empty": "Select multiple images for batch analysis.",
        "analyze_leaf": "🧪 Analyze Leaf",
        "both_selected": "Please use either Single Photo or Batch Analysis, not both at the same time.",
        "select_one": "Please select only one analysis option: Single Photo or Batch Analysis.",
        "analysis_stopped": "Analysis stopped because the image did not pass the quality checks.",
        "running_analysis": "🔬 Running LeafGuard AI analysis...",
        "analyzing_batch": "🔬 Analyzing selected leaf images...",
        "prediction_error": "❌ **Prediction Error:** {}",
        "upload_first": "⚠️ Please upload a leaf image first.",
        "invalid_image": "❌ **Invalid Image:** {}",
        "step2": "🔬 Step 2 — AI Diagnosis",
        "step2_progress": "Step 2 of 4 — Reviewing AI diagnosis",
        "batch_results": "📊 Batch Analysis Results",
        "batch_progress": "Step 2 of 4 — Batch analysis results",
        "batch_results_caption": "The results below were generated using the same image-quality checks, MobileNetV2 model, and confidence safeguard as the single-image workflow.",
        "download_csv": "⬇️ Download Batch Results (CSV)",
        "download_batch_pdf": "📄 Download Batch Analysis Report (PDF)",
        "no_batch_results": "No batch results are available yet.",
        "history_date": "Date & Time", "history_type": "Type", "history_image": "Image", "history_prediction": "Prediction", "history_confidence": "Confidence", "history_status": "Status",
        "back": "← Back",
        "prediction_summary": "### 🌿 Prediction Summary",
        "detected_crop": "Detected crop",
        "model_confidence": "#### 🎯 Model Confidence",
        "model_certainty": "Model Certainty",
        "unrecognized": "⚠️ Unrecognized or uncertain image",
        "unrecognized_caption": "This is the closest model match, but there is not enough evidence for a reliable decision about a supported condition.",
        "low_confidence": "⚠️ Low-confidence prediction",
        "low_confidence_caption": "This prediction is an estimate. Re-check with another clear, well-lit close-up image of the same leaf.",
        "no_disease": "🌱 No disease detected",
        "no_disease_caption": "The model identified the uploaded leaf as healthy.",
        "disease_detected": "⚠️ Crop disease detected",
        "disease_detected_caption": "The model identified a supported crop disease.",
        "confidence_margin": "Top-1 vs Top-2 margin: {:.2f}",
        "decision_uncertain": "Please upload a clear close-up leaf image from a supported crop: Apple, Potato, or Tomato.",
        "decision_low": "For a more reliable result, try another clear, well-lit close-up image of the same leaf.",
        "supported_scope": "Supported scope: Apple, Potato, and Tomato leaf conditions.",
        "step3": "📖 Step 3 — Understand the Result",
        "step3_progress": "Step 3 of 4 — Condition information",
        "no_disease_info": "No detailed description available.",
        "na": "N/A",
        "consult_extension": "Consult a local agricultural extension specialist for guidance.",
        "condition_overview": "### 📖 Condition Overview",
        "common_symptoms": "### 🔍 Common Symptoms",
        "recommended_action": "### 🛡️ Recommended General Action",
        "step3_uncertain": "Detailed disease information is not shown because the image was not confidently matched to a supported condition.",
        "step4": "🔬 Step 4 — Explain This Prediction",
        "step4_progress": "Step 4 of 4 — Explainable AI",
        "prediction_explained": "### 🍃 Prediction Being Explained",
        "crop": "Crop",
        "crop_disease": "Crop Disease",
        "gradcam_unavailable": "ℹ️ Grad-CAM is unavailable because this prediction is currently classified as unrecognized/uncertain.",
        "gradcam_info": "ℹ️ This section explains the Step 2 prediction: **{}**. Grad-CAM does not make a new prediction or change the diagnosis.",
        "gradcam_text": "Grad-CAM highlights the image regions that contributed more strongly to the prediction already shown in Step 2.",
        "gradcam_color_caption": "Red indicates stronger model influence, yellow indicates moderate influence, and blue indicates lower influence.",
        "generate_gradcam": "🔬 Generate Advanced AI Explanation",
        "generating_gradcam": "🧠 Generating AI attention map...",
        "gradcam_title": "### 🧠 Model Attention Visualization",
        "original_leaf": "Original Leaf",
        "ai_heatmap": "AI Attention Heatmap",
        "gradcam_overlay": "Grad-CAM Overlay",
        "where_focused": "### 🧠 Where the Model Focused",
        "primary_focus": "Primary Focus Region",
        "attention_coverage": "Attention Coverage",
        "interpretation_note": "⚠️ **Interpretation Note:** Grad-CAM shows which image regions influenced the model's prediction. It does not prove that a highlighted region contains the disease or represent an exact disease boundary.",
        "pdf_title": "📄 Diagnosis Report",
        "pdf_caption": "Create a downloadable PDF from the existing LeafGuard single-image analysis.",
        "download_pdf": "📄 Download Diagnosis Report",
        "history_title": "🕘 Scan History",
        "history_caption": "Recent analyses from this browser session. History is cleared when the session ends.",
        "clear_history": "🗑️ Clear History",
        "no_scans": "No scans yet. Complete a single-photo or batch analysis to build your history.",
        "chat_title": "💬 LeafGuard AI Assistant",
        "chat_context": "Ask about your current prediction, symptoms, confidence, recommended action, or Grad-CAM.",
        "chat_no_leaf": "No leaf has been analyzed yet. Ask about LeafGuard, supported crops, image quality, or Grad-CAM.",
        "chat_greeting": "Hello! I’m the LeafGuard AI Assistant.",
        "chat_placeholder": "Ask LeafGuard AI...",
        "clear_chat": "Clear chat",
        "try_questions": "Try: What crops are supported? • How does LeafGuard work? • What does Grad-CAM mean?",
        "current_analysis": "Current analysis",
        "language": "🌐 Language",
        "english": "English",
        "hindi": "हिन्दी",
        "features_note": "LeafGuard AI v1",
        "footer": "🌱 LeafGuard AI v1 • MobileNetV2 • 17 Apple, Potato & Tomato conditions • Confidence-aware AI • Grad-CAM Explainability",
        "model_classes": "🤖 MobileNetV2 • 17 Classes",
        "app_summary": "AI-powered leaf health analysis for Apple, Potato, and Tomato, with confidence checks and visual explanations.",
        "glance": "📊 LeafGuard AI at a Glance",
        "supported_classes": "Supported Classes",
        "test_accuracy": "Test Accuracy",
        "held_out": "Held-out benchmark",
        "explainability": "Explainability",
        "visual_attention": "Visual model attention",
        "apple_potato_tomato": "Apple • Potato • Tomato",
    },
    "hi": {
        "app_name": "लीफगार्ड AI",
        "smart_crop": "स्मार्ट फसल स्वास्थ्य और रोग पहचान",
        "supported_crops": "### 🌾 समर्थित फसलें",
        "supported_crops_body": """🍎 **सेब**\n\nस्कैब • ब्लैक रॉट • सीडर एप्पल रस्ट • स्वस्थ\n\n🥔 **आलू**\n\nअर्ली ब्लाइट • लेट ब्लाइट • स्वस्थ\n\n🍅 **टमाटर**\n\nबैक्टीरियल स्पॉट • अर्ली ब्लाइट • लेट ब्लाइट  \nलीफ मोल्ड • सेप्टोरिया • स्पाइडर माइट्स  \nटारगेट स्पॉट • मोज़ेक वायरस • TYLCV • स्वस्थ""",
        "how_it_works": "### 🧭 यह कैसे काम करता है",
        "how_it_works_body": "**01** साफ पत्ती की फोटो अपलोड करें\n\n**02** AI विश्लेषण चलाएँ\n\n**03** परिणाम देखें\n\n**04** पूर्वानुमान की व्याख्या देखें",
        "features": "### ✨ सुविधाएँ",
        "features_body": "• एकल फोटो विश्लेषण  \n• बैच विश्लेषण  \n• छवि गुणवत्ता जांच  \n• भरोसा-आधारित पूर्वानुमान  \n• रोग जानकारी  \n• Grad-CAM व्याख्या  \n• AI सहायक  \n• PDF निदान रिपोर्ट  \n• स्कैन इतिहास  \n• मौसम और वातावरण",
        "safety": "### 🛡️ सुरक्षा",
        "safety_caption": "LeafGuard विश्लेषण से पहले छवि की गुणवत्ता जांचता है और अनिश्चित परिणामों को चिन्हित करता है।",
        "weather": "🌦️ मौसम",
        "weather_heading": "### 🌦️ मौसम और वातावरण",
        "weather_caption": "किसी शहर या कस्बे की पूरी मौसम रिपोर्ट खोलें। मौसम के मान केवल पर्यावरणीय संदर्भ देते हैं, रोग का निदान नहीं।",
        "location": "स्थान",
        "get_weather": "🌤️ मौसम प्राप्त करें",
        "enter_city": "कृपया किसी शहर या कस्बे का नाम दर्ज करें।",
        "fetching_weather": "मौसम प्राप्त किया जा रहा है...",
        "weather_service_error": "इस कंप्यूटर/नेटवर्क से मौसम सेवा तक पहुँचना संभव नहीं हुआ। इंटरनेट कनेक्शन, VPN/प्रॉक्सी/फ़ायरवॉल सेटिंग जांचें और फिर प्रयास करें।",
        "weather_lookup_failed": "मौसम खोज विफल:",
        "unknown_location": "अज्ञात स्थान",
        "unknown_conditions": "अज्ञात स्थिति",
        "current_condition": "**वर्तमान स्थिति:**",
        "temperature": "तापमान",
        "feels_like": "महसूस होने वाला तापमान",
        "humidity": "नमी",
        "rain_next_hour": "अगले घंटे की बारिश",
        "detailed_conditions": "#### विस्तृत स्थितियाँ",
        "precipitation": "🌧️ वर्षा",
        "wind": "💨 हवा",
        "wind_gusts": "💨 हवा के झोंके",
        "cloud_cover": "☁️ बादल",
        "dew_point": "💧 ओसांक",
        "evapotranspiration": "🌿 वाष्पोत्सर्जन",
        "vpd": "📈 वाष्प दाब घाटा",
        "local_weather_time": "स्थानीय मौसम समय",
        "timezone": "समय क्षेत्र",
        "weather_info": "स्थान दर्ज करें और पूरी रिपोर्ट खोलने के लिए **मौसम प्राप्त करें** चुनें।",
        "step1": "📷 चरण 1 — अपनी पत्ती अपलोड और तैयार करें",
        "step1_progress": "चरण 1 / 4 — पत्ती की छवि अपलोड करें",
        "upload_info": "एकल पत्ती की फोटो चुनें या एक बार में कई पत्तियों का विश्लेषण करें। दोनों विकल्प समान LeafGuard गुणवत्ता जांच और MobileNetV2 मॉडल का उपयोग करते हैं।",
        "single_photo": "### 📷 एकल फोटो",
        "single_caption": "एक पत्ती की छवि का विश्लेषण करें",
        "single_uploader": "एक फसल पत्ती की छवि चुनें",
        "uploaded_leaf": "अपलोड की गई पत्ती",
        "resolution": "रिज़ॉल्यूशन",
        "quality_passed": "✅ गुणवत्ता पास",
        "quality_issue": "⚠️ गुणवत्ता समस्या",
        "single_empty": "एकल पत्ती का विश्लेषण करने के लिए एक छवि अपलोड करें।",
        "batch": "### 📂 बैच विश्लेषण",
        "batch_caption": "एक बार में कई पत्तियों की छवियों का विश्लेषण करें",
        "batch_uploader": "कई फसल पत्ती छवियाँ चुनें",
        "selected_images": "{} छवि(याँ) बैच विश्लेषण के लिए चुनी गईं।",
        "batch_empty": "बैच विश्लेषण के लिए कई छवियाँ चुनें।",
        "analyze_leaf": "🧪 पत्ती का विश्लेषण करें",
        "both_selected": "कृपया एक समय में केवल एक विकल्प उपयोग करें: एकल फोटो या बैच विश्लेषण।",
        "select_one": "कृपया केवल एक विश्लेषण विकल्प चुनें: एकल फोटो या बैच विश्लेषण।",
        "analysis_stopped": "छवि गुणवत्ता जांच पास न होने के कारण विश्लेषण रोक दिया गया।",
        "running_analysis": "🔬 LeafGuard AI विश्लेषण चलाया जा रहा है...",
        "analyzing_batch": "🔬 चुनी गई पत्ती छवियों का विश्लेषण किया जा रहा है...",
        "prediction_error": "❌ **पूर्वानुमान त्रुटि:** {}",
        "upload_first": "⚠️ कृपया पहले पत्ती की छवि अपलोड करें।",
        "invalid_image": "❌ **अमान्य छवि:** {}",
        "step2": "🔬 चरण 2 — AI निदान",
        "step2_progress": "चरण 2 / 4 — AI निदान की समीक्षा",
        "batch_results": "📊 बैच विश्लेषण परिणाम",
        "batch_progress": "चरण 2 / 4 — बैच विश्लेषण परिणाम",
        "batch_results_caption": "नीचे दिए गए परिणाम एकल-छवि वर्कफ़्लो की तरह ही छवि-गुणवत्ता जांच, MobileNetV2 मॉडल और भरोसा सुरक्षा जांच का उपयोग करके बनाए गए हैं।",
        "download_csv": "⬇️ बैच परिणाम डाउनलोड करें (CSV)",
        "download_batch_pdf": "📄 बैच विश्लेषण रिपोर्ट डाउनलोड करें (PDF)",
        "no_batch_results": "अभी कोई बैच परिणाम उपलब्ध नहीं हैं।",
        "history_date": "दिनांक और समय", "history_type": "प्रकार", "history_image": "छवि", "history_prediction": "पूर्वानुमान", "history_confidence": "भरोसा", "history_status": "स्थिति",
        "back": "← वापस",
        "prediction_summary": "### 🌿 पूर्वानुमान सारांश",
        "detected_crop": "पहचानी गई फसल",
        "model_confidence": "#### 🎯 मॉडल का भरोसा",
        "model_certainty": "मॉडल निश्चितता",
        "unrecognized": "⚠️ छवि अनिश्चित या पहचान से बाहर है",
        "unrecognized_caption": "यह निकटतम मॉडल मिलान है, लेकिन समर्थित स्थिति के बारे में भरोसेमंद निर्णय के लिए पर्याप्त प्रमाण नहीं हैं।",
        "low_confidence": "⚠️ कम-भरोसे वाला पूर्वानुमान",
        "low_confidence_caption": "यह पूर्वानुमान एक अनुमान है। उसी पत्ती की एक और साफ, अच्छी रोशनी वाली नज़दीकी तस्वीर से दोबारा जांचें।",
        "no_disease": "🌱 कोई रोग नहीं मिला",
        "no_disease_caption": "मॉडल ने अपलोड की गई पत्ती को स्वस्थ पहचाना।",
        "disease_detected": "⚠️ फसल रोग पाया गया",
        "disease_detected_caption": "मॉडल ने एक समर्थित फसल रोग की पहचान की।",
        "confidence_margin": "टॉप-1 बनाम टॉप-2 अंतर: {:.2f}",
        "decision_uncertain": "कृपया समर्थित फसल की साफ़, नज़दीकी पत्ती की छवि अपलोड करें: सेब, आलू या टमाटर।",
        "decision_low": "अधिक भरोसेमंद परिणाम के लिए उसी पत्ती की एक और साफ़, अच्छी रोशनी वाली नज़दीकी तस्वीर आज़माएँ।",
        "supported_scope": "समर्थित दायरा: सेब, आलू और टमाटर की पत्ती की स्थितियाँ।",
        "step3": "📖 चरण 3 — परिणाम समझें",
        "step3_progress": "चरण 3 / 4 — स्थिति की जानकारी",
        "no_disease_info": "विस्तृत विवरण उपलब्ध नहीं है।",
        "na": "लागू नहीं",
        "consult_extension": "मार्गदर्शन के लिए स्थानीय कृषि विस्तार विशेषज्ञ से संपर्क करें।",
        "condition_overview": "### 📖 स्थिति का अवलोकन",
        "common_symptoms": "### 🔍 सामान्य लक्षण",
        "recommended_action": "### 🛡️ सुझाई गई सामान्य कार्रवाई",
        "step3_uncertain": "विस्तृत रोग जानकारी नहीं दिखाई गई क्योंकि छवि का समर्थित स्थिति से भरोसेमंद मिलान नहीं हुआ।",
        "step4": "🔬 चरण 4 — इस पूर्वानुमान को समझें",
        "step4_progress": "चरण 4 / 4 — Explainable AI",
        "prediction_explained": "### 🍃 समझाया जा रहा पूर्वानुमान",
        "crop": "फसल",
        "crop_disease": "फसल रोग",
        "gradcam_unavailable": "ℹ️ यह पूर्वानुमान अनिश्चित/पहचान से बाहर होने के कारण Grad-CAM उपलब्ध नहीं है।",
        "gradcam_info": "ℹ️ यह अनुभाग चरण 2 के पूर्वानुमान **{}** की व्याख्या करता है। Grad-CAM नया पूर्वानुमान नहीं करता और निदान नहीं बदलता।",
        "gradcam_text": "Grad-CAM उन छवि क्षेत्रों को उजागर करता है जिन्होंने चरण 2 में दिए गए पूर्वानुमान में अधिक योगदान दिया।",
        "gradcam_color_caption": "लाल रंग मॉडल के अधिक प्रभाव, पीला मध्यम प्रभाव और नीला कम प्रभाव को दर्शाता है।",
        "generate_gradcam": "🔬 उन्नत AI व्याख्या बनाएं",
        "generating_gradcam": "🧠 AI ध्यान मानचित्र बनाया जा रहा है...",
        "gradcam_title": "### 🧠 मॉडल का ध्यान दृश्य",
        "original_leaf": "मूल पत्ती",
        "ai_heatmap": "AI ध्यान हीटमैप",
        "gradcam_overlay": "Grad-CAM ओवरले",
        "where_focused": "### 🧠 मॉडल ने कहाँ ध्यान दिया",
        "primary_focus": "मुख्य फोकस क्षेत्र",
        "attention_coverage": "ध्यान कवरेज",
        "interpretation_note": "⚠️ **व्याख्या नोट:** Grad-CAM दिखाता है कि कौन से छवि क्षेत्र मॉडल के पूर्वानुमान को प्रभावित करते हैं। यह साबित नहीं करता कि हाइलाइट किया गया क्षेत्र रोग से प्रभावित है या रोग की सटीक सीमा है।",
        "pdf_title": "📄 निदान रिपोर्ट",
        "pdf_caption": "मौजूदा LeafGuard एकल-छवि विश्लेषण से डाउनलोड करने योग्य PDF बनाएं।",
        "download_pdf": "📄 निदान रिपोर्ट डाउनलोड करें",
        "history_title": "🕘 स्कैन इतिहास",
        "history_caption": "इस ब्राउज़र सत्र की हाल की स्कैन। सत्र समाप्त होने पर इतिहास साफ़ हो जाएगा।",
        "clear_history": "🗑️ इतिहास साफ़ करें",
        "no_scans": "अभी कोई स्कैन नहीं है। इतिहास बनाने के लिए एकल फोटो या बैच विश्लेषण पूरा करें।",
        "chat_title": "💬 लीफगार्ड AI सहायक",
        "chat_context": "अपने वर्तमान पूर्वानुमान, लक्षण, भरोसे, सुझाई गई कार्रवाई या Grad-CAM के बारे में पूछें।",
        "chat_no_leaf": "अभी कोई पत्ती विश्लेषित नहीं हुई है। LeafGuard, समर्थित फसलों, छवि गुणवत्ता या Grad-CAM के बारे में पूछें।",
        "chat_greeting": "नमस्ते! मैं लीफगार्ड AI सहायक हूँ।",
        "chat_placeholder": "लीफगार्ड AI से पूछें...",
        "clear_chat": "चैट साफ़ करें",
        "try_questions": "पूछें: कौन-सी फसलें समर्थित हैं? • LeafGuard कैसे काम करता है? • Grad-CAM क्या है?",
        "current_analysis": "वर्तमान विश्लेषण",
        "language": "🌐 भाषा",
        "english": "English",
        "hindi": "हिन्दी",
        "features_note": "लीफगार्ड AI v1",
        "footer": "🌱 लीफगार्ड AI v1 • MobileNetV2 • सेब, आलू और टमाटर की 17 स्थितियाँ • भरोसा-आधारित AI • Grad-CAM व्याख्या",
        "model_classes": "🤖 MobileNetV2 • 17 वर्ग",
        "app_summary": "सेब, आलू और टमाटर की पत्ती के स्वास्थ्य का AI विश्लेषण, भरोसा जांच और दृश्य व्याख्या के साथ।",
        "glance": "📊 लीफगार्ड AI एक नज़र में",
        "supported_classes": "समर्थित वर्ग",
        "test_accuracy": "टेस्ट सटीकता",
        "held_out": "हेल्ड-आउट बेंचमार्क",
        "explainability": "व्याख्यात्मकता",
        "visual_attention": "मॉडल का दृश्य ध्यान",
        "apple_potato_tomato": "सेब • आलू • टमाटर",
    },
}

if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"

# Allow the browser-side native-menu language control to persist the choice
# through a URL query parameter before the page title and translations render.
_requested_language = st.query_params.get("lang")
if _requested_language in {"en", "hi"}:
    st.session_state.ui_language = _requested_language


def t(key, *args):
    value = UI_TRANSLATIONS.get(st.session_state.get("ui_language", "en"), UI_TRANSLATIONS["en"]).get(
        key, UI_TRANSLATIONS["en"].get(key, key)
    )
    return value.format(*args) if args else value


def current_app_name():
    return t("app_name")


DISEASE_NAME_HI = {
    "Apple___Apple_scab": "एप्पल स्कैब",
    "Apple___Black_rot": "ब्लैक रॉट",
    "Apple___Cedar_apple_rust": "सीडर एप्पल रस्ट",
    "Apple___healthy": "स्वस्थ",
    "Potato___Early_blight": "अर्ली ब्लाइट",
    "Potato___Late_blight": "लेट ब्लाइट",
    "Potato___healthy": "स्वस्थ",
    "Tomato___Bacterial_spot": "बैक्टीरियल स्पॉट",
    "Tomato___Early_blight": "अर्ली ब्लाइट",
    "Tomato___healthy": "स्वस्थ",
    "Tomato___Late_blight": "लेट ब्लाइट",
    "Tomato___Leaf_Mold": "लीफ मोल्ड",
    "Tomato___Septoria_leaf_spot": "सेप्टोरिया लीफ स्पॉट",
    "Tomato___Spider_mites Two-spotted_spider_mite": "स्पाइडर माइट्स",
    "Tomato___Target_Spot": "टारगेट स्पॉट",
    "Tomato___Tomato_mosaic_virus": "टमाटर मोज़ेक वायरस",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "टमाटर येलो लीफ कर्ल वायरस",
}


def localized_prediction_name(raw_class_name):
    formatted = str(raw_class_name).replace("___", " - ").replace("_", " ")
    if st.session_state.get("ui_language") == "hi":
        crop_map = {"Apple": "सेब", "Potato": "आलू", "Tomato": "टमाटर"}
        if " - " in formatted:
            crop, condition = formatted.split(" - ", 1)
            return f"{crop_map.get(crop, crop)} - {DISEASE_NAME_HI.get(raw_class_name, condition)}"
    return formatted


DISEASE_INFO_HI = {
    "Apple___Apple_scab": {
        "description": "एप्पल स्कैब Venturia inaequalis से होने वाला फफूंद संक्रमण है, जो सेब की पत्तियों और फलों को प्रभावित करता है।",
        "symptoms": "पत्तियों पर जैतून-हरे से काले मखमली धब्बे, पत्तियों का पीला होना और समय से पहले पत्तियाँ गिरना।",
        "recommendation": "पतझड़ में गिरी पत्तियाँ हटाएँ और नष्ट करें, छतरी की छंटाई से हवा का प्रवाह बढ़ाएँ और रातभर पत्तियाँ गीली न रहने दें।",
    },
    "Apple___Black_rot": {
        "description": "ब्लैक रॉट Botryosphaeria obtusa से होने वाला फफूंद रोग है, जो पत्तियों, फलों और शाखाओं को प्रभावित कर सकता है।",
        "symptoms": "बैंगनी किनारों और हल्के केंद्र वाले 'frog-eye' धब्बे, सड़ते फलों पर काले धब्बे और शाखाओं पर कैंकर।",
        "recommendation": "सुप्तावस्था में मृत लकड़ी और कैंकर काटें, सूखे/ममीफाइड फलों को हटाएँ और बाग की स्वच्छता बनाए रखें।",
    },
    "Apple___Cedar_apple_rust": {
        "description": "सीडर एप्पल रस्ट एक फफूंद रोग है जिसके जीवन चक्र में वैकल्पिक मेज़बान की आवश्यकता होती है।",
        "symptoms": "पत्ती की ऊपरी सतह पर चमकीले पीले-नारंगी धब्बे और नीचे छोटे नली-जैसे उभार।",
        "recommendation": "संभव हो तो पास के लाल सीडर या जुनिपर मेज़बान हटाएँ, प्रतिरोधी किस्में चुनें और पेड़ की सामान्य ताकत बनाए रखें।",
    },
    "Apple___healthy": {
        "description": "कोई रोग नहीं मिला। सेब की पत्तियाँ सामान्य हरी और स्वस्थ वृद्धि दिखाती हैं।",
        "symptoms": "साफ, चिकनी हरी पत्तियाँ जिनमें समान बनावट और कोई स्पष्ट घाव या रंग परिवर्तन नहीं है।",
        "recommendation": "संतुलित सिंचाई, नियमित बाग प्रबंधन और समय-समय पर फसल की निगरानी जारी रखें।",
    },
    "Potato___Early_blight": {
        "description": "अर्ली ब्लाइट Alternaria solani से होने वाला सामान्य फफूंद रोग है।",
        "symptoms": "गहरे भूरे गोल धब्बे, जिनमें सांद्र वृत्ताकार छल्ले और आसपास पीले घेरे दिखाई दे सकते हैं।",
        "recommendation": "फसल चक्र अपनाएँ, मिट्टी के छींटे कम करने के लिए मल्च करें, ऊपर से पानी देने से बचें और निचली संक्रमित पत्तियाँ हटाएँ।",
    },
    "Potato___healthy": {
        "description": "कोई रोग नहीं मिला। आलू की पत्तियाँ स्वस्थ और सक्रिय वृद्धि दिखाती हैं।",
        "symptoms": "जीवंत हरी पत्तियाँ जिन पर धब्बे, ब्लाइट या मुरझाने के संकेत नहीं हैं।",
        "recommendation": "मिट्टी की नमी, हिलिंग और नियमित कीट निगरानी बनाए रखें।",
    },
    "Potato___Late_blight": {
        "description": "लेट ब्लाइट Phytophthora infestans से होने वाला एक गंभीर जल-फफूंद (oomycete) रोग है।",
        "symptoms": "पत्ती के किनारों और सिरों पर बड़े, गहरे पानी-सिक्त घाव; नम मौसम में नीचे की ओर सफेद वृद्धि दिखाई दे सकती है।",
        "recommendation": "संक्रमित पौधों को जल्दी हटाएँ और नष्ट करें, अच्छी जल निकासी रखें और स्थानीय कृषि सलाह का पालन करें।",
    },
    "Tomato___Bacterial_spot": {
        "description": "बैक्टीरियल स्पॉट Xanthomonas प्रजातियों से होने वाला रोग है, जो टमाटर की पत्तियों, तनों और फलों को प्रभावित करता है।",
        "symptoms": "छोटे गहरे पानी-सिक्त धब्बे जो सूखकर गहरे भूरे खुरदरे घाव बन सकते हैं, साथ में पीलापन और पत्ती गिरना।",
        "recommendation": "गीली पत्तियों पर काम करने से बचें, प्रमाणित रोग-मुक्त बीज लें, बहुवर्षीय फसल चक्र अपनाएँ और ऊपर से सिंचाई से बचें।",
    },
    "Tomato___Early_blight": {
        "description": "टमाटर का अर्ली ब्लाइट Alternaria solani से होने वाला सामान्य फफूंद रोग है।",
        "symptoms": "पुरानी निचली पत्तियों पर गहरे भूरे सांद्र छल्लों वाले धब्बे, जिससे पत्तियाँ पीली होकर गिर सकती हैं।",
        "recommendation": "पौधों को सहारा दें, मिट्टी पर मल्च करें, संक्रमित निचली पत्तियाँ हटाएँ और पर्याप्त हवा का प्रवाह रखें।",
    },
    "Tomato___healthy": {
        "description": "कोई रोग नहीं मिला। टमाटर स्वस्थ और मजबूत वृद्धि दिखाता है।",
        "symptoms": "गहरी हरी सामान्य पत्तियाँ जिनमें धब्बे या ऊतक मृत्यु के संकेत नहीं हैं।",
        "recommendation": "जड़ों के पास नियमित पानी दें, उचित ट्रेलिसिंग करें और समय-समय पर पत्तियों का निरीक्षण करें।",
    },
    "Tomato___Late_blight": {
        "description": "लेट ब्लाइट Phytophthora infestans से होने वाला गंभीर रोग है जो टमाटर को तेजी से प्रभावित कर सकता है।",
        "symptoms": "अनियमित गहरे भूरे पानी-सिक्त धब्बे, नमी में नीचे की सतह पर सफेद वृद्धि और तनों पर गहरे घाव।",
        "recommendation": "संक्रमित पौध सामग्री तुरंत हटाकर नष्ट करें, पत्तियों को गीला होने से बचाएँ और बहुवर्षीय फसल चक्र अपनाएँ।",
    },
    "Tomato___Leaf_Mold": {
        "description": "लीफ मोल्ड Passalora fulva से होने वाला फफूंद रोग है और अधिक नमी में आम है।",
        "symptoms": "ऊपरी सतह पर हल्के पीले धब्बे, जिनके नीचे जैतून-हरे से हल्के भूरे मखमली फफूंद की वृद्धि हो सकती है।",
        "recommendation": "पौधों के बीच हवा का प्रवाह और वेंटिलेशन बढ़ाएँ, नमी घटाएँ और घनी निचली पत्तियाँ छाँटें।",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "सेप्टोरिया लीफ स्पॉट Septoria lycopersici से होने वाला पत्ती रोग है।",
        "symptoms": "छोटे गोल धब्बे जिनकी गहरी भूरी किनारी और हल्के केंद्र होते हैं, कभी-कभी छोटे काले फलन शरीर के साथ।",
        "recommendation": "संक्रमित निचली पत्तियाँ हटाएँ, मिट्टी के छींटे रोकने के लिए मल्च करें और ऊपर से स्प्रिंकलर सिंचाई से बचें।",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "टू-स्पॉटेड स्पाइडर माइट्स द्वारा होने वाला नुकसान पत्तियों से रस चूसने वाली सूक्ष्म अरैक्निड प्रजाति के कारण होता है।",
        "symptoms": "पत्तियों पर महीन पीले/सफेद बिंदु, कांस्य-पीला रंग और नीचे महीन जाला।",
        "recommendation": "पत्तियों के नीचे पानी का हल्का स्प्रे करें, आवश्यकता पर नीम तेल या कीटनाशी साबुन उपयोग करें और लाभकारी परभक्षियों को बढ़ावा दें।",
    },
    "Tomato___Target_Spot": {
        "description": "टारगेट स्पॉट Corynespora cassiicola से होने वाला फफूंद पत्ती रोग है।",
        "symptoms": "छोटे बिंदु जो भूरे गोल घावों में फैलते हैं, हल्के केंद्र और सांद्र छल्लों के साथ।",
        "recommendation": "पत्तियाँ जल्दी सूखें इसके लिए पर्याप्त अंतर रखें, संक्रमित निचली पत्तियाँ हटाएँ और फसल चक्र अपनाएँ।",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "टमाटर मोज़ेक वायरस (ToMV) एक टिकाऊ वायरल रोगजनक है जो मुख्यतः संपर्क से फैलता है।",
        "symptoms": "हल्के और गहरे हरे मोज़ेक पैटर्न, पत्ती विकृति, बौनापन और सिकुड़ी हुई वृद्धि।",
        "recommendation": "संक्रमित पौधों को हटाकर नष्ट करें, औज़ारों को अच्छी तरह साफ करें और पौधों को छूने से पहले हाथ धोएँ।",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "टमाटर येलो लीफ कर्ल वायरस (TYLCV) मुख्यतः सिल्वरलीफ व्हाइटफ्लाई द्वारा फैलने वाला वायरल रोग है।",
        "symptoms": "पत्तियों का ऊपर की ओर मुड़ना, किनारों पर पीलापन, पत्तियों का छोटा होना और पौधे की वृद्धि रुकना।",
        "recommendation": "व्हाइटफ्लाई नियंत्रण के लिए जाल या परावर्तक मल्च का उपयोग करें, संक्रमित स्रोत पौधे हटाएँ और प्रतिरोधी किस्में चुनें।",
    },
}


def localized_disease_info(raw_class_name, fallback):
    if st.session_state.get("ui_language") == "hi":
        return DISEASE_INFO_HI.get(raw_class_name, fallback)
    return fallback

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=("लीफगार्ड AI" if st.session_state.ui_language == "hi" else "LeafGuard AI"),
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

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "history_single_recorded_hash" not in st.session_state:
    st.session_state.history_single_recorded_hash = None

if "history_batch_recorded_signature" not in st.session_state:
    st.session_state.history_batch_recorded_signature = None

if "weather_data" not in st.session_state:
    st.session_state.weather_data = None

if "weather_location_label" not in st.session_state:
    st.session_state.weather_location_label = None


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

    st.markdown(f"## 🌱 {current_app_name()}")
    st.caption(t("smart_crop"))

    st.divider()
    st.markdown(t("supported_crops"))
    st.markdown(t("supported_crops_body"))

    st.divider()
    st.markdown(t("how_it_works"))
    st.markdown(t("how_it_works_body"))

    st.divider()
    st.markdown(t("features"))
    st.markdown(t("features_body"))

    st.divider()
    st.markdown(t("safety"))
    st.caption(t("safety_caption"))

    st.divider()
    st.caption(t("features_note"))


# ============================================================
# ============================================================
# TOP-RIGHT WEATHER CONTROL (STEP 1 ONLY)
# Keep the native Streamlit ⋮ menu in the top-right toolbar.
# LeafGuard customizes that native menu with JavaScript below.
# ============================================================

# Weather stays under the top toolbar on Step 1 only.
_weather_spacer, _weather_top = st.columns([8.5, 1.5], gap="small")

if st.session_state.analysis_stage == 1:
    with _weather_top:
        with st.popover(t("weather"), use_container_width=True):
            st.markdown(t("weather_heading"))
            st.caption(t("weather_caption"))

            weather_location = st.text_input(
                t("location"),
                value=st.session_state.weather_location_label or "",
                placeholder="Jammu, India",
                key="weather_location_input",
            )

            weather_btn = st.button(
                t("get_weather"),
                key="weather_get_button",
                use_container_width=True,
            )

            if weather_btn:
                if not weather_location.strip():
                    st.warning(t("enter_city"))
                else:
                    with st.spinner(t("fetching_weather")):
                        try:
                            st.session_state.weather_data = fetch_weather_for_location(
                                weather_location.strip()
                            )
                            st.session_state.weather_location_label = weather_location.strip()
                        except RuntimeError as exc:
                            st.session_state.weather_data = None
                            st.error(str(exc))
                        except Exception as exc:
                            st.session_state.weather_data = None
                            st.error(f"{t('weather_lookup_failed')} {exc}")

            if st.session_state.weather_data:
                w = st.session_state.weather_data
                st.caption(f"📍 {w.get('location', t('unknown_location'))}")
                st.markdown(
                    f"{t('current_condition')} {localized_weather_description(w.get('weather_code'))}"
                )

                temp_col1, temp_col2 = st.columns(2)
                with temp_col1:
                    temp = w.get("temperature")
                    st.metric(
                        t("temperature"),
                        f"{temp:.1f} °C" if isinstance(temp, (int, float)) else "—",
                    )
                with temp_col2:
                    feels = w.get("apparent_temperature")
                    st.metric(
                        t("feels_like"),
                        f"{feels:.1f} °C" if isinstance(feels, (int, float)) else "—",
                    )

                temp_col3, temp_col4 = st.columns(2)
                with temp_col3:
                    humidity = w.get("humidity")
                    st.metric(
                        t("humidity"),
                        f"{humidity:.0f}%" if isinstance(humidity, (int, float)) else "—",
                    )
                with temp_col4:
                    rain_next = w.get("precipitation_probability")
                    st.metric(
                        t("rain_next_hour"),
                        f"{rain_next:.0f}%" if isinstance(rain_next, (int, float)) else "—",
                    )

                st.markdown(t("detailed_conditions"))
                detail_rows = [
                    (t("precipitation"), f"{w['precipitation']:.1f} mm" if isinstance(w.get("precipitation"), (int, float)) else "—"),
                    (t("wind"), f"{w['wind_speed']:.1f} km/h" if isinstance(w.get("wind_speed"), (int, float)) else "—"),
                    (t("wind_gusts"), f"{w['wind_gusts']:.1f} km/h" if isinstance(w.get("wind_gusts"), (int, float)) else "—"),
                    (t("cloud_cover"), f"{w['cloud_cover']:.0f}%" if isinstance(w.get("cloud_cover"), (int, float)) else "—"),
                    (t("dew_point"), f"{w['dew_point']:.1f} °C" if isinstance(w.get("dew_point"), (int, float)) else "—"),
                    (t("evapotranspiration"), f"{w['evapotranspiration']:.2f} mm" if isinstance(w.get("evapotranspiration"), (int, float)) else "—"),
                    (t("vpd"), f"{w['vpd']:.2f} kPa" if isinstance(w.get("vpd"), (int, float)) else "—"),
                ]
                detail_left, detail_right = st.columns(2)
                for idx, (label, value) in enumerate(detail_rows):
                    target_col = detail_left if idx % 2 == 0 else detail_right
                    with target_col:
                        st.markdown(f"**{label}**  \n{value}")

                st.caption(
                    f"{t('local_weather_time')}: {w.get('time', '—')} • "
                    f"{t('timezone')}: {w.get('timezone', '—')}"
                )
            else:
                st.info(t("weather_info"))
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

    st.title(current_app_name())

    st.caption(t("smart_crop"))

with status_col:
    st.caption(t("model_classes"))
    


st.write(t("app_summary"))

st.divider()


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

st.subheader(t("glance"))

metric_1, metric_2, metric_3 = st.columns(
    3
)

with metric_1:

    st.metric(
        label=t("supported_classes"),
        value="17"
    )

    st.caption(t("apple_potato_tomato"))

with metric_2:

    st.metric(
        label=t("test_accuracy"),
        value="87.70%"
    )

    st.caption(t("held_out"))

with metric_3:

    st.metric(
        label=t("explainability"),
        value="Grad-CAM"
    )

    st.caption(t("visual_attention"))


st.divider()


# ============================================================
# SCAN HISTORY HELPERS
# Session-only history for completed single and batch analyses.
# ============================================================

def _format_prediction_for_history(raw_class_name):
    formatted = str(raw_class_name).replace("___", " - ").replace("_", " ")
    if " - " in formatted:
        crop_name, condition_name = formatted.split(" - ", 1)
    else:
        crop_name, condition_name = "Unknown crop", formatted
    return crop_name, condition_name


def _append_scan_history(record, unique_key):
    if unique_key in {item.get("_key") for item in st.session_state.scan_history}:
        return

    history_record = dict(record)
    history_record["_key"] = unique_key
    st.session_state.scan_history.append(history_record)


def _single_history_record(result, image_name):
    raw_class = result.get("predicted_class_name", "Unknown")
    crop_name, condition_name = _format_prediction_for_history(raw_class)
    confidence = float(result.get("confidence", 0.0))
    probabilities = sorted(
        [float(p) for p in result.get("all_probabilities", [])],
        reverse=True,
    )
    top1 = probabilities[0] if probabilities else confidence
    top2 = probabilities[1] if len(probabilities) > 1 else 0.0
    margin = top1 - top2
    uncertain = top1 < 0.50 or margin < 0.20
    healthy = "healthy" in raw_class.lower()

    if uncertain:
        status = "Uncertain"
    elif healthy:
        status = "No disease detected"
    else:
        status = "Crop disease detected"

    return {
        "Date & Time": datetime.now().strftime("%d %b %Y, %H:%M"),
        "Type": "Single",
        "Image": image_name,
        "Crop": crop_name,
        "Prediction": condition_name,
        "Confidence": f"{confidence * 100.0:.2f}%",
        "Status": status,
    }


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
# ============================================================
# 9. STEP 1 INTRO
# ============================================================

if st.session_state.analysis_stage == 1:

    st.subheader(t("step1"))
    st.progress(
        0.25,
        text=t("step1_progress"),
    )
    st.info(t("upload_info"))


# 10. STEP 1 - UPLOAD
# ============================================================

if st.session_state.analysis_stage == 1:

    single_col, batch_col = st.columns(

        2,
        gap="large"
    )

    # ------------------------------------------------------------
    # SINGLE PHOTO
    # ------------------------------------------------------------

    with single_col:

        with st.container(border=True):

            st.markdown(t("single_photo"))
            st.caption(t("single_caption"))

            uploaded_file = st.file_uploader(
                t("single_uploader"),
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
                        st.session_state.history_single_recorded_hash = None

                    image = Image.open(
                        io.BytesIO(single_file_bytes)
                    ).convert("RGB")

                    st.image(
                        image,
                        caption=f"{t('uploaded_leaf')} • {uploaded_file.name}",
                        width="stretch"
                    )

                    single_quality_ok, single_quality_msg = check_image_quality(image)

                    q1, q2 = st.columns(2)

                    with q1:
                        st.write(
                            f"**{t('resolution')}**  \n"
                            f"{image.width} × {image.height}px"
                        )

                    with q2:
                        if single_quality_ok:
                            st.success(t("quality_passed"))
                        else:
                            st.warning(t("quality_issue"))

                    if not single_quality_ok:
                        st.caption(single_quality_msg)

                except Exception as err:
                    st.error(t("invalid_image", err))

            else:
                st.caption(t("single_empty"))

    # ------------------------------------------------------------
    # BATCH ANALYSIS
    # ------------------------------------------------------------

    with batch_col:

        with st.container(border=True):

            st.markdown(t("batch"))
            st.caption(t("batch_caption"))

            batch_files = st.file_uploader(
                t("batch_uploader"),
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
                    st.session_state.history_batch_recorded_signature = None

                st.info(t("selected_images", len(batch_files)))

            else:
                st.session_state.batch_results = []
                st.session_state.batch_image_bytes = {}
                st.session_state.batch_file_signature = None
                st.caption(t("batch_empty"))

    # ------------------------------------------------------------
    # ONE COMMON ANALYZE LEAF BUTTON
    # ------------------------------------------------------------

    st.write("")

    if uploaded_file is not None and batch_files:
        st.warning(t("both_selected"))

    analyze_leaf_btn = st.button(
        t("analyze_leaf"),
        type="primary",
        width="stretch",
        key="common_analyze_leaf_button"
    )

    if analyze_leaf_btn:

        single_selected = uploaded_file is not None
        batch_selected = bool(batch_files)

        if single_selected and batch_selected:

            st.error(t("select_one"))

        elif single_selected:

            if not single_quality_ok:
                st.error(t("analysis_stopped"))

            else:
                with st.spinner(t("running_analysis")):
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

                        history_key = f"single:{current_file_hash}"
                        if st.session_state.history_single_recorded_hash != history_key:
                            _append_scan_history(
                                _single_history_record(result, uploaded_file.name),
                                history_key,
                            )
                            st.session_state.history_single_recorded_hash = history_key

                        st.rerun()

                    except Exception as err:
                        st.error(t("prediction_error", err))

        elif batch_selected:

            predictor = load_leaf_predictor()
            results = []

            with st.spinner(t("analyzing_batch")):

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

            if st.session_state.history_batch_recorded_signature != batch_signature:
                batch_timestamp = datetime.now().strftime("%d %b %Y, %H:%M")
                for batch_row in results:
                    _append_scan_history(
                        {
                            "Date & Time": batch_timestamp,
                            "Type": "Batch",
                            "Image": batch_row.get("Image", "—"),
                            "Crop": batch_row.get("Crop", "—"),
                            "Prediction": batch_row.get("Prediction", "—"),
                            "Confidence": batch_row.get("Confidence", "—"),
                            "Status": batch_row.get("Status", "—"),
                        },
                        f"batch:{batch_signature}:{batch_row.get('Image', '—')}",
                    )
                st.session_state.history_batch_recorded_signature = batch_signature

            st.rerun()

        else:
            st.warning(t("upload_first"))


# ============================================================
# 11. STEP 2 - AI DIAGNOSIS / BATCH RESULTS
#     Appears on the next screen after Step 1 analysis.
# ============================================================

if st.session_state.analysis_stage == 2:

    if st.button(t("back"), key="new_analysis_button"):
        st.session_state.analysis_stage = 1
        st.session_state.analysis_mode = None
        st.session_state.analysis_result = None
        st.session_state.analysis_image_bytes = None
        st.session_state.gradcam_image = None
        st.session_state.batch_results = []
        st.session_state.batch_image_bytes = {}
        st.session_state.batch_file_signature = None
        st.session_state.analysis_file_hash = None
        st.session_state.history_single_recorded_hash = None
        st.session_state.history_batch_recorded_signature = None
        st.session_state.chat_messages = []
        st.session_state.chat_open = False
        st.rerun()

    if st.session_state.analysis_mode == "batch":

        st.divider()
        st.subheader(t("batch_results"))
        st.progress(0.50, text=t("batch_progress"))
        st.caption(t("batch_results_caption"))

        if st.session_state.batch_results:

            batch_display = []
            batch_status_map = {
                "Crop disease detected": "फसल रोग पाया गया",
                "No disease detected": "कोई रोग नहीं मिला",
                "Uncertain": "अनिश्चित",
                "Not analyzed": "विश्लेषण नहीं हुआ",
                "Quality check failed": "गुणवत्ता जांच विफल",
                "Analysis error": "विश्लेषण त्रुटि",
            }
            for batch_item in st.session_state.batch_results:
                batch_display.append({
                    t("history_image"): batch_item.get("Image", "—"),
                    t("crop"): batch_item.get("Crop", "—"),
                    t("history_prediction"): batch_item.get("Prediction", "—"),
                    t("history_confidence"): batch_item.get("Confidence", "—"),
                    t("history_status"): (
                        batch_status_map.get(batch_item.get("Status"), batch_item.get("Status", "—"))
                        if st.session_state.ui_language == "hi"
                        else batch_item.get("Status", "—")
                    ),
                    "गुणवत्ता" if st.session_state.ui_language == "hi" else "Quality": batch_item.get("Quality", "—"),
                })
            st.dataframe(batch_display, width="stretch", hide_index=True)

            import csv
            csv_buffer = io.StringIO()
            fieldnames = ["Image", "Crop", "Prediction", "Confidence", "Status", "Quality"]
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(st.session_state.batch_results)

            st.download_button(
                t("download_csv"),
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
                    t("download_batch_pdf"),
                    data=batch_pdf,
                    file_name="leafguard_batch_analysis_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="batch_pdf_report_download",
                )
            except Exception as batch_pdf_error:
                st.error(f"❌ Could not create the batch PDF report: {batch_pdf_error}")

        else:
            st.info(t("no_batch_results"))

    elif st.session_state.analysis_mode == "single":

        if st.session_state.analysis_result is not None:

            st.divider()

            st.subheader(t("step2"))

            st.progress(
                0.50,
                text=t("step2_progress")
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

            formatted_name = localized_prediction_name(raw_class_name)

            is_healthy = (
                "healthy"
                in raw_class_name.lower()
            )


            # --------------------------------------------------------
            # Clean diagnosis summary
            # --------------------------------------------------------

            st.markdown(t("prediction_summary"))

            prediction_col, confidence_col = st.columns(
                [1.55, 1],
                vertical_alignment="center"
            )

            with prediction_col:

                crop_name = formatted_name.split(" - ", 1)[0] if " - " in formatted_name else "Unknown crop"

                st.markdown(f"### 🌱 {crop_name}")
                st.caption(t("detected_crop"))

                if is_unrecognized:

                    st.warning(t("unrecognized"))

                    st.markdown(
                        f"## {formatted_name}"
                    )

                    st.caption(t("unrecognized_caption"))

                elif confidence_pct < 70.0:

                    st.warning(t("low_confidence"))

                    st.markdown(
                        f"## {formatted_name}"
                    )

                    st.caption(t("low_confidence_caption"))

                elif is_healthy:

                    st.success(t("no_disease"))

                    disease_name = (
                        formatted_name.split(" - ", 1)[-1]
                        if " - " in formatted_name
                        else formatted_name
                    )

                    st.markdown(
                        f"## {disease_name}"
                    )

                    st.caption(
                        t("no_disease_caption")
                    )

                else:

                    st.error(t("disease_detected"))

                    disease_name = (
                        formatted_name.split(" - ", 1)[-1]
                        if " - " in formatted_name
                        else formatted_name
                    )

                    st.markdown(
                        f"## {disease_name}"
                    )

                    st.caption(
                        t("disease_detected_caption")
                    )

            with confidence_col:

                with st.container(border=True):

                    st.markdown(t("model_confidence"))

                    st.metric(
                        label=t("model_certainty"),
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
                        t("confidence_margin", margin)
                    )


            # --------------------------------------------------------
            # Decision note
            # --------------------------------------------------------

            if is_unrecognized:

                st.warning(t("decision_uncertain"))

            elif confidence_pct < 70.0:

                st.info(t("decision_low"))

            else:

                st.caption(t("supported_scope"))


            # ========================================================
            # STEP 3 - UNDERSTAND RESULT
            # ========================================================

            st.divider()

            st.subheader(t("step3"))

            st.progress(
                0.75,
                text=t("step3_progress")
            )

            if is_unrecognized:

                st.info(t("step3_uncertain"))

            else:

                fallback_info = {
                    "description": t("no_disease_info"),
                    "symptoms": t("na"),
                    "recommendation": t("consult_extension"),
                }
                info = localized_disease_info(
                    raw_class_name,
                    DISEASE_INFO.get(raw_class_name, fallback_info),
                )

                info_col1, info_col2 = st.columns(
                    2
                )

                with info_col1:

                    with st.container(border=True):

                        st.markdown(t("condition_overview"))

                        st.write(
                            info["description"]
                        )

                with info_col2:

                    with st.container(border=True):

                        st.markdown(t("common_symptoms"))

                        st.write(
                            info["symptoms"]
                        )

                with st.container(border=True):

                    st.markdown(t("recommended_action"))

                    st.write(
                        info["recommendation"]
                    )


            # ========================================================
            # STEP 4 - GRAD-CAM
            # ========================================================

            st.divider()

            st.subheader(t("step4"))

            st.progress(
                1.0,
                text=t("step4_progress")
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

                st.markdown(t("prediction_explained"))

                explain_col1, explain_col2 = st.columns(2)

                with explain_col1:

                    with st.container(border=True):

                        st.caption(t("crop"))
                        st.markdown(
                            f"### 🌱 {formatted_name.split(' - ', 1)[0]}"
                        )

                with explain_col2:

                    with st.container(border=True):

                        st.caption(t("crop_disease"))
                        st.markdown(
                            f"### 🦠 {disease_display_name}"
                        )

                st.info(
                    t("gradcam_info", DISEASE_NAME_HI.get(raw_class_name, disease_display_name) if st.session_state.ui_language == "hi" else disease_display_name)
                )

            else:

                st.info(t("gradcam_unavailable"))

            st.write(t("gradcam_text"))

            st.caption(t("gradcam_color_caption"))

            # --------------------------------------------------------
            # Generate Grad-CAM button
            # --------------------------------------------------------

            if not is_unrecognized:

                explain_btn = st.button(
                    t("generate_gradcam"),
                    type="secondary",
                    width="stretch",
                    key="gradcam_button"
                )

                if explain_btn:

                    with st.spinner(t("generating_gradcam")):

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

                st.markdown(t("gradcam_title"))

                original_col, heatmap_col, overlay_col = st.columns(3)

                with original_col:

                    st.image(
                        gradcam_data["original"],
                        caption=t("original_leaf"),
                        width="stretch"
                    )

                with heatmap_col:

                    st.image(
                        gradcam_data["heatmap"],
                        caption=t("ai_heatmap"),
                        width="stretch"
                    )

                with overlay_col:

                    st.image(
                        gradcam_data["overlay"],
                        caption=t("gradcam_overlay"),
                        width="stretch"
                    )

                st.markdown(t("where_focused"))

                st.info(
                    gradcam_data["explanation"]
                )

                focus_col1, focus_col2 = st.columns(2)

                with focus_col1:

                    st.metric(
                        t("primary_focus"),
                        gradcam_data["focus_region"]
                    )

                with focus_col2:

                    st.metric(
                        t("attention_coverage"),
                        f"{gradcam_data['attention_coverage']:.1f}%"
                    )

                st.warning(t("interpretation_note"))




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
    st.subheader(t("pdf_title"))
    st.caption(t("pdf_caption"))

    try:
        diagnosis_pdf = build_diagnosis_pdf(
            st.session_state.analysis_result,
            st.session_state.analysis_image_bytes,
            st.session_state.gradcam_image,
        )

        st.download_button(
            t("download_pdf"),
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
                f'<div class="leafguard-chat-title">{t("chat_title")}</div>',
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
            f'<div class="leafguard-chat-context">{t("chat_context")}</div>',
            unsafe_allow_html=True,
        )

        # The assistant is available before upload and becomes result-aware after analysis.
        if st.session_state.analysis_result is None:
            st.markdown(
                f'<div class="leafguard-chat-context">{t("chat_no_leaf")}</div>',
                unsafe_allow_html=True,
            )
        else:
            chat_result = st.session_state.analysis_result
            chat_class = chat_result["predicted_class_name"]
            chat_formatted = localized_prediction_name(chat_class)
            chat_crop = chat_formatted.split(" - ", 1)[0] if " - " in chat_formatted else "Unknown crop"
            chat_disease = chat_formatted.split(" - ", 1)[-1] if " - " in chat_formatted else chat_formatted
            chat_confidence = float(chat_result["confidence"]) * 100.0

            with st.container(border=True):
                st.caption(t("current_analysis"))
                context_col1, context_col2, context_col3 = st.columns(3)
                with context_col1:
                    st.markdown(f"**🌱 {chat_crop}**")
                with context_col2:
                    st.markdown(f"**🦠 {chat_disease}**")
                with context_col3:
                    st.markdown(f"**🎯 {chat_confidence:.2f}%**")

        # Initial greeting.
        if len(st.session_state.chat_messages) == 0:
            greeting = t("chat_greeting")
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
                return "कृपया अपने LeafGuard परिणाम के बारे में प्रश्न लिखें।" if st.session_state.ui_language == "hi" else "Please type a question about your LeafGuard result."

            if any(term in q for term in [
                "supported crop",
                "which crops",
                "what crops",
                "supported plant",
                "supported plants"
            ]):
                return (
                    "LeafGuard AI सेब, आलू और टमाटर की पत्ती की स्थितियों की 17 प्रशिक्षित श्रेणियों में पहचान कर सकता है।"
                    if st.session_state.ui_language == "hi"
                    else "LeafGuard AI currently supports Apple, Potato, and Tomato leaf conditions across 17 trained classes."
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
                    "कृपया पहले पत्ती की फोटो अपलोड करके उसका विश्लेषण करें। परिणाम मिलने के बाद मैं फसल, पूर्वानुमान, भरोसा, लक्षण, सुझाई गई सामान्य कार्रवाई और Grad-CAM समझा सकता हूँ।"
                    if st.session_state.ui_language == "hi"
                    else "Please upload and analyze a leaf first. After a result is available, I can explain the crop, prediction, confidence, symptoms, recommended general action, and Grad-CAM."
                )

            current_result = st.session_state.analysis_result
            current_class = current_result["predicted_class_name"]
            current_confidence = float(current_result["confidence"])
            current_confidence_pct = current_confidence * 100.0
            current_formatted = localized_prediction_name(current_class)
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

            fallback_info = {"description": t("no_disease_info"), "symptoms": t("na"), "recommendation": t("consult_extension")}
            current_info = localized_disease_info(current_class, DISEASE_INFO.get(current_class, fallback_info))

            if any(term in q for term in [
                "what is the crop",
                "which crop",
                "crop name",
                "plant name"
            ]):
                return f"{'पहचानी गई फसल' if st.session_state.ui_language == 'hi' else 'The detected crop is'} {current_crop}."

            if any(term in q for term in [
                "what disease",
                "which disease",
                "disease name",
                "diagnosis",
                "what did you detect",
                "what is wrong"
            ]):
                if current_healthy:
                    return f"{current_crop} {'स्वस्थ है। कोई रोग नहीं मिला।' if st.session_state.ui_language == 'hi' else 'was detected as healthy. No disease was detected.'}"
                return f"{current_crop} की पत्ती पर {current_disease} {'की पहचान हुई है।' if st.session_state.ui_language == 'hi' else 'was identified by LeafGuard.'}"

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
            f'<div class="leafguard-chat-suggestions">{t("try_questions")}</div>',
            unsafe_allow_html=True,
        )

        with st.form("leafguard_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([6, 1])
            with input_col:
                user_question = st.text_input(
                    t("chat_placeholder"),
                    key="leafguard_chat_input",
                    label_visibility="collapsed",
                    placeholder=t("chat_placeholder"),
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


# --------------------------------------------------------
# Customize Streamlit's native top-right ⋮ menu
# Keep the built-in System / Light / Dark theme selector, while replacing
# developer/viewer actions with LeafGuard's Refresh and Language controls.
# Streamlit 1.64 provides st.html(..., unsafe_allow_javascript=True), which
# lets us adapt the already-rendered native menu without adding a second menu.
# --------------------------------------------------------
st.html(
    r"""
    <style>
      /* Keep Streamlit's native theme selector, but make the LeafGuard
         menu contain only Refresh + Language underneath it. */
      .leafguard-native-menu-item {
        width: 100%;
        box-sizing: border-box;
        border: 0;
        background: transparent;
        color: inherit;
        text-align: left;
        padding: 0.55rem 0.75rem;
        border-radius: 0.35rem;
        cursor: pointer;
        font: inherit;
        display: block;
      }
      .leafguard-native-menu-item:hover {
        background: rgba(128, 128, 128, 0.16);
      }
      .leafguard-native-menu-divider {
        border-top: 1px solid rgba(128,128,128,0.24);
        margin: 0.35rem 0;
      }
      .leafguard-language-panel {
        display: none;
        gap: 0.35rem;
        padding: 0.15rem 0.5rem 0.55rem 0.5rem;
      }
      .leafguard-language-panel button {
        flex: 1;
        border: 1px solid rgba(128,128,128,0.28);
        background: transparent;
        color: inherit;
        border-radius: 0.35rem;
        padding: 0.4rem 0.5rem;
        cursor: pointer;
        font: inherit;
      }
      .leafguard-language-panel button:hover {
        background: rgba(128,128,128,0.16);
      }
    </style>
    <script>
    (() => {
      const HIDDEN = [
        "Rerun", "Auto rerun", "Clear cache", "Print", "Record screen",
        "Made with Streamlit"
      ];

      function clean(text) {
        return (text || "").replace(/\s+/g, " ").trim();
      }

      function isHiddenLabel(label) {
        return HIDDEN.some((x) => label === x || label.startsWith(x + " "));
      }

      function findNativeMenu() {
        const candidates = [
          ...document.querySelectorAll('[role="menu"]'),
          ...document.querySelectorAll('[data-baseweb="menu"]'),
          ...document.querySelectorAll('[role="listbox"]')
        ];
        const withStreamlitActions = candidates.find((menu) => {
          const txt = clean(menu.innerText);
          return txt.includes("System") && (txt.includes("Print") || txt.includes("Record screen"));
        });
        if (withStreamlitActions) return withStreamlitActions;

        /* Fallback: find a visible element containing the native action text. */
        const action = [...document.querySelectorAll("button, [role='menuitem'], li, div")]
          .find((el) => {
            const txt = clean(el.innerText);
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && (txt === "Print" || txt.startsWith("Record screen"));
          });
        if (!action) return null;
        return action.closest('[role="menu"]')
          || action.closest('[data-baseweb="menu"]')
          || action.closest('[role="listbox"]')
          || action.parentElement?.parentElement
          || null;
      }

      function makeButton(label, title) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "leafguard-native-menu-item";
        b.textContent = label;
        if (title) b.title = title;
        return b;
      }

      function setLanguage(code) {
        const url = new URL(window.location.href);
        url.searchParams.set("lang", code);
        window.location.assign(url.toString());
      }

      function customize() {
        const menu = findNativeMenu();
        if (!menu) return;

        /* Streamlit can recreate the menu every time it opens. Use a marker
           on the actual menu element so each instance is customized once. */
        if (menu.dataset.leafguardCustomized === "1") return;

        const descendants = [...menu.querySelectorAll("button, [role='menuitem'], li, div")];
        descendants.forEach((el) => {
          const label = clean(el.innerText);
          if (!label) return;
          if (isHiddenLabel(label)) {
            el.style.display = "none";
          }
        });

        const divider = document.createElement("div");
        divider.className = "leafguard-native-menu-divider";

        const refresh = makeButton("↻ Refresh", "Refresh the full LeafGuard page");
        refresh.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          window.location.reload();
        });

        const language = makeButton("🌐 Language", "Choose LeafGuard interface language");
        const panel = document.createElement("div");
        panel.className = "leafguard-language-panel";
        panel.style.display = "none";

        const english = makeButton("English");
        const hindi = makeButton("हिन्दी");
        english.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          setLanguage("en");
        });
        hindi.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          setLanguage("hi");
        });
        panel.append(english, hindi);

        language.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          panel.style.display = panel.style.display === "flex" ? "none" : "flex";
        });

        /* All unwanted native actions are hidden, so appending these places
           the LeafGuard controls immediately below System/Light/Dark. */
        menu.append(divider, refresh, language, panel);
        menu.dataset.leafguardCustomized = "1";
      }

      function scheduleCustomize() {
        customize();
        setTimeout(customize, 50);
        setTimeout(customize, 200);
        setTimeout(customize, 600);
      }

      const observer = new MutationObserver(scheduleCustomize);
      observer.observe(document.documentElement, { childList: true, subtree: true });
      document.addEventListener("click", () => setTimeout(customize, 30), true);
      setTimeout(scheduleCustomize, 250);
      setTimeout(scheduleCustomize, 800);
      setTimeout(scheduleCustomize, 1600);
      setInterval(customize, 1200);
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# --------------------------------------------------------
# Clear Back button styling
# --------------------------------------------------------
st.markdown(
    """
    <style>
    .st-key-new_analysis_button button {
        min-width: 105px !important;
        min-height: 42px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.35rem 0.9rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 12. SCAN HISTORY
# Session-only history of completed analyses.
# ============================================================

st.divider()
st.subheader(t("history_title"))
st.caption(t("history_caption"))

if st.session_state.scan_history:

    history_view = []
    status_hi = {
        "Crop disease detected": "फसल रोग पाया गया",
        "No disease detected": "कोई रोग नहीं मिला",
        "Uncertain": "अनिश्चित",
        "Not analyzed": "विश्लेषण नहीं हुआ",
        "Quality check failed": "गुणवत्ता जांच विफल",
        "Analysis error": "विश्लेषण त्रुटि",
    }
    type_hi = {"Single": "एकल", "Batch": "बैच"}
    for item in reversed(st.session_state.scan_history):
        status_value = item.get("Status", "—")
        type_value = item.get("Type", "—")
        history_view.append(
            {
                t("history_date"): item.get("Date & Time", "—"),
                t("history_type"): type_hi.get(type_value, type_value) if st.session_state.ui_language == "hi" else type_value,
                t("history_image"): item.get("Image", "—"),
                t("crop"): ( {"Apple": "सेब", "Potato": "आलू", "Tomato": "टमाटर"}.get(item.get("Crop"), item.get("Crop", "—")) if st.session_state.ui_language == "hi" else item.get("Crop", "—") ),
                t("history_prediction"): (DISEASE_NAME_HI.get(item.get("Prediction"), item.get("Prediction", "—")) if st.session_state.ui_language == "hi" else item.get("Prediction", "—")),
                t("history_confidence"): item.get("Confidence", "—"),
                t("history_status"): status_hi.get(status_value, status_value) if st.session_state.ui_language == "hi" else status_value,
            }
        )

    history_col1, history_col2 = st.columns([5, 1])

    with history_col1:
        st.dataframe(
            history_view,
            width="stretch",
            hide_index=True,
        )

    with history_col2:
        st.write("")
        st.write("")
        if st.button(t("clear_history"), key="clear_scan_history", width="stretch"):
            st.session_state.scan_history = []
            st.session_state.history_single_recorded_hash = None
            st.session_state.history_batch_recorded_signature = None
            st.rerun()

else:
    st.info(t("no_scans"))




# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(t("footer"))