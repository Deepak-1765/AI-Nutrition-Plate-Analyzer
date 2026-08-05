"""
app.py
------
Gradio web application for the AI Nutrition Plate Analyzer.

Flow:
    1. (Optional) User enters age, height, and weight once, in a
       "Your Profile" panel — used only to compute BMI locally.
    2. User uploads a food image.
    3. The trained CNN (predict.py -> FoodPredictor) classifies the food.
    4. Nutrition facts are looked up from nutrition.csv.
    5. A rule-based health score + diet recommendation is displayed.
    6. If a profile was entered, a second rule-based note explains how
       well this specific meal fits the user's weight goal (gain /
       maintain / manage), based on their BMI category.

All of the "smart" logic here is plain Python conditionals over CSV data
and a standard BMI formula -- there is no AI/LLM API call anywhere in
this file or in predict.py.

Run locally:
    python app.py

Deploy on Render:
    See README.md. This file exposes `demo` at module level and launches
    it with server_name="0.0.0.0" and the PORT environment variable.
"""

import os

import gradio as gr
from PIL import Image

from predict import (
    FoodPredictor,
    generate_recommendation,
    health_score_label,
    health_score_color,
    calculate_bmi,
    bmi_category,
    bmi_color,
    generate_personalized_advice,
)

# ---------------------------------------------------------------------------
# Load the model + nutrition table ONCE at startup (not per-request)
# ---------------------------------------------------------------------------
try:
    predictor = FoodPredictor()
    MODEL_LOAD_ERROR = None
except Exception as e:  # pragma: no cover - defensive startup guard
    predictor = None
    MODEL_LOAD_ERROR = str(e)


# ---------------------------------------------------------------------------
# Small HTML-building helpers (keep app.py's callback readable)
# ---------------------------------------------------------------------------
def _error_card(message: str) -> str:
    return f"""
    <div class="np-card np-card-error">
        <p>⚠️ {message}</p>
    </div>
    """


def _placeholder_card() -> str:
    return """
    <div class="np-card np-card-placeholder">
        <p>Upload a food photo and click <strong>Analyze Food</strong> to see results here.</p>
    </div>
    """


def _macro_pill(icon: str, label: str, value: str) -> str:
    return f"""
    <div class="np-pill">
        <div class="np-pill-icon">{icon}</div>
        <div class="np-pill-value">{value}</div>
        <div class="np-pill-label">{label}</div>
    </div>
    """


def _build_result_card(nutrition: dict, confidence_pct: float, low_confidence_note: str) -> str:
    score = float(nutrition["HealthScore"])
    color = health_score_color(score)
    label = health_score_label(score)
    dynamic_tip = generate_recommendation(nutrition)
    curated_tip = nutrition["Recommendation"]
    food_title = nutrition["Food"].replace("_", " ").title()

    pills = "".join([
        _macro_pill("🔥", "Calories", f'{nutrition["Calories"]} kcal'),
        _macro_pill("💪", "Protein", f'{nutrition["Protein"]} g'),
        _macro_pill("🌾", "Carbs", f'{nutrition["Carbohydrates"]} g'),
        _macro_pill("🥑", "Fat", f'{nutrition["Fat"]} g'),
        _macro_pill("🌿", "Fiber", f'{nutrition["Fiber"]} g'),
        _macro_pill("🍽️", "Serving", f'{nutrition["Serving"]}'),
    ])

    return f"""
    <div class="np-card">
        <div class="np-card-header">
            <div>
                <div class="np-eyebrow">Prediction</div>
                <div class="np-title">{food_title}</div>
            </div>
            <div class="np-confidence">
                <div class="np-confidence-value">{confidence_pct:.1f}%</div>
                <div class="np-confidence-label">confidence</div>
            </div>
        </div>
        {f'<div class="np-warning">{low_confidence_note}</div>' if low_confidence_note else ''}
        <div class="np-pill-row">{pills}</div>
        <div class="np-score-row">
            <div class="np-score-badge" style="background:{color}">{score:.1f} / 10</div>
            <div class="np-score-label">{label}</div>
        </div>
        <div class="np-tip">
            <strong>Diet tip:</strong> {curated_tip}<br/>
            <span class="np-tip-secondary">{dynamic_tip}</span>
        </div>
    </div>
    """


def _build_bmi_card(bmi_value: float, category: str, advice: str) -> str:
    color = bmi_color(category)
    return f"""
    <div class="np-card np-card-bmi">
        <div class="np-card-header">
            <div>
                <div class="np-eyebrow">Your Profile</div>
                <div class="np-title" style="color:{color}">{category}</div>
            </div>
            <div class="np-confidence">
                <div class="np-confidence-value">{bmi_value:.1f}</div>
                <div class="np-confidence-label">BMI</div>
            </div>
        </div>
        <div class="np-tip"><strong>For this meal:</strong> {advice}</div>
    </div>
    """


def _bmi_placeholder_card() -> str:
    return """
    <div class="np-card np-card-placeholder">
        <p>Enter your age, height, and weight above (optional) to get a
        personalized note on how this meal fits your weight goal.</p>
    </div>
    """


def _bmi_error_card(message: str) -> str:
    return f"""
    <div class="np-card np-card-error">
        <p>⚠️ {message}</p>
    </div>
    """


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------
def analyze_food(image: Image.Image, age, height_cm, weight_kg):
    """
    Returns: (result_card_html, bmi_card_html, top_k_markdown)
    """
    # -- Guard: model failed to load at startup ---------------------------
    if predictor is None:
        msg = (
            f"Model could not be loaded: {MODEL_LOAD_ERROR}. Make sure "
            f"food_model.keras and class_names.json exist in the project folder."
        )
        return _error_card(msg), _bmi_placeholder_card(), ""

    # -- Guard: no image uploaded ------------------------------------------
    if image is None:
        return _placeholder_card(), _bmi_placeholder_card(), ""

    # -- Guard: invalid / corrupted image -----------------------------------
    try:
        image = image.convert("RGB")
    except Exception:
        return (
            _error_card("Could not read this image. Please upload a valid JPG/PNG file."),
            _bmi_placeholder_card(),
            "",
        )

    # -- Run prediction -------------------------------------------------------
    try:
        predicted_label, confidence, top_k = predictor.predict(image, top_k=3)
    except Exception as e:
        return _error_card(f"Prediction failed: {e}"), _bmi_placeholder_card(), ""

    confidence_pct = confidence * 100
    low_confidence_note = ""
    if confidence < 0.4:
        low_confidence_note = (
            "Confidence is low — the image may not clearly match one of the 12 "
            "supported dishes. Try a clearer, closer photo."
        )

    # -- Nutrition lookup -------------------------------------------------------
    nutrition = predictor.get_nutrition(predicted_label)
    if nutrition is None:
        msg = f"Predicted \"{predicted_label.replace('_', ' ').title()}\", but no nutrition data was found."
        return _error_card(msg), _bmi_placeholder_card(), ""

    result_html = _build_result_card(nutrition, confidence_pct, low_confidence_note)

    # -- Top-3 predictions ------------------------------------------------------
    top_k_lines = ["**Top predictions:**"]
    for lbl, prob in top_k:
        top_k_lines.append(f"- {lbl.replace('_', ' ').title()}: {prob * 100:.1f}%")
    top_k_markdown = "\n".join(top_k_lines)

    # -- Optional BMI + personalized advice --------------------------------------
    # All three fields are optional; only compute BMI if all three are provided.
    if age not in (None, "") and height_cm not in (None, "") and weight_kg not in (None, ""):
        try:
            age_val = float(age)
            height_val = float(height_cm)
            weight_val = float(weight_kg)

            if not (1 <= age_val <= 120):
                raise ValueError("Age should be between 1 and 120.")
            if not (50 <= height_val <= 250):
                raise ValueError("Height should be between 50 and 250 cm.")
            if not (10 <= weight_val <= 300):
                raise ValueError("Weight should be between 10 and 300 kg.")

            bmi_value = calculate_bmi(weight_val, height_val)
            category = bmi_category(bmi_value)
            advice = generate_personalized_advice(category, nutrition, age=age_val)
            bmi_html = _build_bmi_card(bmi_value, category, advice)
        except ValueError as e:
            bmi_html = _bmi_error_card(str(e))
    else:
        bmi_html = _bmi_placeholder_card()

    return result_html, bmi_html, top_k_markdown


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --np-radius: 16px;
}
.gradio-container { max-width: 1080px !important; margin: auto; }
#np-title { text-align: center; margin-bottom: 0.1em; font-size: 2.1em; }
#np-subtitle { text-align: center; color: var(--body-text-color-subdued); margin-bottom: 1.6em; }

.np-card {
    border-radius: var(--np-radius);
    padding: 20px 22px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    margin-bottom: 14px;
}
.np-card-placeholder p, .np-card-error p {
    margin: 0; color: var(--body-text-color-subdued); text-align: center;
}
.np-card-error { border-color: #dc2626; }
.np-card-bmi { border-left: 4px solid var(--border-color-primary); }

.np-card-header { display: flex; justify-content: space-between; align-items: flex-start; }
.np-eyebrow {
    text-transform: uppercase; font-size: 0.72em; letter-spacing: 0.06em;
    color: var(--body-text-color-subdued); margin-bottom: 2px;
}
.np-title { font-size: 1.5em; font-weight: 700; }

.np-confidence { text-align: right; }
.np-confidence-value { font-size: 1.3em; font-weight: 700; }
.np-confidence-label {
    font-size: 0.75em; color: var(--body-text-color-subdued); text-transform: uppercase;
}

.np-warning {
    margin-top: 10px; padding: 8px 12px; border-radius: 10px;
    background: rgba(245, 158, 11, 0.15); color: #b45309; font-size: 0.9em;
}

.np-pill-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 16px 0;
}
.np-pill {
    background: var(--background-fill-primary); border-radius: 12px;
    padding: 10px 8px; text-align: center; border: 1px solid var(--border-color-primary);
}
.np-pill-icon { font-size: 1.2em; }
.np-pill-value { font-weight: 700; font-size: 1.02em; margin-top: 2px; }
.np-pill-label {
    font-size: 0.72em; color: var(--body-text-color-subdued); text-transform: uppercase;
}

.np-score-row { display: flex; align-items: center; gap: 10px; margin: 14px 0 10px; }
.np-score-badge {
    color: white; font-weight: 700; padding: 4px 12px; border-radius: 999px; font-size: 0.9em;
}
.np-score-label { font-weight: 600; }

.np-tip { font-size: 0.94em; line-height: 1.5; }
.np-tip-secondary { color: var(--body-text-color-subdued); }
"""

with gr.Blocks(title="AI Nutrition Plate Analyzer", theme=gr.themes.Soft(primary_hue="green"), css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🥗 AI Nutrition Plate Analyzer", elem_id="np-title")
    gr.Markdown(
        "Upload a photo of a food item to identify it and get instant "
        "nutrition facts, a health score, and a personalized diet tip.",
        elem_id="np-subtitle",
    )

    with gr.Row():
        # ------------------------- Left column: inputs -------------------------
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Food Image", height=300)

            with gr.Accordion("👤 Your Profile (optional)", open=False):
                gr.Markdown(
                    "Add these once to get a personalized note on how each meal "
                    "fits your weight goal. Nothing is stored or sent anywhere."
                )
                age_input = gr.Number(label="Age (years)", precision=0, minimum=1, maximum=120)
                height_input = gr.Number(label="Height (cm)", minimum=50, maximum=250)
                weight_input = gr.Number(label="Weight (kg)", minimum=10, maximum=300)

            analyze_btn = gr.Button("🔍 Analyze Food", variant="primary")
            gr.Markdown(
                "*Supported: Aloo Gobi, Biryani, Butter Chicken, Chana Masala, "
                "Chapati, Dal Makhani, Dal Tadka, Gulab Jamun, Jalebi, Kadai "
                "Paneer, Naan, Poha.*"
            )

        # ------------------------- Right column: results -----------------------
        with gr.Column(scale=1):
            result_output = gr.HTML(_placeholder_card())
            bmi_output = gr.HTML(_bmi_placeholder_card())

    with gr.Accordion("See top-3 predictions", open=False):
        top_k_output = gr.Markdown()

    inputs = [image_input, age_input, height_input, weight_input]
    outputs = [result_output, bmi_output, top_k_output]

    analyze_btn.click(fn=analyze_food, inputs=inputs, outputs=outputs)
    # Also auto-run when a new image is uploaded, for convenience
    image_input.change(fn=analyze_food, inputs=inputs, outputs=outputs)

    gr.Markdown(
        "---\n*Educational project — nutrition values and BMI-based tips are "
        "approximate reference estimates, not medical or dietary advice.*"
    )


if __name__ == "__main__":
    # Render (and most PaaS hosts) inject the port to bind to via $PORT.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)






