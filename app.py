"""
app.py
------
Gradio web application for the AI Nutrition Plate Analyzer.

Flow:
    1. User uploads a food image.
    2. The trained CNN (predict.py -> FoodPredictor) classifies the food.
    3. Nutrition facts are looked up from nutrition.csv.
    4. A rule-based health score + diet recommendation is displayed.

Run locally:
    python app.py

Deploy on Render:
    See README.md for the Render deployment steps. This file exposes
    `demo` at module level and launches it with a `0.0.0.0` server name
    and the PORT environment variable, which is what Render (and most
    PaaS providers) expect.
"""

import os

import gradio as gr
from PIL import Image

from predict import FoodPredictor, generate_recommendation, health_score_label

# ---------------------------------------------------------------------------
# Load the model + nutrition table ONCE at startup (not per-request)
# ---------------------------------------------------------------------------
try:
    predictor = FoodPredictor()
    MODEL_LOAD_ERROR = None
except Exception as e:  # pragma: no cover - defensive startup guard
    predictor = None
    MODEL_LOAD_ERROR = str(e)


def analyze_food(image: Image.Image):
    """
    Main callback wired to the Gradio interface.

    Returns a tuple matching the output components:
    (prediction_markdown, confidence, calories, protein, carbs,
     fat, fiber, serving, health_score, recommendation, top_k_markdown)
    """
    empty = ("—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "")

    # -- Guard: model failed to load at startup ---------------------------
    if predictor is None:
        error_msg = (
            f"⚠️ Model could not be loaded: {MODEL_LOAD_ERROR}\n\n"
            f"Make sure `food_model.keras` and `class_names.json` exist "
            f"in the project folder (run `train_model.py` first)."
        )
        return (error_msg,) + empty[1:]

    # -- Guard: no image uploaded ------------------------------------------
    if image is None:
        return ("Please upload a food image to get started.",) + empty[1:]

    # -- Guard: invalid / corrupted image -----------------------------------
    try:
        image = image.convert("RGB")
    except Exception:
        return ("⚠️ Could not read this image. Please upload a valid JPG/PNG file.",) + empty[1:]

    # -- Run prediction -------------------------------------------------------
    try:
        predicted_label, confidence, top_k = predictor.predict(image, top_k=3)
    except Exception as e:
        return (f"⚠️ Prediction failed: {e}",) + empty[1:]

    # -- Low-confidence guard: warn the user the result may be unreliable ----
    confidence_pct = confidence * 100
    low_confidence_note = ""
    if confidence < 0.4:
        low_confidence_note = (
            "\n\n_Confidence is low — the image may not clearly match one of "
            "the 60 supported food classes. Try a clearer, closer photo._"
        )

    # -- Nutrition lookup -------------------------------------------------------
    nutrition = predictor.get_nutrition(predicted_label)
    if nutrition is None:
        msg = (
            f"Predicted **{predicted_label.replace('_', ' ').title()}**, but no "
            f"nutrition data was found for this class in nutrition.csv."
        )
        return (msg,) + empty[1:]

    # -- Build rule-based recommendation (dynamic) + curated one from CSV ------
    dynamic_tip = generate_recommendation(nutrition)
    curated_tip = nutrition["Recommendation"]
    score_label = health_score_label(nutrition["HealthScore"])

    recommendation_text = (
        f"{curated_tip}\n\n**Quick analysis:** {dynamic_tip} "
        f"**Overall: {score_label}.**"
    )

    prediction_display = f"### 🍽️ {nutrition['Food'].replace('_', ' ').title()}{low_confidence_note}"

    # -- Top-3 predictions table (nice touch for transparency) -----------------
    top_k_lines = ["**Top predictions:**"]
    for label, prob in top_k:
        top_k_lines.append(f"- {label.replace('_', ' ').title()}: {prob * 100:.1f}%")
    top_k_markdown = "\n".join(top_k_lines)

    return (
        prediction_display,
        f"{confidence_pct:.1f}%",
        f"{nutrition['Calories']} kcal",
        f"{nutrition['Protein']} g",
        f"{nutrition['Carbohydrates']} g",
        f"{nutrition['Fat']} g",
        f"{nutrition['Fiber']} g",
        f"{nutrition['Serving']}",
        f"{nutrition['HealthScore']} / 10",
        recommendation_text,
        top_k_markdown,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
#title { text-align: center; margin-bottom: 0.25em; }
#subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
.gradio-container { max-width: 1000px !important; margin: auto; }
"""

with gr.Blocks(title="AI Nutrition Plate Analyzer", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🥗 AI Nutrition Plate Analyzer", elem_id="title")
    gr.Markdown(
        "Upload a photo of a food item to identify it and get instant "
        "nutrition facts, a health score, and a simple diet recommendation.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Food Image", height=320)
            analyze_btn = gr.Button("Analyze Food", variant="primary")
            gr.Markdown(
                "*Supported: 12 Indian dishes — Aloo Gobi, Biryani, Butter Chicken, "
                "Chana Masala, Chapati, Dal Makhani, Dal Tadka, Gulab Jamun, Jalebi, "
                "Kadai Paneer, Naan, and Poha.*"
            )

        with gr.Column(scale=1):
            prediction_output = gr.Markdown(label="Prediction")
            confidence_output = gr.Textbox(label="Confidence Score", interactive=False)

            with gr.Row():
                calories_output = gr.Textbox(label="Calories", interactive=False)
                protein_output = gr.Textbox(label="Protein", interactive=False)

            with gr.Row():
                carbs_output = gr.Textbox(label="Carbohydrates", interactive=False)
                fat_output = gr.Textbox(label="Fat", interactive=False)

            with gr.Row():
                fiber_output = gr.Textbox(label="Fiber", interactive=False)
                serving_output = gr.Textbox(label="Serving Size", interactive=False)

            health_score_output = gr.Textbox(label="Health Score", interactive=False)
            recommendation_output = gr.Markdown(label="Diet Recommendation")

    with gr.Accordion("See top-3 predictions", open=False):
        top_k_output = gr.Markdown()

    outputs = [
        prediction_output, confidence_output, calories_output, protein_output,
        carbs_output, fat_output, fiber_output, serving_output,
        health_score_output, recommendation_output, top_k_output,
    ]

    analyze_btn.click(fn=analyze_food, inputs=image_input, outputs=outputs)
    # Also auto-run when a new image is uploaded, for convenience
    image_input.change(fn=analyze_food, inputs=image_input, outputs=outputs)

    gr.Markdown(
        "---\n*Educational project — nutrition values are approximate "
        "reference estimates, not medical or dietary advice.*"
    )


if __name__ == "__main__":
    # Render (and most PaaS hosts) inject the port to bind to via $PORT.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
