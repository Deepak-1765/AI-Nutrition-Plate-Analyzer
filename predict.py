"""
predict.py
----------
Core prediction logic for the AI Nutrition Plate Analyzer.

This module is intentionally kept separate from app.py so that the
prediction + nutrition-lookup logic can be reused (e.g. in a script,
a notebook, or a different UI) without pulling in Gradio.

Responsibilities:
    1. Load the trained Keras model + class names once.
    2. Preprocess an uploaded image the same way it was preprocessed
       during training (resize to 224x224, MobileNetV2 preprocessing).
    3. Run inference and return the predicted class + confidence.
    4. Look up nutrition facts for the predicted class from nutrition.csv.
    5. Generate a simple, rule-based diet recommendation and health score
       explanation (on top of the static HealthScore stored in the CSV).
"""

import json
import os

import numpy as np
import pandas as pd
from PIL import Image

# TensorFlow is imported lazily inside load_model() so that this module
# can still be imported (e.g. for unit-testing the nutrition lookup logic)
# in environments where TensorFlow isn't installed.

MODEL_PATH = "food_model.keras"
CLASS_NAMES_PATH = "class_names.json"
NUTRITION_CSV_PATH = "nutrition.csv"
IMG_SIZE = (224, 224)


class FoodPredictor:
    """Wraps the trained model, class list, and nutrition table."""

    def __init__(self,
                 model_path: str = MODEL_PATH,
                 class_names_path: str = CLASS_NAMES_PATH,
                 nutrition_csv_path: str = NUTRITION_CSV_PATH):
        import tensorflow as tf  # local import, see note above

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                f"Run train_model.py first to generate it."
            )
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names file not found at '{class_names_path}'.")
        if not os.path.exists(nutrition_csv_path):
            raise FileNotFoundError(f"Nutrition CSV not found at '{nutrition_csv_path}'.")

        self.model = tf.keras.models.load_model(model_path)

        with open(class_names_path) as f:
            self.class_names = json.load(f)

        self.nutrition_df = pd.read_csv(nutrition_csv_path)
        # Normalize the lookup key so "apple_pie" / "Apple Pie" both work
        self.nutrition_df["_lookup_key"] = (
            self.nutrition_df["Food"].str.strip().str.lower().str.replace(" ", "_")
        )

        # Keep the preprocess_input function matched to training
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        self._preprocess_input = preprocess_input

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """Resizes + preprocesses a PIL image into a model-ready batch of 1."""
        image = image.convert("RGB").resize(IMG_SIZE)
        arr = np.array(image, dtype=np.float32)
        arr = self._preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)  # add batch dimension
        return arr

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, image: Image.Image, top_k: int = 3):
        """
        Runs the model on a single image.

        Returns:
            predicted_label (str), confidence (float 0-1),
            top_k_predictions (list of (label, confidence) tuples)
        """
        batch = self._prepare_image(image)
        probs = self.model.predict(batch, verbose=0)[0]

        top_indices = probs.argsort()[::-1][:top_k]
        top_k_predictions = [(self.class_names[i], float(probs[i])) for i in top_indices]

        predicted_label = top_k_predictions[0][0]
        confidence = top_k_predictions[0][1]

        return predicted_label, confidence, top_k_predictions

    # ------------------------------------------------------------------
    # Nutrition lookup
    # ------------------------------------------------------------------
    def get_nutrition(self, food_label: str) -> dict:
        """Looks up nutrition info for a predicted food label."""
        key = food_label.strip().lower().replace(" ", "_")
        row = self.nutrition_df[self.nutrition_df["_lookup_key"] == key]

        if row.empty:
            return None

        row = row.iloc[0]
        return {
            "Food": row["Food"],
            "Calories": row["Calories"],
            "Protein": row["Protein"],
            "Carbohydrates": row["Carbohydrates"],
            "Fat": row["Fat"],
            "Fiber": row["Fiber"],
            "Serving": row["Serving"],
            "HealthScore": row["HealthScore"],
            "Recommendation": row["Recommendation"],
        }


# -------------------------------------------------------------------------
# Rule-based recommendation helper
# -------------------------------------------------------------------------
def generate_recommendation(nutrition: dict) -> str:
    """
    Generates a short, human-readable diet suggestion using simple
    rule-based logic on top of the macro data. This purposely does NOT
    call any AI/LLM API -- it's plain Python conditionals, as required.

    The CSV already stores a curated `Recommendation` per food, but this
    function shows how you could derive additional, dynamic tips from
    the raw macros (e.g. if you add new foods without writing a custom
    recommendation by hand).
    """
    tips = []

    protein = nutrition["Protein"]
    fat = nutrition["Fat"]
    carbs = nutrition["Carbohydrates"]
    fiber = nutrition["Fiber"]
    calories = nutrition["Calories"]

    if protein >= 20:
        tips.append("High in protein.")
    elif protein <= 5:
        tips.append("Low in protein — consider pairing with a protein-rich side.")

    if fat >= 25:
        tips.append("High in fat — enjoy in moderation.")
    elif fat <= 5:
        tips.append("Low in fat.")
    else:
        tips.append("Moderate fat content.")

    if fiber >= 5:
        tips.append("Good source of dietary fiber.")
    elif fiber <= 1:
        tips.append("Low in fiber — add a side of vegetables or salad.")

    if calories >= 450:
        tips.append("Calorie-dense — best as a shared or occasional meal.")

    if carbs >= 50:
        tips.append("High in carbohydrates — best balanced with physical activity.")

    return " ".join(tips)


def health_score_label(score: float) -> str:
    """Maps a numeric health score (0-10) to a short qualitative label."""
    if score >= 8:
        return "Excellent choice"
    elif score >= 6:
        return "Healthy"
    elif score >= 4:
        return "Moderate — enjoy occasionally"
    else:
        return "Indulgent — treat food"
