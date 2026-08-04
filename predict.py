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
    6. Optionally combine that with the user's BMI (from age/height/weight)
       to produce a personalized, rule-based note about whether this meal
       fits their weight goals. This is plain Python conditional logic --
       no AI/LLM API calls, and no medical diagnosis.
"""

import json
import os

import numpy as np
import pandas as pd
from PIL import Image

# TensorFlow is imported lazily inside FoodPredictor.__init__ so that this
# module can still be imported (e.g. for unit-testing the nutrition/BMI
# logic) in environments where TensorFlow isn't installed.

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
        # Normalize the lookup key so "aloo_gobi" / "Aloo Gobi" both work
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
# Rule-based recommendation helper (food-only, no profile needed)
# -------------------------------------------------------------------------
def generate_recommendation(nutrition: dict) -> str:
    """
    Generates a short, human-readable diet suggestion using simple
    rule-based logic on top of the macro data. This purposely does NOT
    call any AI/LLM API -- it's plain Python conditionals, as required.
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


def health_score_color(score: float) -> str:
    """Maps a health score to a hex color for UI badges."""
    if score >= 8:
        return "#16a34a"   # green
    elif score >= 6:
        return "#65a30d"   # lime green
    elif score >= 4:
        return "#f59e0b"   # amber
    else:
        return "#dc2626"   # red


# -------------------------------------------------------------------------
# BMI + personalized, weight-goal-aware recommendation
# -------------------------------------------------------------------------
def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Standard BMI formula: weight(kg) / height(m)^2."""
    if weight_kg is None or height_cm is None or height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive numbers.")
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def bmi_category(bmi: float) -> str:
    """Standard WHO adult BMI categories."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def bmi_color(category: str) -> str:
    """Maps a BMI category to a hex color for UI badges."""
    return {
        "Underweight": "#2563eb",     # blue
        "Normal weight": "#16a34a",   # green
        "Overweight": "#f59e0b",      # amber
        "Obese": "#dc2626",           # red
    }.get(category, "#6b7280")


def generate_personalized_advice(bmi_cat: str, nutrition: dict, age: float = None) -> str:
    """
    Combines the user's BMI category with the predicted meal's macros to
    produce a short, rule-based note about how well this meal fits a
    healthy-weight goal. Plain Python conditionals only -- this is a
    simplified educational heuristic, not medical or dietary advice.
    """
    calories = nutrition["Calories"]
    protein = nutrition["Protein"]
    fat = nutrition["Fat"]
    fiber = nutrition["Fiber"]

    tips = []

    if bmi_cat == "Underweight":
        if calories >= 350:
            tips.append(
                "This is a good calorie-dense choice that can support healthy weight gain."
            )
        else:
            tips.append(
                "This meal is relatively light for a weight-gain goal — consider pairing it "
                "with a calorie-dense side (nuts, dairy, whole grains)."
            )
        if protein < 15:
            tips.append("Adding a protein-rich side would also support healthy muscle gain.")

    elif bmi_cat == "Normal weight":
        tips.append("This fits well within a balanced diet for maintaining your current weight.")
        if fat >= 25 or calories >= 450:
            tips.append("Keep the portion moderate to stay comfortably in your healthy range.")

    elif bmi_cat == "Overweight":
        if calories >= 400 or fat >= 20:
            tips.append(
                "This meal is fairly calorie-dense — a smaller portion or a side salad "
                "would help support gradual, healthy weight management."
            )
        else:
            tips.append("A reasonably light choice that fits well with a weight-management goal.")

    elif bmi_cat == "Obese":
        if calories >= 350 or fat >= 18:
            tips.append(
                "This meal is calorie- and fat-dense. A smaller portion, less oil, and extra "
                "vegetables would better support your health goals."
            )
        else:
            tips.append("This is a lighter option that aligns well with a weight-management plan.")

    if age is not None and age >= 50 and fiber < 3:
        tips.append("Fiber needs often increase with age — a fiber-rich side would help.")

    return " ".join(tips)
