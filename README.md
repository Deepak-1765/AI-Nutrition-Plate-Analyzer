# 🥗 AI Nutrition Plate Analyzer

An educational, deployment-ready AI/ML internship project: upload a photo of a
food item, and the app identifies it with a CNN (transfer learning on
MobileNetV2) and instantly shows nutrition facts, a health score, and a
simple diet recommendation — all with a clean Gradio web UI.

This version is trained on a **custom Indian food dataset (12 classes)**.

---

## 📋 Overview

| | |
|---|---|
| **Type** | Image classification + rule-based nutrition lookup |
| **Model** | MobileNetV2 (transfer learning), fine-tuned on a custom dataset |
| **Classes** | 12 Indian dishes (see `nutrition.csv`) |
| **UI** | Gradio |
| **Nutrition logic** | Plain Python rules (no AI/LLM API calls) |
| **Deployment** | Render-ready, no code changes needed |

### Supported classes

`Aloo Gobi`, `Biryani`, `Butter Chicken`, `Chana Masala`, `Chapati`,
`Dal Makhani`, `Dal Tadka`, `Gulab Jamun`, `Jalebi`, `Kadai Paneer`,
`Naan`, `Poha`

---

## ✨ Features

- 📷 Upload any food photo (JPG/PNG) through a clean, card-based Gradio UI
- 🧠 CNN-based food classification (12 classes) with a confidence score
- 🔍 Top-3 prediction breakdown for transparency
- 🍎 Automatic nutrition facts lookup: Calories, Protein, Carbohydrates, Fat, Fiber, Serving size
- 💯 0–10 Health Score for each food, color-coded
- 📝 Rule-based diet recommendation (combines a curated tip with dynamic macro analysis)
- 👤 **Optional profile (age, height, weight)** — computes BMI locally and
  shows a personalized, rule-based note on how well *this specific meal*
  fits a weight-gain / maintenance / weight-management goal
- 🛡️ Graceful handling of invalid uploads / missing images / low-confidence predictions
- ⚙️ Modular codebase: model training, inference, and UI are cleanly separated

**Example output**

```
Prediction:        Butter Chicken
Confidence:        92.3%
Calories:           490 kcal
Protein:            28 g
Carbohydrates:      12 g
Fat:                35 g
Fiber:              2 g
Serving:            250 g
Health Score:       4.5 / 10
Recommendation:     High protein but rich in cream and butter. Best paired
                    with a whole wheat roti and salad, in moderation.

Your Profile:       BMI 29.4 (Overweight)
For this meal:      This meal is fairly calorie-dense — a smaller portion
                    or a side salad would help support gradual, healthy
                    weight management.
```

> The BMI feature is entirely optional, computed locally with the
> standard `weight(kg) / height(m)²` formula, and nothing is stored —
> it's plain rule-based Python, not a diagnosis. See the Disclaimer below.

---

## 🗂️ Project Structure

```
ai-nutrition-plate-analyzer/
├── app.py                  # Gradio web application (entry point)
├── train_model.py          # Model training script (transfer learning)
├── predict.py               # Prediction + nutrition lookup module (reusable)
├── nutrition.csv           # Nutrition data for all 12 supported food classes
├── class_names.json        # Ordered list of class labels (index -> label)
├── food_model.keras        # Trained CNN weights
├── requirements.txt        # Pinned, compatible dependency versions
└── README.md                # This file
```

---

## 🍽️ Dataset

- **Dataset:** a custom, independently collected and labeled set of
  Indian food photos across **12 classes** (listed above).
- **Model:** MobileNetV2 pretrained on ImageNet, fine-tuned on this
  dataset using the two-phase transfer learning pipeline in
  `train_model.py` (frozen-backbone head training, followed by
  fine-tuning of the top backbone layers at a low learning rate).
- **Nutrition data:** `nutrition.csv` was built manually with
  **realistic, approximate values per standard serving**, based on
  typical nutrition references for each dish. These are educational
  estimates, not medical-grade or lab-measured values — actual
  nutrition varies by recipe, portion, and preparation method.

### `nutrition.csv` columns

| Column | Description |
|---|---|
| `Food` | Class name, matches the folder name used during training (e.g. `butter_chicken`) |
| `Calories` | kcal per serving |
| `Protein` | grams per serving |
| `Carbohydrates` | grams per serving |
| `Fat` | grams per serving |
| `Fiber` | grams per serving |
| `Serving` | typical serving size |
| `HealthScore` | 0–10 curated health rating |
| `Recommendation` | short, curated diet tip |

`nutrition.csv` rows are in the same order as `class_names.json`
(verified to match the model's 12-unit softmax output layer) — this
keeps the CSV lookup logic in `predict.py` reliable regardless of row
order, since lookups are done by matching the `Food` name, not by index.

---

## 🛠️ Installation

```bash
git clone <your-repo-url>
cd ai-nutrition-plate-analyzer

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🏋️ Training the Model

`food_model.keras` in this repo was trained on the custom 12-class Indian
food dataset described above. If you'd like to retrain (e.g. after adding
more images, more classes, or tuning hyperparameters):

### 1. Organize your images in the ImageFolder layout

```
dataset/
├── train/<class_name>/*.jpg
└── val/<class_name>/*.jpg
```

Class folder names must exactly match the `Food` values in
`nutrition.csv` (e.g. `aloo_gobi`, `butter_chicken`, `dal_tadka`, ...).

### 2. Train

```bash
python train_model.py --data_dir dataset --epochs 15 --fine_tune_epochs 5
```

This runs a two-phase transfer learning process:
1. **Head training** — MobileNetV2 backbone frozen, only the new
   classification head is trained (fast, stabilizes quickly).
2. **Fine-tuning** — the top layers of MobileNetV2 are unfrozen and
   trained at a very low learning rate for a few epochs to squeeze out
   extra accuracy.

Outputs:
- `food_model.keras` — the trained model
- `class_names.json` — class index → label mapping (regenerated to
  guarantee it matches the model's output layer)
- `training_history.png` — accuracy/loss curves

> If you'd rather pull additional classes from the public Food-101
> dataset instead of hand-collecting more images, `train_model.py`
> still includes an optional `--prepare --food101_dir <path>` step for
> that — see the docstring at the top of the file.

---

## ▶️ Running Locally

```bash
python app.py
```

Then open the printed local URL (default: `http://localhost:7860`) in your
browser, upload a food photo, and click **Analyze Food**.

---

## ☁️ Render Deployment

This project is ready to deploy on [Render](https://render.com) as a **Web
Service** with no code changes.

1. Push this project — including `food_model.keras` and
   `class_names.json` — to a GitHub repository.
2. On Render: **New → Web Service** → connect your repo.
3. Configure:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
4. Deploy. Render sets the `$PORT` environment variable automatically —
   `app.py` already reads it (`os.environ.get("PORT", 7860)`) and binds to
   `0.0.0.0`, so no further configuration is required.

> **Model file size:** `food_model.keras` is ~9.8 MB (MobileNetV2-based),
> well within Render's free-tier repo/deploy limits.

---

## 📸 Screenshots

_Add screenshots of the running app here, e.g.:_

```
screenshots/
├── upload_screen.png
├── prediction_result.png
└── low_confidence_example.png
```

---

## 🚀 Future Improvements

- Add more Indian dishes (or other regional cuisines) by collecting more
  labeled images and adding matching rows to `nutrition.csv`
- Estimate serving size dynamically from the image (e.g. plate-area
  heuristics) instead of a fixed value per class
- Add a Grad-CAM visualization to show which part of the image the model
  focused on
- Track daily meals and calorie totals across multiple uploads (would
  require basic local storage, e.g. a CSV log — no auth/cloud needed)
- Multi-food detection (object detection instead of single-label
  classification) for plates with multiple items
- Model quantization (TFLite) for faster inference on low-resource hosts

---

## ⚠️ Disclaimer

This is an educational project. Nutrition values are approximate
reference estimates and the health score is a simplified heuristic —
this app is **not** a substitute for professional dietary or medical
advice. The BMI feature uses the standard adult BMI formula and WHO
category thresholds for a general-audience estimate; BMI does not
account for muscle mass, body composition, age-specific ranges, or
individual health conditions, and should not be used to self-diagnose
being under/overweight. Consult a doctor or registered dietitian for
personal health guidance.
