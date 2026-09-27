"""Explainable heart disease risk prediction demo (Gradio).

Enter one patient's clinical features to see
(1) the risk probability and (2) "why the model decided so" as per-feature contributions.
"""
from __future__ import annotations

import gradio as gr

from src.features import (
    CATEGORY_LABELS,
    DISPLAY_NAME,
    FEATURE_COLUMNS,
)
from src.explain import Explainer
from src.train import MODEL_PATH, load_model, train_and_save

import pandas as pd

# train on the fly if no model exists
if not MODEL_PATH.exists():
    train_and_save()
_model, _columns = load_model()
_explainer = Explainer(_model, _columns)

# example patient (typical high risk)
EXAMPLE = [63, 1, 4, 145, 233, 0, 2, 150, 0, 2.3, 3, 0, 6]


def predict(age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal):
    values = [age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]
    x = pd.DataFrame([values], columns=FEATURE_COLUMNS).astype(float)
    result = _explainer.explain_one(x, top_k=6)

    proba = result["risk_probability"]
    verdict = "high ⚠️" if proba >= 0.5 else "low ✅"
    summary = f"## Heart disease risk: **{proba*100:.0f}%** ({verdict})\n\n### Evidence (by contribution)\n"
    for r in result["top_factors"]:
        bar = "█" * min(10, int(abs(r["impact"]) * 40) + 1)
        summary += f"- **{r['label']}** → {r['direction']}  `{bar}`\n"
    summary += (
        "\n> risk ↑ = this feature pushes the risk up, "
        "risk ↓ = pushes it down.\n"
        "> Research/education demo, not a diagnostic tool."
    )
    return summary


def _num_choices(feature):
    return [(f"{v} — {lab}", v) for v, lab in CATEGORY_LABELS[feature].items()]


with gr.Blocks(title="Explainable heart disease risk prediction") as demo:
    gr.Markdown(
        "# 🫀 Explainable heart disease risk prediction\n"
        "Enter clinical features to see the risk probability and **its evidence** per feature. "
        "The goal is *explanation*, not accuracy. (UCI Heart Disease, public data)"
    )
    with gr.Row():
        with gr.Column():
            age = gr.Slider(20, 90, value=63, step=1, label=DISPLAY_NAME["age"])
            sex = gr.Radio(_num_choices("sex"), value=1, label=DISPLAY_NAME["sex"])
            cp = gr.Radio(_num_choices("cp"), value=4, label=DISPLAY_NAME["cp"])
            trestbps = gr.Slider(80, 200, value=145, step=1, label=DISPLAY_NAME["trestbps"])
            chol = gr.Slider(100, 600, value=233, step=1, label=DISPLAY_NAME["chol"])
            fbs = gr.Radio(_num_choices("fbs"), value=0, label=DISPLAY_NAME["fbs"])
            restecg = gr.Radio(_num_choices("restecg"), value=2, label=DISPLAY_NAME["restecg"])
        with gr.Column():
            thalach = gr.Slider(60, 220, value=150, step=1, label=DISPLAY_NAME["thalach"])
            exang = gr.Radio(_num_choices("exang"), value=0, label=DISPLAY_NAME["exang"])
            oldpeak = gr.Slider(0, 6, value=2.3, step=0.1, label=DISPLAY_NAME["oldpeak"])
            slope = gr.Radio(_num_choices("slope"), value=3, label=DISPLAY_NAME["slope"])
            ca = gr.Slider(0, 3, value=0, step=1, label=DISPLAY_NAME["ca"])
            thal = gr.Radio(_num_choices("thal"), value=6, label=DISPLAY_NAME["thal"])
    btn = gr.Button("Predict risk + explain", variant="primary")
    out = gr.Markdown()
    btn.click(
        predict,
        [age, sex, cp, trestbps, chol, fbs, restecg,
         thalach, exang, oldpeak, slope, ca, thal],
        out,
    )

if __name__ == "__main__":
    demo.launch()
