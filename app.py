"""설명 가능한 심질환 위험 예측 데모 (Gradio).

한 환자의 임상 지표를 입력하면
(1) 위험 확률과 (2) '왜 그렇게 판단했는지'를 지표별 기여도로 보여준다.
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

# 모델 없으면 즉석 학습
if not MODEL_PATH.exists():
    train_and_save()
_model, _columns = load_model()
_explainer = Explainer(_model, _columns)

# 예시 환자 (전형적 고위험)
EXAMPLE = [63, 1, 4, 145, 233, 0, 2, 150, 0, 2.3, 3, 0, 6]


def predict(age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal):
    values = [age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]
    x = pd.DataFrame([values], columns=FEATURE_COLUMNS).astype(float)
    result = _explainer.explain_one(x, top_k=6)

    proba = result["risk_probability"]
    verdict = "높음 ⚠️" if proba >= 0.5 else "낮음 ✅"
    summary = f"## 심질환 위험: **{proba*100:.0f}%** ({verdict})\n\n### 판단 근거 (기여 순)\n"
    for r in result["top_factors"]:
        bar = "█" * min(10, int(abs(r["impact"]) * 40) + 1)
        summary += f"- **{r['label']}** → {r['direction']}  `{bar}`\n"
    summary += (
        "\n> 위험 ↑ = 이 지표가 위험을 높이는 방향으로 작용, "
        "위험 ↓ = 낮추는 방향.\n"
        "> 연구·교육용 데모이며 진단 도구가 아닙니다."
    )
    return summary


def _num_choices(feature):
    return [(f"{v} — {lab}", v) for v, lab in CATEGORY_LABELS[feature].items()]


with gr.Blocks(title="설명 가능한 심질환 위험 예측") as demo:
    gr.Markdown(
        "# 🫀 설명 가능한 심질환 위험 예측\n"
        "임상 지표를 입력하면 위험 확률과 **그 근거**를 지표별로 보여줍니다. "
        "정확도가 아니라 *설명*이 목적입니다. (UCI Heart Disease, 공개 데이터)"
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
    btn = gr.Button("위험도 예측 + 설명", variant="primary")
    out = gr.Markdown()
    btn.click(
        predict,
        [age, sex, cp, trestbps, chol, fbs, restecg,
         thalach, exang, oldpeak, slope, ca, thal],
        out,
    )

if __name__ == "__main__":
    demo.launch()
