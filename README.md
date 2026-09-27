# 🫀 Explainable Heart Disease Risk

Heart disease risk prediction from clinical features, with per-patient SHAP attributions reported back in clinical terms.

The demo predicts heart disease risk from clinical indicators and explains **why the model decided so** by translating per-patient SHAP contributions into "clinician language". The goal is **explainability**, not an accuracy race.

> ⚠️ Research and education demo. Not a real diagnostic or clinical tool.

## Why this project

Black-box predictions are hard to trust in the clinic. For a single patient's prediction, this demo decomposes **how much each clinical indicator raised (↑) or lowered (↓)** the risk, so a clinician can review the evidence.

## Performance

| Metric | Value |
|---|---|
| 5-fold cross-validated AUC | **0.907** |
| 5-fold cross-validated accuracy | **0.825** |

RandomForest + SHAP TreeExplainer. UCI Heart Disease (Cleveland, 303 records, public data).

## Repository layout

```
src/features.py   clinical feature metadata (mapping to clinician language)
src/data.py       UCI data loading/caching, missing-value imputation, target binarization
src/train.py      model training, cross-validation, saving
src/explain.py    SHAP-based per-patient explanation layer
app.py            Gradio demo (risk probability + evidence)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train      # train the model
python app.py            # launch the demo
```

## Data source

UCI Machine Learning Repository, Heart Disease (Cleveland).
https://archive.ics.uci.edu/dataset/45/heart+disease

## Citation / DOI

Releases are archived on Zenodo, which issues a DOI (badge to be added once issued). Citation metadata: `CITATION.cff`.

## License

MIT

**Status:** research/education demo.

More projects: https://github.com/lshpy
