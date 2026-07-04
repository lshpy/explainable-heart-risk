# 🫀 Explainable Heart Disease Risk (설명 가능한 심질환 위험 예측)

임상 지표로 심질환 위험을 예측하고, **모델이 왜 그렇게 판단했는지**를
환자별 SHAP 기여도로 '의사 언어'로 되돌려주는 데모.
정확도 경쟁이 아니라 **설명 가능성(explainability)** 이 목적입니다.

> ⚠️ 연구·교육용 데모입니다. 실제 진단·진료 도구가 아닙니다.

## 왜 이 프로젝트인가
블랙박스 예측은 임상에서 신뢰받기 어렵습니다. 이 데모는 한 환자의
예측에 대해 각 임상 지표가 위험을 **얼마나 높였는지(↑)/낮췄는지(↓)**
분해해, 임상의가 근거를 검토할 수 있게 합니다.

## 성능
| 지표 | 값 |
|---|---|
| 5-fold 교차검증 AUC | **0.907** |
| 5-fold 교차검증 정확도 | **0.825** |

RandomForest + SHAP TreeExplainer. UCI Heart Disease (Cleveland, 303건, 공개 데이터).

## 구조
```
src/features.py   임상 지표 메타데이터 (의사 언어 매핑)
src/data.py       UCI 데이터 로드/캐싱, 결측 대치, 타깃 이진화
src/train.py      모델 학습·교차검증·저장
src/explain.py    SHAP 기반 환자별 설명 레이어
app.py            Gradio 데모 (위험 확률 + 근거)
```

## 실행
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train      # 모델 학습
python app.py            # 데모 실행
```

## 데이터 출처
UCI Machine Learning Repository — Heart Disease (Cleveland).
https://archive.ics.uci.edu/dataset/45/heart+disease

## 인용 / DOI
릴리즈는 Zenodo에 아카이빙되어 DOI가 발급됩니다. (발급 후 배지 추가)

## 라이선스
MIT
