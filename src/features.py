"""임상 지표(feature) 메타데이터.

각 지표를 '의사 언어'로 설명하기 위한 이름/단위/범주 정의.
XAI 데모의 핵심: 모델이 무엇을 근거로 판단했는지를 임상 용어로 되돌려준다.
"""

# UCI Heart Disease (Cleveland) 13개 입력 지표
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "num"  # 0 = 질환 없음, 1~4 = 질환 있음 → 이진화

# 사람이 읽는 이름
DISPLAY_NAME = {
    "age": "나이",
    "sex": "성별",
    "cp": "흉통 유형",
    "trestbps": "안정시 혈압",
    "chol": "혈청 콜레스테롤",
    "fbs": "공복 혈당 >120",
    "restecg": "안정시 심전도",
    "thalach": "최대 심박수",
    "exang": "운동 유발 협심증",
    "oldpeak": "ST 분절 하강",
    "slope": "운동 ST 기울기",
    "ca": "주요 혈관 수(조영)",
    "thal": "지중해빈혈 검사",
}

UNIT = {
    "age": "세",
    "trestbps": "mmHg",
    "chol": "mg/dL",
    "thalach": "회/분",
    "oldpeak": "",
}

# 범주형 값 → 임상 의미
CATEGORY_LABELS = {
    "sex": {0: "여성", 1: "남성"},
    "cp": {1: "전형적 협심증", 2: "비전형 협심증", 3: "비협심증 통증", 4: "무증상"},
    "fbs": {0: "정상", 1: "높음(>120mg/dL)"},
    "restecg": {0: "정상", 1: "ST-T 이상", 2: "좌심실 비대"},
    "exang": {0: "없음", 1: "있음"},
    "slope": {1: "상승", 2: "평탄", 3: "하강"},
    "thal": {3: "정상", 6: "고정 결손", 7: "가역 결손"},
}


def humanize(feature: str, value) -> str:
    """지표 값을 임상 표현으로 변환. 예: cp=4 → '흉통 유형: 무증상'."""
    name = DISPLAY_NAME.get(feature, feature)
    if feature in CATEGORY_LABELS:
        label = CATEGORY_LABELS[feature].get(int(round(value)), str(value))
        return f"{name}: {label}"
    unit = UNIT.get(feature, "")
    v = round(float(value), 1)
    return f"{name}: {v}{unit}".rstrip()
