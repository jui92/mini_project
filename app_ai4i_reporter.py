# app_ai4i_reporter.py
import os
import io
import time
import json
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from typing import Tuple, List
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# -------------------------
# 0) 기본 설정
# -------------------------
st.set_page_config(page_title="설비 로그 → 진단 보고서 자동생성기", page_icon="🛠️", layout="wide")
st.title("🛠️ 설비 로그 → 진단 보고서 자동생성기 (AI4I 2020 기반)")

AI4I_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
except Exception:
    openai_client = None

MODEL_PATH = "artifacts/ai4i_rf.pkl"
STATS_PATH = "artifacts/ai4i_stats.json"
os.makedirs("artifacts", exist_ok=True)

TARGET_LABELS = ["Normal", "TWF", "HDF", "PWF", "OSF", "RNF"]

# -------------------------
# 1) 데이터 로딩/가공
# -------------------------
def download_ai4i() -> pd.DataFrame:
    r = requests.get(AI4I_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df

def derive_failure_type(df: pd.DataFrame) -> pd.DataFrame:
    # AI4I 컬럼 가정: ['UDI','Product ID','Type','Air temperature [K]','Process temperature [K]',
    # 'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Machine failure',
    # 'TWF','HDF','PWF','OSF','RNF']
    df = df.copy()
    def pick_failure(row):
        if row.get('Machine failure', 0) == 0:
            return "Normal"
        # 여러 개가 1일 수 있어 우선순위를 정하든, 다중이면 첫 번째로 택1
        for k in ["TWF","HDF","PWF","OSF","RNF"]:
            if k in row and row[k] == 1:
                return k
        return "Normal"
    df["failure_type"] = df.apply(pick_failure, axis=1)
    return df

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # 분류에 사용할 특징 열 선택
    feature_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    X = df[feature_cols].copy()
    y = df["failure_type"].copy()
    return X, y

# -------------------------
# 2) 학습 파이프라인
# -------------------------
def build_pipeline(categorical_cols: List[str], numeric_cols: List[str]) -> Pipeline:
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, categorical_cols),
            ("num", num_tf, numeric_cols)
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[("prep", preproc), ("clf", clf)])
    return pipe

def fit_and_save_model(df: pd.DataFrame):
    X, y = split_features_target(df)
    categorical_cols = ["Type"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = build_pipeline(categorical_cols, numeric_cols)
    pipe.fit(X_train, y_train)

    # 성능 보고
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, labels=TARGET_LABELS, zero_division=0, output_dict=False)
    cm = confusion_matrix(y_test, y_pred, labels=TARGET_LABELS)
    # 통계(분위수) 저장: 보고서 생성 시 기준점으로 사용
    stats = {
        "numeric_cols": numeric_cols,
        "quantiles": {c: {
            "q10": float(df[c].quantile(0.10)),
            "q50": float(df[c].quantile(0.50)),
            "q90": float(df[c].quantile(0.90)),
            "min": float(df[c].min()),
            "max": float(df[c].max())
        } for c in numeric_cols}
    }
    joblib.dump(pipe, MODEL_PATH)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return pipe, report, cm, stats

def load_model_and_stats():
    if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
        pipe = joblib.load(MODEL_PATH)
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return pipe, stats
    return None, None

# -------------------------
# 3) 보고서 생성기 (템플릿 / LLM 선택적)
# -------------------------
@dataclass
class PredictionResult:
    label: str
    proba: float
    probas: dict

def predict_row(pipe: Pipeline, row: dict) -> PredictionResult:
    df = pd.DataFrame([row])
    probs = pipe.predict_proba(df)[0]
    classes = list(pipe.classes_)
    idx = int(np.argmax(probs))
    return PredictionResult(
        label=classes[idx],
        proba=float(probs[idx]),
        probas={cls: float(p) for cls, p in zip(classes, probs)}
    )

def feature_flags(row: dict, stats: dict) -> List[str]:
    """ 센서값이 분위수 기준으로 상/하위 구간에 있는지 플래그 생성 """
    flags = []
    q = stats["quantiles"]
    for c in stats["numeric_cols"]:
        v = row.get(c, None)
        if v is None: 
            continue
        q10, q90 = q[c]["q10"], q[c]["q90"]
        if v <= q10:
            flags.append(f"{c} 낮음(≤p10)")
        elif v >= q90:
            flags.append(f"{c} 높음(≥p90)")
    return flags

def template_report(row: dict, pred: PredictionResult, stats: dict) -> str:
    flags = feature_flags(row, stats)
    risk = "낮음"
    if pred.label != "Normal":
        if pred.proba >= 0.8: risk = "매우 높음"
        elif pred.proba >= 0.6: risk = "높음"
        else: risk = "중간"
    bullets = "\n".join([f"- {f}" for f in flags]) or "- 특이치 없음"
    rec = {
        "TWF": "공구 마모 점검 및 교체 주기 조정 권장.",
        "HDF": "냉각·방열계통 점검(팬/열교환기, 윤활/냉각유 상태 확인).",
        "PWF": "전원 품질/전류 피크 점검, 전력계통 로깅 분석.",
        "OSF": "기계적 과부하/진동 상태 모니터링, 하중 조건 재설정.",
        "RNF": "불규칙 고장 패턴. 최근 작업조건/정비이력 교차점검.",
        "Normal": "즉각 조치 불필요. 추세 모니터링 유지."
    }[pred.label]
    return f"""### 진단 요약
- 예측 상태: **{pred.label}** (신뢰 {pred.proba:.2f})
- 위험도: **{risk}**

### 센서 특이치
{bullets}

### 권장 조치
- {rec}

### 확률 분포
{json.dumps(pred.probas, ensure_ascii=False, indent=2)}
"""

def llm_report(row: dict, pred: PredictionResult, stats: dict) -> str:
    if openai_client is None:
        return template_report(row, pred, stats)
    sys = ("너는 산업 설비 진단 보고서 작성 전문가다. 간결한 한국어로 핵심만 정리하라. "
           "과도한 단정 금지. 숫자와 추정은 조심스럽게.")
    user = f"""
[센서 입력]
{json.dumps(row, ensure_ascii=False, indent=2)}

[모델 예측]
라벨: {pred.label}, 신뢰: {pred.proba:.2f}
분포: {json.dumps(pred.probas, ensure_ascii=False)}

[데이터 분위수 기준]
{json.dumps(stats['quantiles'], ensure_ascii=False)[:1500]} ... (생략)

[요구사항]
1) "진단 요약 / 센서 특이치 / 권장 조치 / 확률 분포" 4단락
2) 센서 특이치는 분위수 기준(상/하위 10%)만 간단히 언급
3) 과도한 확정 표현 금지, 안전 문구 포함
"""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":user}
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        return template_report(row, pred, stats)

# -------------------------
# 4) UI
# -------------------------
tab_train, tab_infer, tab_batch = st.tabs(["① 학습/평가", "② 단건 진단", "③ 배치 진단"])

with tab_train:
    st.subheader("AI4I 2020 데이터 준비")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("UCI에서 AI4I 데이터 내려받기"):
            with st.spinner("다운로드 중..."):
                df = download_ai4i()
                st.session_state["ai4i_raw"] = df
            st.success(f"다운로드 완료! rows={len(df):,}")
            st.dataframe(df.head())

    with colB:
        uploaded = st.file_uploader("또는 CSV 업로드(컬럼명 AI4I 포맷)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state["ai4i_raw"] = df
            st.success(f"업로드 완료! rows={len(df):,}")
            st.dataframe(df.head())

    st.markdown("---")
    if st.button("데이터 가공 → 학습/저장"):
        if "ai4i_raw" not in st.session_state:
            st.warning("먼저 데이터셋을 준비하세요.")
        else:
            with st.spinner("가공/학습 중..."):
                raw = st.session_state["ai4i_raw"]
                df = derive_failure_type(raw)
                pipe, report, cm, stats = fit_and_save_model(df)
            st.success("학습 완료 & 모델 저장!")
            st.code(report)
            st.write("혼동행렬(label 순서):", TARGET_LABELS)
            st.dataframe(pd.DataFrame(cm, index=TARGET_LABELS, columns=TARGET_LABELS))

with tab_infer:
    st.subheader("단건 진단")
    pipe, stats = load_model_and_stats()
    if pipe is None:
        st.info("먼저 ① 탭에서 모델을 학습/저장하세요.")
    else:
        # 입력 위젯 (범위는 학습 데이터 분위수에 맞춰 설정)
        q = stats["quantiles"]
        type_opt = st.selectbox("Type(가상 제품군)", ["L","M","H"])
        air = st.number_input("Air temperature [K]", value=float(q["Air temperature [K]"]["q50"]))
        proc = st.number_input("Process temperature [K]", value=float(q["Process temperature [K]"]["q50"]))
        rpm = st.number_input("Rotational speed [rpm]", value=float(q["Rotational speed [rpm]"]["q50"]))
        torque = st.number_input("Torque [Nm]", value=float(q["Torque [Nm]"]["q50"]))
        wear = st.number_input("Tool wear [min]", value=float(q["Tool wear [min]"]["q50"]))

        row = {
            "Type": type_opt,
            "Air temperature [K]": air,
            "Process temperature [K]": proc,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear
        }

        if st.button("진단 보고서 생성", type="primary"):
            with st.spinner("추론/보고서 생성 중..."):
                pred = predict_row(pipe, row)
                report_txt = llm_report(row, pred, stats)
            st.markdown(report_txt)
            st.download_button("보고서 .txt 저장", data=report_txt.encode("utf-8"),
                               file_name="diagnosis_report.txt", mime="text/plain")

with tab_batch:
    st.subheader("배치 진단 (CSV 업로드)")
    pipe, stats = load_model_and_stats()
    if pipe is None:
        st.info("먼저 ① 탭에서 모델을 학습/저장하세요.")
    else:
        up = st.file_uploader("센서 로그 CSV 업로드 (열: Type, Air..., Process..., Rotational..., Torque..., Tool wear...)", type=["csv"], key="batch")
        if up is not None:
            df_in = pd.read_csv(up)
            st.write(f"입력 샘플 (rows={len(df_in):,})")
            st.dataframe(df_in.head())

            if st.button("일괄 진단 실행"):
                with st.spinner("배치 추론 중..."):
                    preds = pipe.predict_proba(df_in)
                    classes = list(pipe.classes_)
                    labels = np.array(classes)[preds.argmax(1)]
                    confid = preds.max(1)
                    out = df_in.copy()
                    out["pred_label"] = labels
                    out["pred_proba"] = confid

                    # 간단한 템플릿 보고서도 같이 컬럼으로 생성(짧게)
                    short_reports = []
                    for i, row in out.iterrows():
                        rdict = {
                            "Type": row["Type"],
                            "Air temperature [K]": row["Air temperature [K]"],
                            "Process temperature [K]": row["Process temperature [K]"],
                            "Rotational speed [rpm]": row["Rotational speed [rpm]"],
                            "Torque [Nm]": row["Torque [Nm]"],
                            "Tool wear [min]": row["Tool wear [min]"],
                        }
                        pr = PredictionResult(row["pred_label"], row["pred_proba"],
                                              {c: float(p) for c, p in zip(classes, preds[i])})
                        short_reports.append(template_report(rdict, pr, stats).replace("\n", " ")[:500])
                    out["short_report"] = short_reports

                st.success("완료!")
                st.dataframe(out.head())
                # 다운로드
                buffer = io.StringIO()
                out.to_csv(buffer, index=False)
                st.download_button("결과 CSV 다운로드", buffer.getvalue().encode("utf-8"),
                                   file_name="batch_diagnosis.csv", mime="text/csv")

st.caption("※ 본 도구는 참고용입니다. 실제 설비 의사결정은 전문가 점검과 병행하세요.")
