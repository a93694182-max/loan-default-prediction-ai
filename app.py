import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="신용대출 연체 위험 예측",
    page_icon="🏦",
    layout="wide"
)

model = joblib.load("loan_model.pkl")

st.title("🏦 신용대출 연체 위험 예측 AI")
st.write("고객 정보를 입력하면 연체 위험 여부를 예측합니다.")

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "연소득",
        value=50000000
    )

    credit = st.number_input(
        "대출금액",
        value=20000000
    )

    annuity = st.number_input(
        "연간 상환금",
        value=3000000
    )

    goods_price = st.number_input(
        "상품 금액",
        value=20000000
    )

with col2:

    children = st.number_input(
        "자녀 수",
        value=0
    )

    family_members = st.number_input(
        "가족 수",
        value=2
    )

    age = st.number_input(
        "나이",
        value=35
    )

    employment_years = st.number_input(
        "근속 연수",
        value=5
    )

if st.button("🚀 연체 위험 예측하기", use_container_width=True):

    input_data = pd.DataFrame([{
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": family_members,
        "DAYS_BIRTH": -age * 365,
        "DAYS_EMPLOYED": -employment_years * 365,
        "DAYS_REGISTRATION": -3000,
        "DAYS_ID_PUBLISH": -2000,
        "REGION_POPULATION_RELATIVE": 0.02,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "OBS_30_CNT_SOCIAL_CIRCLE": 1,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 0,
        "AMT_REQ_CREDIT_BUREAU_MON": 0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.7:
        risk_level = "높음"
        result_box = st.error
        result_icon = "🔴"
        result_message = "연체 위험이 높은 고객으로 판단됩니다."
    elif probability >= 0.4:
        risk_level = "보통"
        result_box = st.warning
        result_icon = "🟡"
        result_message = "연체 가능성이 일부 존재하므로 추가 심사가 필요합니다."
    else:
        risk_level = "낮음"
        result_box = st.success
        result_icon = "🟢"
        result_message = "정상 상환 가능성이 높은 고객으로 판단됩니다."

    st.divider()
    st.markdown("## 📊 예측 결과")

    st.metric("연체 위험 확률", f"{probability * 100:.2f}%")
    st.progress(float(probability))

    result_box(f"{result_icon} 위험 등급 : {risk_level}")

    st.info(f"""
    ### 📋 AI 분석 요약

    - 예측 결과: **{"연체 위험" if prediction == 1 else "정상 고객"}**
    - 연체 위험 확률: **{probability * 100:.2f}%**
    - 위험 등급: **{risk_level}**
    - 사용 모델: **XGBoost**
    """)

    st.markdown(f"""
    ### 💡 판단 의견

    {result_message}
    """)


