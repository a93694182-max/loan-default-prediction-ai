import streamlit as st
import pandas as pd
import joblib
import shap

st.set_page_config(
    page_title="신용대출 연체 위험 예측",
    page_icon="🏦",
    layout="wide"
)

model = joblib.load("loan_model.pkl")

classifier = model.named_steps["classifier"]
imputer = model.named_steps["imputer"]

FEATURE_NAMES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR"
]



logo_col, title_col = st.columns([1.5, 6], vertical_alignment="center")

with logo_col:
    st.image(
        "images/logo.png",
        width=1000
    )

with title_col:
    st.markdown("""
### AI 신용대출 연체 위험 예측

##### XGBoost 기반 머신러닝 모델을 활용하여 고객의 연체 위험을 예측합니다.
""")



with st.sidebar:


    st.markdown("# 🛡️ Loan Risk AI")
    st.caption("신용대출 연체 위험 예측 시스템")

    st.divider()

    st.markdown("# 📊 모델 정보")

    st.markdown("""
<span style="font-size:19px; font-weight:bold;">모델</span>

<span style="font-size:16px;">XGBoost</span>

<br>

<span style="font-size:19px; font-weight:bold;">ROC-AUC</span>

<span style="font-size:16px;">0.7522</span>
""", unsafe_allow_html=True)
    
    st.divider()

    st.markdown("# 📈 데이터 정보")

    st.markdown("""
<span style="font-size:19px; font-weight:bold;">데이터</span>

<span style="font-size:16px;">307,511건</span>

<br>

<span style="font-size:19px; font-weight:bold;">변수</span>

<span style="font-size:16px;">122개</span>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("# 🧑🏻‍💻 개발자")


    st.write("오재원")

    st.divider()

    st.caption("Version 1.0")


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

if st.button(
    "🔍 연체 위험 분석 시작",
    use_container_width=True
):

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
    
    st.divider()

    st.markdown("## 👤 입력 정보")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):

            st.markdown("### 💰 금융 정보")

            st.markdown(f"""
    **연소득**

    {income:,.0f} 원

    ---

    **대출금액**

    {credit:,.0f} 원

    ---

    **상품금액**

    {goods_price:,.0f} 원

    ---

    **연간 상환금**

    {annuity:,.0f} 원
    """)

    with right:
        with st.container(border=True):

            st.markdown("### 👨🏻 고객 정보")

            st.markdown(f"""
    **나이**

    {age} 세

    ---

    **근속연수**

    {employment_years} 년

    ---

    **가족 수**

    {family_members} 명

    ---

    **자녀 수**

    {children} 명
    """)

    st.divider()
    st.markdown("## 🔎 AI 판단 근거")

    input_imputed = imputer.transform(input_data)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_imputed)

    shap_df = pd.DataFrame({
        "변수": FEATURE_NAMES,
        "영향도": shap_values[0]
    })

    shap_df["절대값"] = shap_df["영향도"].abs()

    shap_df = shap_df.sort_values(
        by="절대값",
        ascending=False
    ).head(5)

    st.write("AI가 이번 예측에서 중요하게 본 상위 변수입니다.")

    st.dataframe(
        shap_df[["변수", "영향도"]]
    )

    st.bar_chart(
        shap_df.set_index("변수")["영향도"]
    )
