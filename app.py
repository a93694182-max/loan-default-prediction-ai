import streamlit as st
import pandas as pd
import joblib
import shap
import sqlite3
from datetime import datetime



def save_prediction(income, credit, probability, risk_level, prediction):
    cursor.execute("""
    INSERT INTO predictions
    (created_at, income, credit, probability, risk_level, prediction)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        income,
        credit,
        probability,
        risk_level,
        prediction
    ))
    conn.commit()

def load_predictions():
    prediction_df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )
    return prediction_df


    



st.set_page_config(
    page_title="신용대출 연체 위험 예측",
    page_icon="🏦",
    layout="wide"
)

model = joblib.load("loan_model.pkl")



conn = sqlite3.connect("loan_predictions.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    income REAL,
    credit REAL,
    probability REAL,
    risk_level TEXT,
    prediction TEXT
)
""")

conn.commit()



classifier = model.named_steps["classifier"]
imputer = model.named_steps["imputer"]

FEATURE_KOREAN = {
    "AMT_INCOME_TOTAL": "연소득",
    "AMT_CREDIT": "대출금액",
    "AMT_ANNUITY": "연간 상환금",
    "AMT_GOODS_PRICE": "상품금액",
    "CNT_CHILDREN": "자녀 수",
    "CNT_FAM_MEMBERS": "가족 수",
    "DAYS_BIRTH": "나이",
    "DAYS_EMPLOYED": "근속연수",
    "DAYS_REGISTRATION": "가입기간",
    "DAYS_ID_PUBLISH": "신분정보 갱신일",

    "REGION_POPULATION_RELATIVE": "지역 인구 비율",
    "REGION_RATING_CLIENT": "지역 신용등급",
    "REGION_RATING_CLIENT_W_CITY": "도시 신용등급",

    "EXT_SOURCE_1": "외부 신용평가 1",
    "EXT_SOURCE_2": "외부 신용평가 2",
    "EXT_SOURCE_3": "외부 신용평가 3",

    "OBS_30_CNT_SOCIAL_CIRCLE": "주변 연체 건수",
    "DEF_30_CNT_SOCIAL_CIRCLE": "주변 부도 건수",

    "AMT_REQ_CREDIT_BUREAU_DAY": "신용조회(1일)",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "신용조회(1주)",
    "AMT_REQ_CREDIT_BUREAU_MON": "신용조회(1개월)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "신용조회(3개월)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "신용조회(1년)"
}

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


    # ⭐ 여기 추가
    save_prediction(
        float(income),
        float(credit),
        float(probability),
        risk_level,
        "연체 위험" if prediction == 1 else "정상 고객"
    )

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

    st.write("AI가 이번 예측에서 중요하게 판단한 상위 요인입니다.")

    for _, row in shap_df.iterrows():
        feature = FEATURE_KOREAN.get(row["변수"], row["변수"])

        if row["영향도"] > 0:
            st.error(
                f"🔴 **{feature}**\n\n"
                "연체 위험을 높이는 방향으로 영향을 주었습니다."
            )
        else:
            st.success(
                f"🟢 **{feature}**\n\n"
                "연체 위험을 낮추는 방향으로 영향을 주었습니다."
            )
            
    
    st.divider()

    with st.expander("👤 입력 정보 보기", expanded=False):

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


st.markdown("## 📋 분석 기록")

prediction_df = load_predictions()

if prediction_df.empty:
    st.info("아직 저장된 분석 기록이 없습니다.")

else:
    prediction_df = prediction_df.rename(columns={
        "created_at": "분석 시간",
        "income": "연소득",
        "credit": "대출금액",
        "probability": "연체 위험 확률",
        "risk_level": "위험 등급",
        "prediction": "분석 결과"
    })

    page_size = 6

    if "history_page" not in st.session_state:
        st.session_state.history_page = 0

    total_pages = (len(prediction_df) - 1) // page_size + 1

    start = st.session_state.history_page * page_size
    end = start + page_size

    page_df = prediction_df.iloc[start:end]

    nav1, nav2, nav3 = st.columns([2.3, 0.8, 2.3])

    with nav2:

        st.markdown(
            f"<div style='text-align:center; font-size:16px; margin-bottom:10px;'>"
            f"페이지 {st.session_state.history_page + 1} / {total_pages}"
            f"</div>",
            unsafe_allow_html=True
        )

        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button("◀ 이전"):
                if st.session_state.history_page > 0:
                    st.session_state.history_page -= 1
                    st.rerun()

        with btn2:
            if st.button("다음 ▶"):
                if st.session_state.history_page < total_pages - 1:
                    st.session_state.history_page += 1
                    st.rerun()

    card_cols = st.columns(2)

    for idx, row in page_df.reset_index(drop=True).iterrows():
        col = card_cols[idx % 2]

        if row["위험 등급"] == "높음":
            box = col.error
            icon = "🔴"
        elif row["위험 등급"] == "보통":
            box = col.warning
            icon = "🟡"
        else:
            box = col.success
            icon = "🟢"

        box(f"""
### {icon} 분석 결과 : {row["분석 결과"]}

**📅 분석 시간**  
{row["분석 시간"]}

**💰 연소득**  
{row["연소득"]:,.0f} 원

**🏦 대출금액**  
{row["대출금액"]:,.0f} 원

**📊 위험도**  
{row["연체 위험 확률"] * 100:.2f} %

**⚠ 위험등급**  
{row["위험 등급"]}
""")