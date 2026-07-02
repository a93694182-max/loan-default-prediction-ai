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


def money_input(label, default):

    text = st.text_input(
        label,
        value=f"{default:,}"
    )

    value = int(text.replace(",", ""))

    formatted = f"{value:,}"

    if formatted != text:
        st.session_state[label] = formatted

    return value


def delete_all_predictions():
    conn = sqlite3.connect("loan_predictions.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()

    







st.set_page_config(
    page_title="신용대출 연체 위험 예측",
    page_icon="🏦",
    layout="wide"
)



st.markdown("""
<style>
div[data-testid="stNumberInput"] p {
    font-size: 20px !important;
    font-weight: 700 !important;
}

div.stButton > button {
    font-size: 20px !important;
    font-weight: 700 !important;
    height: 48px !important;
}

div.stButton > button p {
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* Expander 제목 */
details summary p {
    font-size: 22px !important;
    font-weight: bold !important;
}
            
div[data-testid="stDownloadButton"] > button {
    font-size: 20px !important;
    font-weight: 700 !important;
    height: 48px !important;
}

div[data-testid="stDownloadButton"] > button p {
    font-size: 20px !important;
    font-weight: 700 !important;
}
            
</style>
""", unsafe_allow_html=True)




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


outer1, center, outer2 = st.columns([1.5, 7, 1.5])

with center:

    logo_col, title_col = st.columns([1.5, 6], vertical_alignment="center")

    with logo_col:
        st.image(
            "images/logo.png",
            width=1000
        )

    with title_col:

        st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)

        st.markdown("""
    ### AI 신용대출 연체 위험 예측

    ##### XGBoost 기반 머신러닝 모델을 활용하여 고객의 연체 위험을 예측합니다.
    """)


    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "연소득",
            min_value=0,
            max_value=1000000000,
            value=50000000,
            step=1000000
        )

        credit = st.number_input(
            "대출금액",
            min_value=0,
            max_value=1000000000,
            value=20000000,
            step=1000000
        )

        annuity = st.number_input(
            "연간 상환금",
            min_value=0,
            max_value=200000000,
            value=3000000,
            step=100000
        )

        goods_price = st.number_input(
            "상품 금액",
            min_value=0,
            max_value=1000000000,
            value=20000000,
            step=1000000
        )

    with col2:

        children = st.number_input(
            "자녀 수",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        family_members = st.number_input(
            "가족 수",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        age = st.number_input(
            "나이",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        employment_years = st.number_input(
            "근속 연수",
            min_value=0,
            max_value=60,
            value=5,
            step=1
        )


    if st.button(
        "🔍 연체 위험 분석 시작",
        use_container_width=True
    ):


        # 입력값 검증
        warnings = []

        if credit > income * 10:
            warnings.append("대출금액이 연소득에 비해 매우 높은 수준입니다.")

        if annuity > income:
            warnings.append("연간 상환금이 연소득을 초과합니다.")

        if goods_price < credit * 0.5:
            warnings.append("상품 금액이 대출금액에 비해 매우 낮은 수준입니다.")

        if warnings:

            message = "### ⚠️ 입력값 확인 안내\n\n"

            for w in warnings:
                message += f"- {w}\n"

            st.warning(message)


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


        
        save_prediction(
            float(income),
            float(credit),
            float(probability),
            risk_level,
            "연체 위험" if prediction == 1 else "정상 고객"
        )

        st.divider()
        st.markdown("## 📊 예측 결과")

        st.markdown(
            """
        <h5 style="margin-bottom:0px;">
        연체 위험 확률
        </h5>
        """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
        <h4 style="margin-top:-10px; margin-bottom:15px;">
        {probability * 100:.2f}%
        </h4>
        """,
            unsafe_allow_html=True
        )
        
        st.progress(float(probability))

        result_box(f"{result_icon} 위험 등급 : {risk_level}")

        st.info(f"""
        ### 📋 AI 분석 요약

        - 예측 결과: **{"연체 위험" if prediction == 1 else "정상 고객"}**
        - 연체 위험 확률: **{probability * 100:.2f}%**
        - 위험 등급: **{risk_level}**
        - 사용 모델: **XGBoost**
        """)

        st.markdown("## 💡 판단 의견")

        st.markdown(
            f"""
        <h5 style="line-height:1.8; margin-top:0px;">
        {result_message}
        </h5>
        """,
            unsafe_allow_html=True
        )
        
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


        st.markdown(
            """
        <h5 style="margin-bottom:0px;">
        AI가 이번 예측에서 중요하게 판단한 상위 요인입니다
        </h5>
        """,
            unsafe_allow_html=True
        )
    

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

    if "history_expanded" not in st.session_state:
        st.session_state.history_expanded = False

    with st.expander(
        "📋 분석 기록 보기",
        expanded=st.session_state.history_expanded
    ):

        prediction_df = load_predictions()

        download_df = prediction_df.rename(columns={
            "created_at": "분석 시간",
            "income": "연소득",
            "credit": "대출금액",
            "probability": "연체 위험 확률",
            "risk_level": "위험 등급",
            "prediction": "분석 결과"
        })

        download_df["연체 위험 확률"] = (
            download_df["연체 위험 확률"] * 100
        ).round(2).astype(str) + "%"


        csv = download_df.to_csv(
            index=False,
            encoding="utf-8-sig"
        )
        

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


            card_cols = st.columns(2)

            for idx, row in page_df.reset_index(drop=True).iterrows():
                col = card_cols[idx % 2]

                if row["위험 등급"] == "높음":
                    color = "#ff4b4b"
                    icon = "🔴"
                elif row["위험 등급"] == "보통":
                    color = "#f9c74f"
                    icon = "🟡"
                else:
                    color = "#22c55e"
                    icon = "🟢"

                with col.container(border=True):

                    st.markdown(
                        f"""
                <div style="padding:8px;">

                <h3 style="color:{color}; margin-bottom:15px;">
                {icon}  {row["분석 결과"]}
                </h3>

                <p>📅 <b>분석 시간</b> : {row["분석 시간"]}</p>

                <p>💰 <b>연소득</b> : {row["연소득"]:,.0f} 원</p>

                <p>🏦 <b>대출금액</b> : {row["대출금액"]:,.0f} 원</p>

                <p>📊 <b>위험도</b> : {row["연체 위험 확률"]*100:.2f}%</p>

                <p>⚠ <b>위험등급</b> :
                <span style="color:{color}; font-weight:bold;">
                {row["위험 등급"]}
                </span>
                </p>

                </div>
                """,
                        unsafe_allow_html=True
                    )
            st.markdown("---")

            nav1, nav2, nav3, nav4, nav5 = st.columns([3, 1, 2, 1, 3])

            with nav2:
                if st.button("◀", key="history_prev", use_container_width=True):
                    if st.session_state.history_page > 0:
                        st.session_state.history_page -= 1
                        st.session_state.history_expanded = True
                        st.rerun()

            with nav3:
                st.markdown(
                    f"""
            <div style="text-align:center; padding-top:2px;">
                <span style="
                    font-size:40px !important;
                    font-weight:800 !important;
                    line-height:1.2 !important;
                ">
                    페이지 {st.session_state.history_page + 1} / {total_pages}
                </span>
            </div>
            """,
                    unsafe_allow_html=True
                )

            with nav4:
                if st.button("▶", key="history_next", use_container_width=True):
                    if st.session_state.history_page < total_pages-1:
                        st.session_state.history_page += 1
                        st.session_state.history_expanded = True
                        st.rerun()


        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:

            st.download_button(
                label="📥 분석 기록 다운로드",
                data=csv,
                file_name="loan_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with btn_col2:

        
            if "delete_confirm" not in st.session_state:
                st.session_state.delete_confirm = False

            if not st.session_state.delete_confirm:

                if st.button(
                    "🗑 분석 기록 전체 삭제",
                    use_container_width=True
                ):
                    st.session_state.delete_confirm = True
                    st.rerun()

            else:

                st.warning("정말 모든 분석 기록을 삭제하시겠습니까?")

                confirm1, confirm2 = st.columns(2)

                with confirm1:
                    if st.button("✅ 삭제", use_container_width=True):
                        delete_all_predictions()
                        st.session_state.delete_confirm = False
                        st.session_state.history_page = 0
                        st.success("분석 기록이 모두 삭제되었습니다.")
                        st.rerun()

                with confirm2:
                    if st.button("취소", use_container_width=True):
                        st.session_state.delete_confirm = False
                        st.rerun()