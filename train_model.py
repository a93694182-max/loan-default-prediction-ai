import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib










df = pd.read_csv("data/application_train.csv")

features = [
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

X = df[features]
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier",
     XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",

        scale_pos_weight=11
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("정확도:", round(accuracy * 100, 2), "%")

print(classification_report(
    y_test,
    y_pred
))

print("\n혼동행렬")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



# Confusion Matrix 시각화

disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred),
    display_labels=["Normal", "Default"]
)

fig, ax = plt.subplots(figsize=(6,5))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("images/confusion_matrix.png")

plt.show()



#####################################################




y_prob = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(
    y_test,
    y_prob
)

print("ROC-AUC:", round(auc, 4))

''

# ROC Curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {auc:.4f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig("images/roc_curve.png")

plt.show()



#####################################################





# Feature Importance

classifier = model.named_steps["classifier"]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig, ax = plt.subplots(figsize=(9,6))

bars = ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.invert_yaxis()

ax.set_title(
    "Top 10 Feature Importance",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Importance")


ax.set_xlim(
     0,
     importance_df["Importance"].max() * 1.15
    )



# 막대 끝에 숫자 표시
for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.002,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()


# 학습된 XGboost 파이프라인 저장
joblib.dump(model, "loan_model.pkl")

print("모델 저장 완료")