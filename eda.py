import pandas as pd

df = pd.read_csv("data/application_train.csv")

print("행 개수, 열 개수")
print(df.shape)

print("\n컬럼 목록")
print(df.columns.tolist())

print("\n데이터 정보")
print(df.info())

print("\n결측치 상위 20개")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nTARGET 분포")
print(df["TARGET"].value_counts())