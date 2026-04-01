import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
import pickle
import json
import numpy as np

# 1. Kaggle 데이터 로드
df = pd.read_csv('train.csv')

# 2. 데이터 전처리 (샘플 PDF의 전처리 단계 반영)
# 필요한 컬럼만 선택
cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = df[cols]

# 결측치 처리 및 인코딩
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 3. 데이터 분할
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4-1. Logistic Regression 모델 학습
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

# 4-2. Random Forest 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 5. 두 모델 평가
# Logistic Regression 평가
logreg_pred = logreg_model.predict(X_test)
logreg_pred_proba = logreg_model.predict_proba(X_test)[:, 1]
logreg_fpr, logreg_tpr, _ = roc_curve(y_test, logreg_pred_proba)
logreg_auc = auc(logreg_fpr, logreg_tpr)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_f1 = f1_score(y_test, logreg_pred)

# Random Forest 평가
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
rf_auc = auc(rf_fpr, rf_tpr)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
cm = confusion_matrix(y_test, rf_pred)

# 성능 지표 출력
print("=" * 70)
print("📊 모델 비교 성능 평가 (Logistic Regression vs Random Forest)")
print("=" * 70)
print("\n🔵 Logistic Regression:")
print(f"  ✓ 정확도 (Accuracy): {logreg_accuracy:.4f} ({logreg_accuracy*100:.2f}%)")
print(f"  ✓ F1-Score: {logreg_f1:.4f}")
print(f"  ✓ AUC Score: {logreg_auc:.4f}")

print("\n🔴 Random Forest (최종 선택 모델):")
print(f"  ✓ 정확도 (Accuracy): {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"  ✓ F1-Score: {rf_f1:.4f}")
print(f"  ✓ AUC Score: {rf_auc:.4f}")
print(f"  ✓ AUC 향상도: +{(rf_auc - logreg_auc):.4f}")
print("=" * 70)

# ROC 데이터 저장 (앱에서 시각화할 수 있도록)
roc_data = {
    "logreg": {
        "fpr": logreg_fpr.tolist(),
        "tpr": logreg_tpr.tolist(),
        "auc": round(logreg_auc, 4),
        "accuracy": round(logreg_accuracy, 4),
        "f1_score": round(logreg_f1, 4)
    },
    "rf": {
        "fpr": rf_fpr.tolist(),
        "tpr": rf_tpr.tolist(),
        "auc": round(rf_auc, 4),
        "accuracy": round(rf_accuracy, 4),
        "f1_score": round(rf_f1, 4)
    }
}

with open('roc_data.json', 'w') as f:
    json.dump(roc_data, f, indent=4)

# 성능 지표를 JSON으로 저장 (앱에서 불러올 수 있도록)
metrics = {
    "accuracy": round(rf_accuracy, 4),
    "f1_score": round(rf_f1, 4),
    "auc_score": round(rf_auc, 4),
    "accuracy_percentage": round(rf_accuracy * 100, 2),
    "recall": round(cm[1,1]/(cm[1,0]+cm[1,1]), 4),
    "specificity": round(cm[0,0]/(cm[0,0]+cm[0,1]), 4),
    "precision": round(cm[1,1]/(cm[0,1]+cm[1,1]), 4),
    "logreg_auc": round(logreg_auc, 4),
    "logreg_accuracy": round(logreg_accuracy, 4)
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# 6. 최종 모델(Random Forest) 저장
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\n✅ 학습 완료!")
print("✅ ROC 데이터가 roc_data.json에 저장되었습니다.")
print("✅ 성능 지표가 model_metrics.json에 저장되었습니다.")