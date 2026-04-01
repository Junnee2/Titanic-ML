import pandas as pd
import pickle
import numpy as np

# 훈련 데이터 분석
df = pd.read_csv('train.csv')
cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = df[cols]
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 생존한 승객들의 통계
print("=" * 60)
print("📊 생존한 승객(Survived=1)의 통계")
print("=" * 60)
survived = df[df['Survived'] == 1]
print(f"\n평균 객실 등급: {survived['Pclass'].mean():.2f}")
print(f"평균 나이: {survived['Age'].mean():.2f}")
print(f"평균 운임: {survived['Fare'].mean():.2f}")
print(f"성별 분포:\n{survived['Sex'].value_counts().to_dict()} (0=남성, 1=여성)")
print(f"평균 동반 형제/배우자: {survived['SibSp'].mean():.2f}")
print(f"평균 동반 부모/자녀: {survived['Parch'].mean():.2f}")

print("\n" + "=" * 60)
print("사망한 승객(Survived=0)의 통계")
print("=" * 60)
dead = df[df['Survived'] == 0]
print(f"\n평균 객실 등급: {dead['Pclass'].mean():.2f}")
print(f"평균 나이: {dead['Age'].mean():.2f}")
print(f"평균 운임: {dead['Fare'].mean():.2f}")
print(f"성별 분포:\n{dead['Sex'].value_counts().to_dict()} (0=남성, 1=여성)")
print(f"평균 동반 형제/배우자: {dead['SibSp'].mean():.2f}")
print(f"평균 동반 부모/자녀: {dead['Parch'].mean():.2f}")

# 모델 테스트
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("\n" + "=" * 60)
print("🎯 생존 확률 높은 조합들")
print("=" * 60)

test_cases = [
    [1, 1, 10, 0, 1, 150],  # 1등실, 여성, 10살, 부모 동반, 높은 운임
    [1, 1, 25, 0, 0, 100],  # 1등실, 여성, 25살, 혼자, 중간 운임
    [2, 1, 20, 0, 0, 50],   # 2등실, 여성, 20살, 혼자, 낮은 운임
    [3, 1, 20, 0, 0, 20],   # 3등실, 여성, 20살, 혼자, 매우 낮은 운임
    [1, 0, 25, 0, 0, 100],  # 1등실, 남성, 25살, 혼자, 중간 운임
]

for i, features in enumerate(test_cases, 1):
    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0][1]
    status = "✅ 생존" if pred == 1 else "❌ 사망"
    print(f"\n테스트 {i}: {status} (생존 확률: {prob:.1%})")
    print(f"  → 객실등급: {features[0]}, 성별: {'여성' if features[1] == 1 else '남성'}, "
          f"나이: {features[2]}, 운임: ${features[5]:.0f}")

print("\n" + "=" * 60)
print("💡 생존 가능성이 높은 조건")
print("=" * 60)
print("✓ 성별: 여성 (매우 중요)")
print("✓ 객실 등급: 1등실 또는 2등실")
print("✓ 나이: 어릴수록 좋음 (어린이 우대)")
print("✓ 운임: 높을수록 좋음 (부유층)")
print("✓ 동반인: 부모 또는 배우자와 함께")
