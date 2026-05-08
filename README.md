# Titanic-ML
🚢 Titanic Survival Prediction Model
EDA부터 피처 엔지니어링, 모델링까지 머신러닝 전 과정을 아우르는 골든타임 예측 프로젝트

1. 프로젝트 개요
<br>
작업 기간: 2026.03.30 ~ 2026.04.03 (1인 프로젝트)
<br>
<br>
<br>
<br>
타이타닉 탑승객 데이터를 분석하여 생존 여부를 예측하는 분류(Classification) 모델을 구축했습니다. 단순한 모델링을 넘어 '골든 타임'이라는 독창적인 파생 변수를 도입하여 데이터 분석가로서의 가설 검증 역량을 발휘했습니다.
<br>
<br>
<br>
<br>
주요 역할: 데이터 전처리, 탐색적 데이터 분석(EDA), 피처 엔지니어링, 예측 모델링 설계 및 최적화 (기여도 100%)
<br>
<br>
<br>
<br>
2.사용 기술 (Tech Stack)
<br>
Language: Python

Library: Pandas, Scikit-learn, Matplotlib, Seaborn

Model: Random Forest (Classifier)

Tool: Visual Studio Code, Jupyter Notebook
<br>
<br>
<br>
<br>
3. 핵심 구현 내용 (Key Features)
<br>
🔍 탐색적 데이터 분석 (EDA) 및 시각화
<br>
Matplotlib/Seaborn을 활용해 성별, 객실 등급(Pclass), 요금(Fare) 등 생존에 결정적인 영향을 미치는 변수들 간의 상관관계 시각화.

단순 수치 확인을 넘어 데이터 속에 숨겨진 생존 패턴(예: 여성 및 고학년 등급 승객의 생존율) 도출.

🛠 피처 엔지니어링 & 전처리
<br>
골든 타임(Golden Time) 계산: 사고 발생 후 대피까지의 시간 확보 가능성을 수치화하여 생존 확률과 연동하는 파생 변수 생성.

데이터 정제: 통계적 근거를 바탕으로 한 결측치(Age, Embarked 등) 처리 및 범주형 데이터 인코딩.

📈 모델링 및 최적화
<br>
Random Forest 알고리즘을 베이스라인으로 설정.

GridSearchCV를 활용한 하이퍼파라미터 튜닝을 통해 모델의 일반화 성능 극대화 및 과적합(Overfitting) 방지.
<br>
<br>
<br>
<br>
4. 기술적 의사결정 (Technical Decision)
<br>
선택: Random Forest

이유: 정형 데이터 분류에서 강력한 성능을 보이며, '특성 중요도(Feature Importance)'를 추출할 수 있어 어떤 변수가 생존 예측에 기여했는지 설명 가능한(XAI) 모델을 구축하기 위함.

대안: Logistic Regression (비선형 패턴 학습 한계), XGBoost (데이터 규모 대비 과적합 리스크).

결과: 최종 평가 지표 AUC Score 0.9 달성.
<br>
<br>
<br>
<br>
5. 트러블슈팅 및 배운 점 (Troubleshooting & Lessons Learned)
<br>
💡 가설 수립의 중요성 체감
문제: 초기에는 '성별'이 생존의 절대적 기준이라 가정했으나, 분석 과정에서 '객실 등급'과의 복합적인 영향이 더 크다는 것을 발견.

해결: 가설을 수정하고 재학습을 진행하여 모델의 일관성을 확보함. 이를 통해 알고리즘 선택보다 도메인 지식에 기반한 피처 엔지니어링이 성능에 더 결정적임을 깨달음.

⚙️ 코드 모듈화 및 관리
문제: 반복적인 실험 과정에서 전처리 코드와 시각화 코드가 혼재되어 유지보수가 어려워짐.

교훈: 향후 프로젝트에서는 전처리 파이프라인, 학습 섹션, 결과 출력 섹션을 명확히 분리(모듈화)하여 생산성을 높여야 한다는 실무적 통찰을 얻음.
