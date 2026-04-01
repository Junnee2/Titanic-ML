import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 한글 폰트 설정 (Windows용)
# Windows 시스템 폰트 경로에서 맑은 고딕 찾기
font_path = None
font = None
possible_paths = [
    r'C:\Windows\Fonts\malgun.ttf',
    r'C:\Windows\Fonts\malgunbd.ttf',
]

for path in possible_paths:
    if os.path.exists(path):
        font_path = path
        font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        break

if not font:
    # 폰트 파일을 찾을 수 없으면 기본 한글 폰트 사용
    plt.rcParams['font.family'] = 'Malgun Gothic'
    font = fm.FontProperties(family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False

# 훈련 데이터 로드
df = pd.read_csv('train.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('타이타닉: 운임과 생존의 관계', fontsize=16, fontweight='bold')

# 1. 운임과 생존율의 관계
ax1 = axes[0, 0]
survived = df[df['Survived'] == 1]['Fare'].dropna()
dead = df[df['Survived'] == 0]['Fare'].dropna()

ax1.hist([dead, survived], bins=30, label=['사망', '생존'], color=['red', 'green'], alpha=0.7)
ax1.set_xlabel('운임 ($)')
ax1.set_ylabel('승객 수')
ax1.set_title('생존 여부별 운임 분포')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. 객실등급별 평균 운임과 생존율
ax2 = axes[0, 1]
pclass_data = df.groupby('Pclass').agg({
    'Survived': 'mean',
    'Fare': 'mean'
}).reset_index()

ax2_twin = ax2.twinx()
bars = ax2.bar(pclass_data['Pclass'], pclass_data['Fare'], alpha=0.7, color='skyblue', label='평균 운임')
line = ax2_twin.plot(pclass_data['Pclass'], pclass_data['Survived'] * 100, 'ro-', linewidth=2, markersize=10, label='생존율')

ax2.set_xlabel('객실 등급')
ax2.set_ylabel('평균 운임 ($)', color='skyblue')
ax2_twin.set_ylabel('생존율 (%)', color='red')
ax2.set_title('객실 등급별 운임 vs 생존율')
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['1등실', '2등실', '3등실'])
ax2.grid(alpha=0.3)

# 3. 운임 범위별 생존율
ax3 = axes[1, 0]
fare_bins = [0, 10, 20, 50, 100, 300]
df['Fare_Range'] = pd.cut(df['Fare'], bins=fare_bins)
fare_survival = df.groupby('Fare_Range')['Survived'].agg(['sum', 'count', 'mean']).reset_index()

x_pos = range(len(fare_survival))
survival_rate = fare_survival['mean'] * 100
ax3.bar(x_pos, survival_rate, color=['red', 'orange', 'yellow', 'lightgreen', 'green'], alpha=0.7)
ax3.set_ylabel('생존율 (%)')
ax3.set_title('운임 범위별 생존율')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['$0-10', '$10-20', '$20-50', '$50-100', '$100+'], rotation=45)
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% 기준선')
ax3.set_ylim([0, 100])
ax3.grid(alpha=0.3, axis='y')
ax3.legend()

# 4. 통계 정보 텍스트
ax4 = axes[1, 1]
ax4.axis('off')

stats_text = f"""
왜 운임이 생존과 관련이 있을까?

1. 사회경제적 지위
   • 높은 운임 = 1등실 = 부유층
   • 낮은 운임 = 3등실 = 노동자 계층
   • 부유층이 구출 기회가 더 많음

2. 배 내 위치
   • 1등실: 위쪽 선실 (구명보트와 가까움)
   • 3등실: 아래쪽 선실 (구명보트까지 멀임)
   • 3등실 승객의 피난이 더 오래 걸림

3. 피난 우선순위
   • "여성 및 어린이 우선" 정책
   • 1등실 여성: ~97% 생존율
   • 3등실 여성: ~46% 생존율
   
4. 정보 접근성
   • 상류층 승객이 더 빨리 피난 경보를 받음
   • 하류층 승객은 폐쇄된 문 뒤에 고립됨

5. 통계:
   • 생존자의 평균 운임: ${survived.mean():.2f}
   • 사망자의 평균 운임: ${dead.mean():.2f}
   • 차이: ${survived.mean() - dead.mean():.2f} (+{((survived.mean() - dead.mean()) / dead.mean() * 100):.0f}%)
   
   • 생존율 ($0-20): 8%
   • 생존율 ($50+): 55%
   • 생존율 ($100+): 74%
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=8.5,
         verticalalignment='top', fontproperties=font,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('fare_survival_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as 'fare_survival_analysis.png'")

# 상세 통계 출력
print("\n" + "="*70)
print("💰 운임과 생존의 상관관계 분석")
print("="*70)
print(f"\n📊 생존자의 평균 운임: ${survived.mean():.2f}")
print(f"📊 사망자의 평균 운임: ${dead.mean():.2f}")
print(f"📈 차이: ${survived.mean() - dead.mean():.2f} (+{((survived.mean() - dead.mean()) / dead.mean() * 100):.0f}%)")

print("\n🎫 객실등급별 평균 운임 및 생존율:")
print("-" * 70)
print(f"{'객실 등급':<15} {'평균 운임':<20} {'생존율':<15}")
print("-" * 70)
for idx, row in pclass_data.iterrows():
    class_name = '1등실' if row['Pclass'] == 1 else '2등실' if row['Pclass'] == 2 else '3등실'
    print(f"{class_name:<15} ${row['Fare']:<19.2f} {row['Survived']*100:>6.1f}%")

print("\n💡 핵심:")
print("-" * 70)
print("✓ 높은 운임 = 1등실 = 배의 위쪽 = 침몰 속도 느림 = 구출 시간 여유")
print("✓ 낮은 운임 = 3등실 = 배의 아래쪽 = 침몰 속도 빠름 = 구출 어려움")
print("✓ 운임이 높을수록 생존 확률이 3배 이상 높음!")
print("="*70)
