import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. 데이터 불러오기
file_name = '/Users/doik/Desktop/SKALA/workspace/ModelServing/telecom/server/uploaded_files/milan_telecom_timeseries_1.csv'
print(f"📄 '{file_name}' 파일을 불러옵니다...")
df = pd.read_csv(file_name)

# 2. datetime 처리
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')
df = df.set_index('datetime')

print("-" * 30)

# ==========================================
# 🔍 데이터 개수 및 구성 확인
# ==========================================
print(f"✅ 1. 데이터 개수 (총 행 수): {len(df)}개")
print("\n[처음 5행 데이터]")
print(df.head())

print("-" * 30)

# ==========================================
# 📅 날짜 범위 확인
# ==========================================
start_date = df.index.min()
end_date = df.index.max()
data_duration = end_date - start_date

print(f"✅ 2. 데이터 날짜 범위:")
print(f"   - 시작일: {start_date}")
print(f"   - 종료일: {end_date}")
print(f"   - 총 기간: {data_duration.days}일 {data_duration.seconds // 3600}시간")

print("-" * 30)

# ==========================================
# 📈 시계열 그래프
# ==========================================
print("✅ 3. 시계열 그래프를 생성합니다...")

plt.figure(figsize=(15, 7))
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(df.index, df['internet'], linewidth=1, label='Internet Traffic')

plt.title('Internet Traffic Trend (Aggregated Time Series)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Traffic Volume', fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

print("✅ 모든 작업이 완료되었습니다!")