# TeleAI Ops — 텔레콤 트래픽 예측 시스템

밀라노 텔레콤 인터넷 트래픽을 LSTM 신경망으로 예측하고, RMSE 기반으로 재학습 필요 여부를 판정하는 **Human-in-the-Loop ML 운영 플랫폼**입니다.

---

## 주요 기능

- **LSTM 기반 트래픽 예측**: 10분 단위 인터넷 트래픽을 7일(1008 스텝) 과거 데이터로 예측
- **실시간 성능 모니터링**: RMSE 임계값 초과 여부 자동 감지
- **Human-in-the-Loop 재학습**: 수동(운영자 승인) / 자동 모드 지원
- **LLM 성능 분석 보고서**: OpenAI GPT-4o-mini 기반 자동 보고서 생성
- **웹 대시보드**: 실측 vs 예측 차트, 롤링 RMSE 차트, KPI 카드

---

## 시스템 구성

```
telecom/
├── server_model/
│   ├── main.py               # FastAPI 서버 (API + 정적 파일 서빙)
│   ├── model.py              # LSTM 모델 학습
│   ├── weight_used_model.py  # 저장된 모델로 추론
│   └── config.py             # 경로 및 하이퍼파라미터 설정
├── server/
│   ├── model/                # 학습된 모델 저장 위치
│   ├── uploaded_files/       # 업로드된 CSV 파일
│   ├── view-model-architecture/  # 예측 그래프 이미지
│   └── retrain_log.json      # 재학습 이력 로그
├── public/
│   ├── index.html            # 운영 대시보드
│   ├── report.html           # LLM 성능 분석 보고서 페이지
│   └── js/
│       ├── get.js            # 초기 대시보드 로드
│       ├── post.js           # CSV 업로드 및 차트 렌더링
│       ├── telecom.js        # 모드 전환 및 재학습 처리
│       └── report.js         # 보고서 페이지 로직
├── .env                      # 환경변수 (OPENAI_API_KEY)
└── requirements.txt          # Python 의존성
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install fastapi "uvicorn[standard]" pandas pytz python-multipart matplotlib
pip install numpy scikit-learn keras tensorflow python-dotenv requests
```

또는:

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`telecom/.env` 파일 생성:

```env
OPENAI_API_KEY=sk-proj-...
```

### 3. 서버 실행

```bash
cd telecom
python -m uvicorn server_model.main:app --port 8001 --reload
```

### 4. 대시보드 접속

브라우저에서 `http://localhost:8001` 접속

---

## 사용 방법

### 기본 워크플로우

```
1. 대시보드 접속 → 기존 모델로 원본 데이터 초기 예측 표시
2. milan_telecom_December_2013.CSV 파일 업로드 → 자동으로 예측 실행 및 RMSE 계산
3. RMSE 임계값 초과 시 → 토스트 알림 표시
4. [보고서 보기] 클릭 → LLM 성능 분석 보고서 확인
5. 보고서 하단에서 재학습 [확인] 또는 [거부] 선택
```

### 입력 데이터 형식

업로드할 CSV 파일은 밀라노 텔레콤 형식을 따라야 합니다:

| 컬럼 | 설명 |
|------|------|
| `datetime` | 10분 단위 타임스탬프 |
| `internet` | 인터넷 트래픽 (예측 타겟) |
| `smsin` | SMS 수신 트래픽 |
| `callin` | 통화 수신 트래픽 |

> 예시 파일: `server/uploaded_files/milan_telecom_final_1.csv`

### 재학습 모드

| 모드 | 설명 |
|------|------|
| **수동 (Manual)** | RMSE 초과 시 토스트 알림 → 운영자가 보고서 확인 후 재학습 승인/거부 |
| **자동 (Auto)** | RMSE 초과 시 즉시 자동 재학습 |

---

## LSTM 모델 구조

| 레이어 | 유닛 | 설정 |
|--------|------|------|
| LSTM | 64 | return_sequences=True |
| Dropout | — | 0.2 |
| LSTM | 64 | return_sequences=True |
| Dropout | — | 0.2 |
| LSTM | 32 | — |
| Dropout | — | 0.2 |
| Dense | 1 | 출력 (다음 10분 트래픽) |

**입력 특성**: `[internet, smsin, callin, hour, dayofweek, is_weekend]` (6개)  
**LOOKBACK**: 1008 스텝 (7일 × 144 스텝/일)  
**학습/테스트 분할**: 72% / 28%  
**에포크**: 3 | **배치 크기**: 64 | **옵티마이저**: Adam

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 메인 대시보드 (index.html) |
| GET | `/report.html` | LLM 보고서 페이지 |
| GET | `/health` | 서버 상태 확인 |
| GET | `/init` | 기존 모델로 원본 데이터 초기 추론 |
| POST | `/upload` | CSV 업로드 및 예측 실행 |
| POST | `/retrain` | 운영자 승인 후 모델 재학습 |
| POST | `/reject` | 재학습 거부 및 이력 기록 |
| GET | `/report` | LLM 성능 분석 보고서 생성 |
| POST | `/set-mode` | 운영 모드 전환 (auto/manual) |
| GET | `/download` | 예측 그래프 이미지 다운로드 |

---

## 주요 설정값 (config.py)

```python
LOOKBACK        = 1008   # 시퀀스 길이 (7일 × 144 스텝)
RMSE_THRESHOLD  = 200.0  # 재학습 판정 임계값
```

---

## 재학습 전략

재학습 시 **원본 학습 데이터 + 사용자 업로드 CSV를 합산**하여 새 모델을 학습합니다.

```
원본 데이터 (milan_telecom_final_1.csv)
         +
사용자 업로드 CSV
         ↓
    합산 데이터셋으로 재학습
         ↓
새 RMSE 계산 → 개선 여부 비교
```

재학습 이력은 `server/retrain_log.json`에 JSON 형식으로 기록됩니다.

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 웹 프레임워크 | FastAPI + Uvicorn |
| ML 프레임워크 | Keras (TensorFlow 백엔드) |
| 데이터 처리 | Pandas, NumPy, Scikit-learn |
| 시각화 | Matplotlib (서버), Chart.js (클라이언트) |
| LLM | OpenAI GPT-4o-mini |
| 프론트엔드 | Vanilla JS, Bootstrap 5, Chart.js |
