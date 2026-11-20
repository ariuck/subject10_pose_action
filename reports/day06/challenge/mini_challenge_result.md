# Day 06 Mini Challenge — 3팀 무인 상점 이상행동 17개

교재 Mini Challenge는 5일차 본수업 18개 Dataset으로 6일차 흐름을 다시 수행하는 실습이다.
우리는 같은 흐름을 **3팀 Dataset**(5일차 흐름으로 QA·승인·Split까지 끝낸 170개)에 적용했다.

- 교재 6일차 코드(34~40번)는 경로와 "6개 Action / Clip 6개"가 코드 안에 고정되어 있어,
  교재와 같은 공통 함수(`src/sequence_utils.py`)와 같은 모델 설정(`configs/day06_baseline.json`)을 불러오는
  3팀 실행 파일 하나로 다시 구성했다: `scripts/team03/team_day06_baseline.py`

## 1. 사용한 데이터

- Train Subject: S01, S02, S03
- Validation Subject: S04
- Train Clip 수: 102 (3명 × 17동작 × 정면·측면)
- Validation Clip 수: 34
- Action 수: 17
- Test Subject S05 (34개)는 조건을 정한 뒤 마지막에 한 번만 사용했다.

## 2. Sequence Dataset

- Sequence Steps: 30
- Frame Feature 수: 38 (Baseline) / 45 (3팀 Feature 포함)
- Sequence Feature 수: 1,140 (Baseline) / 1,350 (45-D)
- Train 행 수: 102 (Clip 1개 = 1행)
- Validation 행 수: 34
- 영상 길이가 달라도 교재 `resample_sequence()`로 모든 Clip을 30 Step으로 맞췄다.

## 3. Random Forest 학습

- Train Clips: 102
- Feature Count: 1,140
- Classes: 17
- 생성된 Model: 교재 35번과 같은 세 파일을 조건별 폴더에 저장 (joblib은 Git 제외, JSON 두 개는 기록)
  - 38-D Baseline: `artifacts/team03/day06/baseline_38d/` — `rf_action_baseline.joblib`, `rf_feature_columns.json`, `rf_training_info.json`
  - 45-D 최종 조건: `artifacts/team03/day06/team_45d/` — 같은 세 파일
  - 저장: `python scripts/team03/team_day06_baseline.py save --condition <조건>` (Train S01~S03으로만 학습)
- 설정: n_estimators 300, max_depth null, random_state 42, class_weight balanced (교재와 동일)

## 4. Baseline Validation

- Accuracy: 0.7353
- Macro F1: 0.7373
- 정답 Clip 수: 25
- 오답 Clip 수: 9

## 5. 오분류

| Clip | 실제 | 예측 | Confidence |
|---|---|---|---|
| S04_A001_T02 | kiosk_strike | push_two_hands | 0.193 |
| S04_A002_T02 | throw_object | swing_strike | 0.160 |
| S04_A004_T02 | shake_claw_machine | table_strike | 0.170 |
| S04_A005_T01 | push_two_hands | shake_claw_machine | 0.233 |
| S04_A005_T02 | push_two_hands | shake_claw_machine | 0.200 |
| S04_A006_T01 | swing_strike | body_bump | 0.180 |
| S04_A008_T01 | sit_on_chair_phone | body_bump | 0.127 |
| S04_A013_T01 | table_strike | push_two_hands | 0.187 |
| S04_A016_T01 | faint_back | vomit | 0.153 |

원본 MP4에서 확인한 내용 (Skeleton 확인 사진: `reports/team_projects/team03/review_sheets/day06_misclassified_1.jpg`)

- **밀치기 ↔ 흔들기**: 정면에서 두 손을 어깨 높이로 앞으로 내미는 자세가 거의 같다. 실제 뽑기기계·사람이 화면에 없어서 손이 무엇을 미는지, 앞뒤로 반복하는지가 Pose만으로는 잘 드러나지 않는다. 물체 검출이나 반복 횟수 같은 정보가 필요하다.
- **뒤로 기절하기(정면)**: 카메라에서 먼 곳에 카메라 방향으로 누워 몸이 작고 짧게 보인다. 4일차 팀 분석에서 확인한 "정면에서는 누운 자세가 잘 안 잡힌다"는 문제와 같다.
- **의자 위에 앉아 폰보기(정면)**: 앉아 있는 시간이 짧고 정면에서는 무릎이 펴져 보여 서 있는 동작과 비슷했다.
- 오분류는 대부분 **팔을 쓰는 서 있는 동작끼리**(내려치기·던지기·휘두르기·밀치기·흔들기) 일어났다. Confidence도 0.13~0.23으로 낮아 17개 중 확신 없이 고른 경우였다.

## 6. 본수업 Feature 비교 결과 (3팀 데이터로 다시 확인)

- Static VS Sequence: Static(가운데 Frame 38개) 0.5882 / 0.5941 → Sequence 0.7353 / 0.7373
  - 교재 18개에서는 차이가 없었지만, 3팀 데이터에서는 **시간 정보를 쓰는 Sequence가 크게 높았다.** 쓰러지기·내려치기처럼 시간에 따른 변화가 핵심인 동작이 많기 때문으로 보인다.
- Coordinates + Angles VS Coordinates Only: 0.7353 / 0.7373 → 좌표만 0.6765 / 0.6728
  - 교재 18개에서는 좌표만 쓴 쪽이 높았지만, 3팀 데이터에서는 **관절각도를 빼면 낮아졌다.**
- 결과를 일반화하면 안 되는 이유: Validation이 한 사람(S04) 34개뿐이라 Clip 하나가 Accuracy를 약 0.03씩 바꾼다. 같은 실험이 교재 데이터와 3팀 데이터에서 반대로 나온 것도 데이터가 작기 때문이다.

## 7. 한 변수 실험

Baseline에서 **한 번에 한 가지 조건만** 바꿔 5가지를 비교했다.

| 바꾼 조건 | 변경 전 | 변경 후 | 가설 |
|---|---|---|---|
| Frame Feature | 38-D | **45-D** (3팀 Feature 7개 추가) | 눕기·앉기·차기는 상체 기울기·골반-바닥 높이·속도가 도움이 될 것이다 |
| 사용 구간 | Clip 전체 | 동작 구간만 (`in_action`) | 앞뒤 대기 구간을 빼면 동작 특징이 선명해질 것이다 |
| 촬영 방향 | 정면+측면 | 정면만 / 측면만 | 측면이 눕기를 더 잘 보이지만 Train이 절반으로 줄어 비슷하거나 낮아질 것이다 |

## 8. 실험 결과

| 조건 | Feature 수 | Accuracy | Macro F1 | 오분류 | Baseline 대비 |
|---|---:|---:|---:|---:|---|
| Baseline (38-D) | 1,140 | 0.7353 | 0.7373 | 9 / 34 | — |
| **45-D** | 1,350 | 0.7647 | **0.7725** | 8 / 34 | 좋아짐 |
| 동작 구간만 | 1,140 | **0.7941** | 0.7588 | 7 / 34 | 좋아짐 |
| 정면만 | 1,140 | 0.6471 | 0.5882 | 6 / 17 | 나빠짐 |
| 측면만 | 1,140 | 0.6471 | 0.5529 | 6 / 17 | 나빠짐 |

- 45-D에서 예측이 바뀐 Clip은 6개다. `S04_A008_T01` 의자에 앉기가 body_bump → **정답**으로 바뀌었고 Confidence가 0.127 → 0.257로 올랐다. 나머지 5개는 다른 오답으로 바뀌었다(밀치기·흔들기·내려치기끼리).
- 동작 구간만 쓰면 Accuracy가 가장 높았지만 Macro F1은 45-D보다 낮았다. 맞힌 Clip 수는 늘었지만 일부 동작은 오히려 더 틀렸다는 뜻이다.
- 정면만·측면만은 Train이 51개로 줄어 낮아졌다. 한 방향만 찍는 것보다 **두 방향을 함께 쓰는 편이 낫다.**
- 이유에 대한 생각: 45-D의 몸 기울기·골반 높이는 앉기·눕기 구분에 도움이 됐지만, 팔로 하는 서 있는 동작끼리의 혼동은 그대로였다. 이 동작들은 **물체 위치(키오스크·테이블·뽑기기계)** 없이는 구분이 어렵다.

### 최종 조건 선택과 Test

- 선택 기준(Test 전에 정함): Validation **Macro F1이 가장 높은 조건** → 3팀 45-D (0.7725)
- S05 Test (처음 보는 사람 34개): **Accuracy 0.7647 / Macro F1 0.7529** (26개 정답)
- 참고로 같은 Test에서 38-D Baseline은 0.6765 / 0.6630이었다. 45-D의 이점이 새로운 사람에서도 유지됐다.
- Test 결과를 본 뒤에는 조건을 다시 바꾸지 않았다.

## 9. Test를 사용하지 않은 이유

- 실험 중에 Test를 보고 조건을 고르면 Test 사람(S05)에게 맞춘 조건이 되어, 처음 보는 사람에 대한 성능을 정직하게 잴 수 없다. 그래서 조건은 Validation(S04)만으로 정하고 Test는 마지막에 한 번만 봤다.

## 10. 오늘 다시 이해한 핵심

- 같은 Pipeline이라도 데이터가 바뀌면 실험 결론이 달라질 수 있다 (교재 18개: Static = Sequence, 좌표만이 좋음 / 3팀 170개: Sequence가 좋음, 각도도 필요).
- 3팀 Feature(45-D)는 자세형 동작에서 효과가 있었고, 새로운 사람(S05)에서도 38-D보다 높았다.
- 남은 혼동은 **팔로 물체를 다루는 동작**이다. 다음 단계에 필요한 것은 Feature를 더 늘리는 것보다 **물체 위치 정보와 데이터(Take·사람) 추가**다.
