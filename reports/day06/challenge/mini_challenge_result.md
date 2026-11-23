# Day 06 Mini Challenge — 3팀 무인 상점 이상행동 17개

교재 Mini Challenge는 5일차 본수업 18개 Dataset으로 6일차 흐름을 다시 수행하는 실습이다.
우리는 같은 흐름을 **3팀 Dataset**(5일차 흐름으로 QA·승인·Split까지 끝낸 170개)에 적용했다.

- 교재 6일차 코드(34~40번)는 경로와 "6개 Action / Clip 6개"가 코드 안에 고정되어 있어,
  교재와 같은 공통 함수(`src/sequence_utils.py`)와 같은 모델 설정(`configs/day06_baseline.json`)을 불러오는
  3팀 실행 파일 하나로 다시 구성했다: `scripts/team03/team_day06_baseline.py`

> 2026-11-23 팀에서 사람 ID 순서와 동작 이름(`configs/team03_action_classes.csv`)을 통일했다.
> 데이터 이름을 모두 바꾼 뒤 번호 기준 Split(Train S01~S03 / Validation S04 / Test S05) 그대로 다시 학습했다.
> 같은 번호에 들어가는 사람이 바뀌었기 때문에 이 문서의 숫자는 모두 새로 학습한 결과다.

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

- Accuracy: 0.6176
- Macro F1: 0.6235
- 정답 Clip 수: 21
- 오답 Clip 수: 13

## 5. 오분류

| Clip | 실제 | 예측 | Confidence |
|---|---|---|---|
| S04_A001_T02 | kiosk_tap | push | 0.220 |
| S04_A002_T01 | throw_one_hand | kiosk_tap | 0.160 |
| S04_A003_T01 | front_kick | sit_on_object | 0.173 |
| S04_A004_T01 | grab_shake | shotgun | 0.223 |
| S04_A004_T02 | grab_shake | push | 0.233 |
| S04_A005_T01 | push | shotgun | 0.193 |
| S04_A005_T02 | push | grab_shake | 0.210 |
| S04_A006_T02 | swing | grab_shake | 0.190 |
| S04_A010_T02 | lie_backward | lie_forward | 0.243 |
| S04_A011_T02 | body_slam | swing | 0.177 |
| S04_A012_T02 | crouch_take_item | vomit | 0.180 |
| S04_A013_T01 | shotgun | vomit | 0.190 |
| S04_A013_T02 | shotgun | grab_shake | 0.190 |

- 13개 중 9개가 **팔을 쓰는 서 있는 동작끼리**(두드리기·던지기·잡고 흔들기·밀치기·휘두르기·샷건·몸통 박치기)의 혼동이었다. 특히 잡고 흔들기 ↔ 밀치기 ↔ 샷건이 서로 섞였다.
- 정면에서 두 손을 앞으로 내미는 자세가 비슷하고, 실제 뽑기기계·테이블이 화면에 없어서 손이 무엇을 다루는지가 Pose만으로는 드러나지 않는다. 물체 위치나 반복 횟수 같은 정보가 필요하다.
- 나머지 4개는 구토로 잘못 본 2개(쪼그려 물건 빼내기, 샷건), 앞차기 → 위에 앉기, 누운 방향을 헷갈린 뒤로 눕기 → 앞으로 눕기였다.
- Confidence는 0.16~0.24로 낮아 17개 중 확신 없이 고른 경우였다.

## 6. 본수업 Feature 비교 결과 (3팀 데이터로 다시 확인)

- Static VS Sequence: Static(가운데 Frame 38개) 0.6471 / 0.6020 → Sequence 0.6176 / 0.6235
  - Accuracy는 Static이 1개 더 맞혔지만, 동작별 평균(Macro F1)은 Sequence가 높았다. 차이가 작아 어느 쪽이 낫다고 말하기 어렵다.
- Coordinates + Angles VS Coordinates Only: 0.6176 / 0.6235 → 좌표만 0.6176 / 0.5905
  - 맞힌 개수는 같았지만, 관절각도를 빼면 Macro F1이 낮아졌다.
- 결과를 일반화하면 안 되는 이유: Validation이 한 사람(S04) 34개뿐이라 Clip 하나가 Accuracy를 약 0.03씩 바꾼다. 이름 변경 전(Validation 사람이 달랐을 때)에는 Sequence가 Static보다 확실히 높았는데, 사람이 바뀌자 차이가 거의 사라졌다.

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
| Baseline (38-D) | 1,140 | 0.6176 | 0.6235 | 13 / 34 | — |
| **45-D** | 1,350 | 0.7353 | **0.7196** | 9 / 34 | 좋아짐 |
| 동작 구간만 | 1,140 | 0.7353 | 0.7098 | 9 / 34 | 좋아짐 |
| 정면만 | 1,140 | 0.7647 | 0.7157 | 4 / 17 | 좋아짐 |
| 측면만 | 1,140 | 0.5882 | 0.4902 | 7 / 17 | 나빠짐 |

- 45-D에서 예측이 바뀐 Clip은 8개다. 5개가 정답으로 바뀌었고(앞차기, 잡고 흔들기 2개, 몸통 박치기, 샷건) 1개가 새로 틀렸다(뒤로 넘어지기 → 앞으로 넘어지기).
- 동작 구간만 써도 45-D와 같은 9개 오답까지 줄었다. 앞뒤 대기 구간이 Baseline의 혼동을 늘리고 있었다는 뜻이다.
- 정면만은 Train이 51개로 줄었는데도 Baseline보다 높았고, 측면만은 크게 낮았다. 다만 Validation이 17개뿐이라 정면이 항상 낫다고 보기는 어렵다.
- 이유에 대한 생각: 45-D의 속도 Feature가 팔·다리를 빠르게 쓰는 동작(앞차기·몸통 박치기·샷건)을 나누는 데 도움이 됐다. 하지만 밀치기는 끝까지 한 번도 맞히지 못했다. 이 동작들은 **물체 위치(키오스크·테이블·뽑기기계)** 없이는 구분이 어렵다.

### 최종 조건 선택과 Test

- 선택 기준(Test 전에 정함): Validation **Macro F1이 가장 높은 조건** → 3팀 45-D (0.7196)
- S05 Test (처음 보는 사람 34개): **Accuracy 0.6471 / Macro F1 0.6020** (22개 정답)
- 참고로 같은 Test에서 38-D Baseline은 0.5882 / 0.5652였다. 45-D의 이점은 새로운 사람에서도 유지됐다.
- 다만 Validation(0.7353)보다 크게 낮았다. S05의 휘두르기·밀치기·앞차기·샷건·잡고 흔들기 6개가 "몸통 박치기"로, 옆으로 쓰러지기 2개가 "바닥에 앉기"로 예측됐다.
- 동작 이름 대신 그룹(타격·파손, 밀기·흔들기, 앉기, 눕기·쓰러짐 등)으로 채점하면 25 / 34 (0.7353)였다.
- Test 결과를 본 뒤에는 조건을 다시 바꾸지 않았다.

## 9. Test를 사용하지 않은 이유

- 실험 중에 Test를 보고 조건을 고르면 Test 사람(S05)에게 맞춘 조건이 되어, 처음 보는 사람에 대한 성능을 정직하게 잴 수 없다. 그래서 조건은 Validation(S04)만으로 정하고 Test는 마지막에 한 번만 봤다.

## 10. 오늘 다시 이해한 핵심

- 같은 Pipeline이라도 데이터가 바뀌면 실험 결론이 달라질 수 있다. 교재 18개와 3팀 170개의 결과가 달랐고, 3팀 안에서도 Split 사람 구성이 바뀌자 Static vs Sequence 차이가 사라졌다.
- 3팀 Feature(45-D)는 Validation과 새로운 사람(S05) 모두에서 38-D보다 높았다. 이 결론은 사람 구성이 바뀌어도 유지됐다.
- Validation과 Test 점수 차이(0.74 vs 0.65)가 커서, 한 사람으로 낸 점수 하나를 모델 성능으로 믿기 어렵다. 다음에는 5명을 한 번씩 Test로 빼는 평가가 필요하다.
- 남은 혼동은 **팔로 물체를 다루는 동작**이다. 다음 단계에 필요한 것은 Feature를 더 늘리는 것보다 **물체 위치 정보와 데이터(Take·사람) 추가**다.
