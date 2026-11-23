# Subject 10 - Pose Estimation & Action Analysis

YOLO Pose로 찾은 관절을 Feature로 바꾸고, 시간 변화를 Random Forest로 분류해서 동작을 인식하는 프로젝트입니다.
수업 데이터(6개 동작)로 Pipeline을 만들고, 같은 Pipeline을 **3팀 무인 상점 이상행동 17개 동작**(5명 × 17동작 × 정면·측면 = 170개 영상)에 적용했습니다.

- 진행하면서 겪은 문제와 해결 과정, 개선 결과: [`reports/improvement_log.md`](reports/improvement_log.md)

## Day 01 - Pose 추론 환경

```text
JPG / MP4 / Phone Camera → Frame → YOLO Pose → Person BBox → COCO 17 Keypoint → Skeleton
```

- 환경: Python 3.12, PyTorch 2.15 + CUDA, RTX 3060
- COCO8-Pose 이미지 8장과 Pose Demo MP4(30 FPS, 60 Frame)에서 사람과 17개 관절을 찾았다.
- 사람이 겹친 이미지에서 차이가 컸다.
  - 049번: 잘 찾음
  - 110번: 8명이 겹쳐서 불안정함
- 직접 촬영한 4가지 자세를 비교했다.
  - 촬영한 자세: 옆으로 서기, 쪼그려 앉기, 엎드리기, 벽에 붙어 웅크리기
  - 가장 잘 된 자세: 쪼그려 앉기
  - 가장 불안정한 자세: 벽에 붙어 웅크리기 (사람 검출 실패)

## Day 02 - Pose 품질 분석

```text
이미지 · MP4 → YOLO Pose → BBox + 17 Keypoint → Keypoint Confidence → Threshold → 조건별 · Frame별 품질 비교
```

**촬영 조건별 이미지 6장**

| 조건 | 평균 Keypoint Confidence | LOW 관절 |
|---|---|---|
| 정면 | 0.975 | 0 |
| 측면 | 0.754 | 4 (오른쪽 눈·귀·팔꿈치·손목) |
| 먼 거리 (BBox 면적 0.300 → 0.149) | 0.971 | 0 |

- 측면이 6장 중 가장 불안정했다. 반면 이번 거리 범위에서는 사람이 절반 크기로 작아져도 품질이 거의 유지됐다.
- 정면 이미지에서 Threshold를 올려 보니 남는 관절이 줄었다.
  - 0.70: 17개 모두
  - 0.90: 오른쪽 귀 제외
  - 0.95: 양쪽 귀 제외
- MP4에서는 60 Frame 모두 사람을 찾았다.
  - 약 0.8초(25~26번 Frame)에서 얼굴 주변 Confidence가 떨어졌다.
  - 영상을 확인해 보니 바닥 자세에서 얼굴 방향이 바뀌는 장면이었다.

**직접 촬영한 동작 영상 36개 (1080×1920, 60 FPS, 정면·측면 카메라 2대)**

| 카메라 | 평균 Confidence | LOW 관절 |
|---|---|---|
| 정면 | 0.96~0.98 | 거의 없음 |
| 측면 | 0.82~0.86 | 거의 모든 Frame에서 4개 (반대쪽 눈·귀·손목) |

- 바닥에 눕거나 엎드린 동작이 가장 불안정했다. 평균 0.724, LOW 관절 최대 11개.
- 미검출: 벽에 붙어 바닥에 웅크린 장면에서 1,512 Frame 중 171 Frame
- 오검출: 화면 구석의 물체를 `person 0.35`로 검출했다. Person Confidence 기준 0.25를 넘었기 때문이다.
- 측면의 가려진 팔
  - Threshold 0.30에서는 앞쪽 팔에 겹쳐 그려졌다.
  - 0.50에서는 제외됐다.

## Day 03 - Joint Angle, Normalization & Frame Feature

```text
17 Keypoint + Confidence → 관절각도 (Confidence 낮으면 SKIP)
→ Hip Center 이동 + Shoulder Width 정규화
→ Normalized 좌표 34개 + 관절각도 4개 = 38-D Frame Feature → 영상의 각 Frame에 반복
```

**관절각도**

| 이미지 | 왼팔꿈치 | 오른팔꿈치 | 왼무릎 | 오른무릎 |
|---|---|---|---|---|
| 팔 올리기 | 132.50° | 125.38° | 174.53° | 173.54° |
| 앉기 | 133.50° | 136.46° | 97.88° | 117.49° |
| 허리 숙이기 | **SKIP** | 158.53° | 174.52° | 173.93° |

- 앉기는 무릎 각도로 서 있는 자세와 확실히 구분됐다.
- 팔꿈치 각도는 서기와 앉기에서 비슷했다. 관절각도는 "얼마나 굽혔는지"만 보여 준다.

**정규화 효과**

| 비교 | Pixel 평균 관절 거리 | Normalized 평균 관절 거리 |
|---|---:|---:|
| 같은 사람·같은 자세, 가까이 / 멀리 (직접 촬영) | 163.60 | 0.086 |
| 다른 사람·비슷한 팔 올리기 | 69.40 | 0.339 |
| 왼팔 올리기 vs 오른팔 올리기 (좌우 반전 전 → 후) | - | 0.609 → 0.121 |

- 정규화를 하면 사람의 위치·크기 차이는 줄었다. 하지만 사람마다 체형과 팔 각도가 달라서, 자세 자체의 차이는 남았다.
- 좌우 반전은 x 부호만 바꾸면 안 된다. Left / Right 관절 번호도 바꿔야 손목·팔꿈치 차이가 크게 줄었다.
- 손목이 어깨보다 위인지 보는 규칙(`wrist_y - shoulder_y < 0`)에서는 앉아서 팔을 든 사진도 True가 나왔다. Feature 하나로는 서로 다른 자세를 구분하기 어렵다.
- 영상 60 Frame을 모두 38-D로 바꿨다.
  - 왼팔꿈치 각도: 140° → 116°
  - 손목-어깨 높이 차: 10.99 → 0.747
  - 팔을 들어 올리는 흐름이 숫자로 보였다.

**문제 해결 1: Confidence가 높은데 관절 위치가 틀림**
- 상황: 허리를 숙인 측면 사진에서 왼팔꿈치·손목이 실제 왼팔이 아니라 오른팔 위에 겹쳐 찍혔다.
- 원인: Confidence가 0.817 / 0.884로 높아서 Confidence 검사로는 걸러지지 않았다.
- 해결: **겹쳐 찍힌 팔·다리 검사**를 추가했다. 한쪽 관절이 반대쪽 팔다리 선에 거의 붙어 있으면(overlap < 0.05) Confidence가 낮은 쪽 각도를 SKIP한다.
- 근거: `reports/day03/bend_pose_left_right_check.md`

**문제 해결 2: 측면에서 어깨 폭 정규화가 무너짐 (3팀 17개 동작, 36개 영상)**

| 확인 항목 | 결과 |
|---|---|
| 관절각도 | 무릎은 서기 161~173°, 눕기 159~164° → 둘 다 펴져 있어서 서기와 눕기를 구분 못 함 |
| 어깨 폭 정규화 (측면 눕기, 가까이 / 멀리) | 두 어깨가 겹쳐 어깨 폭이 18.99px까지 작아짐 → 차이 `8.199` |
| **상체 길이 정규화** | 눕기 가까이 / 멀리 `0.438`, 서기 가까이 / 멀리 `0.167`, 눕기 vs 서기 `1.209` |
| 영상 (뒤로 기절하기, 808 Frame) | 상체 기울기: 서기 약 7° → 누움 77~93° (3.3~10.6초 유지) → 서기 약 1° |

- 관절각도만으로는 서기·눕기를 구분할 수 없었다. 그래서 상체 기울기, 골반-발목 높이, BBox 비율 Feature를 설계했다.
- 동작마다 잘 보인 Feature가 달랐다.
  - 내려치기: 팔 올림
  - 던지기: 손목 속도
  - 앞차기: 발목 속도
  - 몸통박치기: 골반 속도
- 흔들기·밀치기처럼 물체와 닿는 동작은 Pose만으로 부족해서 객체 검출이 필요하다.
- 결과: `reports/day03/challenge/feature_design.md`, `reports/day03/challenge/team3_action_summary.md`

```text
src/team_features.py   상체 기울기, 골반-발목 높이, BBox 비율, 팔 올림, 발목 높이 차, 속도, 자세 규칙
scripts/team3/         자세 판정, 팀 Feature 추출, 상체 길이 정규화 비교, 동작별 요약
```

## Day 04 - Action Sequence & 3팀 45-D Dataset

```text
Action Clip → Frame → YOLO Pose → 17 Keypoint → Normalization + Joint Angle
→ Frame Feature → 시간 순서 유지 → Sequence → Action Label
```

**수업 데이터 6개 동작 (824 Frame, Pose·Feature 성공 824 / 824)**

| 동작 | 평균 Confidence | 시간에 따라 잘 변한 Feature |
|---|---|---|
| wave | 0.972 | 오른쪽 Wrist y `-0.03 → -3.05 → -0.04`, 오른쪽 Elbow `177° → 116° → 173°` |
| drink_return | 0.954 | 오른쪽 Wrist y `-0.05 → -2.10 → -0.02`, 오른쪽 Elbow `170° → 61° → 172°` |
| bend_return | 0.958 | 양쪽 Wrist y `0 → 0.95 → 0`, Hip Center y `249 → 258 → 250` |
| sit_stand | 0.974 | Knee `160° / 149° → 178° / 173°`, Hip Center y `286 → 250` |
| walk_turn_walk | 0.888 | Hip Center x `472 → 279 → 462` |
| leg_raise_lower | 0.968 | 대표 Frame 5개로는 안 보임 → 9개로 늘리자 오른쪽 Ankle y `2.4 → 1.81` (1.84초) |

- sit_stand 영상은 앉은 자세로 시작해서 일어서는 동작만 있었다. 대표 Frame과 Sequence 값이 모두 이렇게 나와서 그대로 기록했다.
- walk_turn_walk의 인접 Row 변화량(17.478)은 성능 점수가 아니다. 방향 전환, 정규화 변화, Keypoint 흔들림이 섞인 값이다.

**3팀 17개 동작 (정면·측면 36개 영상)**
- Pose 성공률: 18,950 Frame 중 18,742 (98.9%)
  - 실패 Frame은 모두 바닥에서 몸을 말거나 엎드린 장면이었다.
- 같은 이름의 정면·측면 결과가 서로 덮어써졌다.
  - Label에 `_front` / `_side`를 붙여 36개를 따로 저장했다.
- 휴대폰 영상은 영상 정보의 Frame 수가 실제보다 1~6개 많아서 대표 Frame 추출이 멈췄다.
  - 실제로 읽히는 Frame을 기준으로 뽑도록 처리했다(108장).
- 측면에서는 어깨 폭이 작아져 정규화 좌표가 튀었다.
  - 인접 Row 변화량 중앙값: 정면 0.227, 측면 4.475
- 뒤로 기절하기 vs 앞으로 넘어지기
  - 시작·끝 자세가 같아서 중간의 시간 변화로만 구분됐다.
  - 기절: 골반이 1초 안에 약 190px 내려가 머물렀다.
  - 넘어지기: 골반이 1초에 약 380px 앞으로 이동했다.

**3팀 RGB 영상 170개 → 45-D Sequence Dataset**

```text
S0x_A0nn_T0t.mp4 (5명 × 17동작 × 정면·측면, 실명 대신 S01~S05)
→ Clip Manifest → Keypoint Dataset (Pixel 좌표 + Confidence + BBox)
→ 자동 QA (AUTO_PASS 166 / CHECK 4) → Human QA (PASS 170)
→ 동작 시작·끝 구간 (in_action 표시)
→ 38-D + 3팀 7개 = 45-D Sequence 170개
```

| 추가 Feature (+7) | 의미 |
|---|---|
| `torso_tilt_deg` | 상체 기울기 (0° 서기, 90° 수평) |
| `bbox_aspect_ratio` | 사람 BBox 가로/세로 비율 |
| `hip_floor_height` | 골반과 바닥(BBox 아래 끝)의 상대 높이 / 상체 길이 |
| `left/right_ankle_speed` | 양발목 속도 (상체 길이/초, 0.1초 간격으로 떨림 완화) |
| `left/right_wrist_speed` | 양손목 속도 (문·물체 위치는 영상에 없어서 대체) |

- 70,159 Frame 중 Pose 성공 69,926 (99.7%), 45-D 성공 68,813
- 측면에서 눕기·쓰러짐은 서서 하는 동작과 잘 구분됐다.

  | 동작 | BBox 비율 | 골반-바닥 높이 |
  |---|---|---|
  | 눕기·쓰러짐 | 2.4~3.2 | 0.3~0.5 |
  | 서서 하는 동작 | 약 0.5 | 약 1.6 |

- 보고서: `reports/team_projects/team03/team03_data_preparation_report.md`

## Day 05 - Keypoint Dataset QA & Subject-wise Split

```text
MP4 → Keypoint Dataset → 자동 QA → Human QA → Approved Clips → Subject-wise Split → Leakage check
```

| | 수업 데이터 (6동작 × 3명) | 3팀 (17동작 × 5명 × 정면·측면) |
|---|---|---|
| 자동 QA | AUTO_PASS 13 / CHECK 5 | AUTO_PASS 166 / CHECK 4 |
| Approved | 18 | 170 |
| Train / Validation / Test | S01 / S02 / S03 | S01~S03 (102) / S04 (34) / S05 (34) |
| Leakage check | OK | OK |

- 수업 데이터의 CHECK 5개
  - 원본 영상을 확인해 보니 **옷걸이에 걸린 외투**를 두 번째 사람으로 오검출한 것이었다.
  - 그래서 모두 PASS로 판정했다.
- Frame을 무작위로 나눠 보니(Train 1,946 / Test 487 Frame) 3명 모두, Clip 18개 모두가 Train과 Test에 함께 들어갔다.
  - 같은 사람의 같은 동작으로 시험을 보는 셈이라, 평가는 사람(Subject) 기준으로 나눴다.
- 3팀 동작은 동작마다 중요한 관절이 달랐다. 그래서 Keypoint 추출·QA 기준을 17개 동작에 맞게 만들었다. 경로 처리 오류도 함께 고쳤다. (`scripts/team03/`)
- 측면 영상이 QA에서 불리하게 걸렸다. 판단을 돕는 참고 열 3개를 추가해서 CHECK 원인을 바로 설명할 수 있게 했다.
- 결과: `reports/day05/challenge/mini_challenge_result.md`

## Day 06 - Action Recognition

```text
Keypoint CSV → Frame Feature → Sequence Resampling (30 Step)
→ Random Forest (300 Tree) → Validation → 오분류 분석 → Feature 비교 → Final Test
```

**수업 데이터 (Train 6 / Validation 6 / Test 6 Clip)**

| 비교 | Validation Accuracy | Macro F1 |
|---|---:|---:|
| Sequence (Baseline) | 0.5000 | 0.4444 |
| Static | 0.6667 | 0.5556 |
| 좌표만 | 1.0000 | 1.0000 |

- 틀린 3개
  - walk_turn_walk → leg_raise_lower
  - drink_return → wave
  - wave → drink_return
- wave Clip 하나는 두 동작의 투표가 0.22 근처로 거의 같아서, 작은 차이에도 결과가 뒤집혔다.
- Train이 6 Clip뿐이라 Static과 좌표만 쓴 조건이 더 높게 나온 결과를 일반적인 경향으로 볼 수 없었다. Test(S03)는 6 / 6이었다.

**3팀 17개 이상행동 (Train 102 / Validation 34 / Test 34)** - 한 번에 한 조건만 바꿔 Validation으로 비교했다.

| 조건 | Feature | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|---:|
| Baseline (38-D) | 1,140 | 0.7353 | 0.7373 |
| Static (가운데 Frame) | 38 | 0.5882 | 0.5941 |
| 좌표만 | 1,020 | 0.6765 | 0.6728 |
| **3팀 45-D** | 1,350 | 0.7647 | **0.7725** |
| 동작 구간만 | 1,140 | 0.7941 | 0.7588 |
| 정면만 / 측면만 | 1,140 | 0.6471 / 0.6471 | 0.5882 / 0.5529 |

- 데이터가 늘어나자 수업 데이터와 반대로 나왔다. Sequence가 Static보다, 각도를 넣은 것이 좌표만 쓴 것보다 높았다.
- 남은 혼동은 물체 없이 팔로 하는 동작끼리였다. (밀치기 ↔ 흔들기, 내려치기·던지기·휘두르기)
- Feature를 한 단계씩 늘려 본 Validation Macro F1

  | 단계 | Macro F1 |
  |---|---:|
  | 38-D | 0.7373 |
  | + 자세 3개 | 0.7647 |
  | + 발목 속도 | 0.7922 |
  | + 손목 속도 | 0.7725 |

  - 손목 속도는 오히려 역효과였다. 팔 동작은 모두 손목이 빨라서 서로를 구분하는 데 도움이 안 됐다.
- 최종 조건은 45-D로 정했다. Test를 보기 전에 정한 기준(Validation Macro F1 최고)으로 골랐다.
- **Final Test (S05)**

  | 모델 | Accuracy | Macro F1 | 맞힌 개수 |
  |---|---:|---:|---:|
  | 45-D (최종) | 0.7647 | 0.7529 | 26 / 34 |
  | 38-D Baseline | 0.6765 | 0.6630 | - |

- 결과: `reports/day06/challenge/mini_challenge_result.md`, `reports/team_projects/team03/feature_expansion_log.md`

```text
scripts/team03/team_day06_baseline.py   조건 비교 실험 · 모델과 학습 정보 저장
```

## Day 07 - Sliding Window Realtime Action Recognition

학습한 모델을 저장영상과 스마트폰 카메라 Stream(1280×720)의 연속 Frame에 연결했다.

```text
Saved MP4 / Phone Camera Stream
→ YOLO Pose → Frame Feature
→ Timed Sliding Window (최근 5초, 80% 이상 차면 예측)
→ 30 Step Resampling → Random Forest (0.25초마다)
→ Confidence Threshold (미만이면 UNKNOWN)
→ Stabilization (최근 5개 중 3표) → Stable Action
```

**수업 모델로 저장영상 실험 (S02)**

| 실험 | Prediction | UNKNOWN 비율 | 평균 Confidence |
|---|---:|---:|---:|
| wave, Window 5초 | 0 | - | - |
| wave, Window 3초 / 2초 | 5 / 8 | 1.0 / 1.0 | 0.249 / 0.260 |
| walk_turn_walk, 기본 (Threshold 0.55, Buffer 5) | 21 | 1.0 | 0.231 |
| walk_turn_walk, Threshold 0.40 / 0.70 | 21 / 21 | 1.0 / 1.0 | 0.231 |
| walk_turn_walk, Buffer 3 / 7 | 21 / 21 | 1.0 / 1.0 | 0.231 |

- wave 영상은 3.76초라서 Window 5초의 80%(4초)를 채우지 못했다. 그래서 예측이 0번이었다. Window를 줄이자 예측이 나왔고 Raw는 대부분 wave였다.
- walk_turn_walk의 Raw는 21번 중 정답이 1번뿐이었다(대부분 drink_return).
- Confidence가 0.2대라서 Threshold와 Buffer를 바꿔도 결과는 모두 UNKNOWN이었다. 한 사람의 6개 Clip으로 학습한 모델이라 표가 여러 동작으로 갈라진 것으로 본다.

**3팀 모델 - A007 매트 위에 앉기 (학습에 쓰지 않은 S05 영상, 10.2초)**

| Threshold | UNKNOWN 비율 | Stable 일치 비율 | Raw 정답 |
|---:|---:|---:|---:|
| 0.55 (기본) | 1.00 | 0.00 | 23 / 25 |
| 0.20 | 0.32 | 0.64 | 23 / 25 |

- Raw 예측은 대부분 맞았다. 하지만 17개 Class 모델은 Confidence가 원래 낮아서(0.16~0.33), 기본 기준으로는 전부 보류됐다.
- Threshold는 모델의 판단을 바꾸지 않고 내보낼지만 정한다. 그래서 기준값은 모델과 Class 수에 맞춰 정해야 한다.
- 0.20에서는 5.5~9.3초 구간이 Stable `sit_on_mat`이 됐다. 틀린 Raw 2개는 0.20을 넘지 못했다.
- 3팀 S05 영상 34개 중 15개가 5초보다 짧았다. 짧은 이상행동을 실시간으로 잡으려면 Window 조정이나 동작 구간 검출이 필요하다.
- 처리 속도
  - 수업 영상(640×480): 37 FPS
  - 3팀 영상(1080×1920): 22 FPS
  - 입력 해상도의 영향이 컸다.
- 결과: [`mini_challenge_summary.md`](reports/day07/challenge/team03/mini_challenge_summary.md), [`runtime_comparison.md`](reports/day07/runtime_comparison.md), [`fps_test.md`](reports/day07/fps_test.md)

```text
scripts/team03/team_day07_runtime.py   3팀 모델 · 17개 동작 이름표로 저장영상 / 카메라 실시간 실행
scripts/team03/team_record_clip.py     폰 Stream으로 평가용 영상 녹화
```
