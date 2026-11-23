# Day 06 Action Recognition Summary

## 1. Dataset

- Train Subject: S01
- Validation Subject: S02
- Test Subject: S03
- Train Clips: 6
- Validation Clips: 6
- Test Clips: 6
- Action Classes: 6
- Total Clips: 18
- 입력 검사 (`33_check_day06_inputs.py`): All Keypoint CSV OK, Action Coverage OK, Clip Leakage NONE → `Day06 input data: READY`

## 2. Frame Feature

- Normalized Coordinate: 34
- Joint Angle: 4
- Frame Feature Count: 38

## 3. Original Baseline

- Sequence Steps: 30
- Sequence Feature Count: 1140
- min_valid_frames: 10
- n_estimators: 300
- max_depth: null
- random_state: 42
- Sequence 생성 (`34_build_sequence_dataset.py`): Train / Validation / Test 각 6개, 유효 Frame 수(85, 125, 266, …)가 교재와 같음

## 4. Baseline Validation

- Accuracy: 0.5000 (교재 0.6667)
- Macro F1: 0.4444 (교재 0.5556)
- Correct Clips: 3
- Misclassified Clips: 3
  - walk_turn_walk → leg_raise_lower (Confidence 0.2133, 교재와 같음)
  - drink_return → wave (0.2267, 교재 0.2233)
  - wave → drink_return (0.2200) ← 교재에서는 정답이었던 Clip

교재와 다른 이유: 입력 Keypoint·Sequence·Static 결과·좌표만 결과는 교재와 같았고, S02 wave 하나만 달라졌다.
이 Clip은 drink_return과 wave의 Tree 투표가 거의 같았다(0.22 근처). Train이 6개뿐이라
라이브러리 버전(scikit-learn 1.9.1) 차이 같은 아주 작은 차이로도 동점이 뒤집힐 수 있다.
Validation이 6개라 Clip 하나가 Accuracy를 0.1667씩 바꾼다.

## 5. Static VS Sequence

- Static Feature Count: 38
- Static Accuracy: 0.6667
- Static Macro F1: 0.5556

- Sequence Feature Count: 1140
- Sequence Accuracy: 0.5000
- Sequence Macro F1: 0.4444

해석:

- 교재에서는 두 결과가 같았고, 이번 실행에서는 위의 wave 한 Clip 차이로 Sequence가 낮게 나왔다.
- 현재의 작은 Validation Dataset(6개)에서는 Static과 Sequence의 성능 차이를 판단할 수 없다.

## 6. Feature Set Comparison

- Coordinates + Angles
  - Feature Count: 1140
  - Accuracy: 0.5000
  - Macro F1: 0.4444

- Coordinates Only
  - Feature Count: 1020
  - Accuracy: 1.0000 (교재와 같음)
  - Macro F1: 1.0000

해석:

- 현재 S02 Validation에서는 Coordinates Only가 더 높은 결과를 보였다.
- 하지만 Validation Clip이 6개뿐이므로 일반적인 결론으로 확대하지 않는다.

## 7. Mini Challenge

우리 팀(3팀) Dataset 170개(Train S01~S03 102개 / Validation S04 34개 / Test S05 34개, 17개 동작)로 진행했다.
팀에서 사람 ID와 동작 이름을 통일한 뒤(2026-11-23) 다시 학습한 결과다.

- 변경한 조건: Frame Feature 38-D → 3팀 45-D (상체 기울기, BBox 비율, 골반-바닥 높이, 양발목·양손목 속도)
- 변경 전: 1,140 Feature
- 변경 후: 1,350 Feature
- 실험 전 가설: 눕기·앉기·차기는 몸 기울기·높이·속도 정보가 도움이 될 것이다

- Baseline Accuracy: 0.6176
- Baseline Macro F1: 0.6235

- 변경 후 Accuracy: 0.7353
- 변경 후 Macro F1: 0.7196

- 예측이 바뀐 Clip: 8개 (앞차기·잡고 흔들기 2개·몸통 박치기·샷건 5개가 정답으로 바뀜, 뒤로 넘어지기 1개가 새로 틀림)
- 결과 해석: 속도 Feature가 빠르게 움직이는 동작을 나누는 데 도움이 됐지만, 물체 없이 팔로 하는 동작(밀치기·잡고 흔들기·샷건)끼리의 혼동은 남았다.
- 같은 방식으로 동작 구간만 사용(0.7353 / 0.7098), 정면만(0.7647 / 0.7157), 측면만(0.5882 / 0.4902)도 비교했다. Static(0.6471 / 0.6020)은 Accuracy가 조금 높고 Macro F1은 낮아 Sequence와 차이가 거의 없었고, 좌표만(0.6176 / 0.5905)은 Macro F1이 낮았다.
- 자세한 내용: `reports/day06/challenge/mini_challenge_result.md`

## 8. Final Selected Config

본수업 (교재 18개)

- Sequence Steps: 30
- Sequence Feature Count: 1140
- min_valid_frames: 10
- n_estimators: 300
- max_depth: null
- random_state: 42

최종 조건을 선택한 이유:

- 교재 40번은 Baseline 모델(`rf_action_baseline.joblib`)로 Test를 평가한다. Coordinates Only가 Validation에서 높았지만 Validation이 6개뿐이라 그 결과만으로 조건을 바꾸지 않고 Baseline을 유지했다.

3팀 Mini Challenge

- 3팀 45-D, 30 Step, 1,350 Feature, 나머지는 Baseline과 같음
- 선택 이유: Test 전에 정한 기준(Validation Macro F1 최고)에 따라 선택

## 9. Final Test

본수업

- Test Subject: S03
- Test Clips: 6
- Accuracy: 1.0000
- Macro F1: 1.0000
- Correct Clips: 6
- Misclassified Clips: 0

3팀 Mini Challenge

- Test Subject: S05
- Test Clips: 34
- Accuracy: 0.6471 (참고: 38-D Baseline 0.5882)
- Macro F1: 0.6020 (참고: 38-D Baseline 0.5652)
- Correct Clips: 22
- Misclassified Clips: 12 (휘두르기·밀치기·앞차기·샷건·잡고 흔들기 → 몸통 박치기 6개, 밀치기 → 샷건·잡고 흔들기 → 밀치기·두드리기 ↔ 던지기 4개, 옆으로 쓰러지기 → 바닥에 앉기 2개)
- Validation(0.7353)보다 크게 낮았다. 이름 통일 전 Split에서는 같은 방법으로 Test 0.7647이었다. Test 사람에 따라 점수가 크게 흔들린다.

Test 결과를 본 뒤 조건을 다시 변경하지 않은 이유:

- Test를 보고 조건을 바꾸면 Test 사람에게 맞춘 결과가 되어, 처음 보는 사람에 대한 성능을 정직하게 확인할 수 없기 때문이다.

## 10. 현재 Dataset의 한계

- 본수업: 3 Subjects × 6 Actions × 1 Take의 작은 교육용 Dataset이다. Action당 Subject별 Clip이 1개뿐이다.
- 3팀: 5 Subjects × 17 Actions × 2방향 × 1 Take. Validation·Test가 각각 한 사람이라 사람 차이의 영향이 크다. 정상행동 영상이 없어 "정상 vs 이상"은 아직 평가하지 못한다.
- 현재 Accuracy를 현업 성능으로 해석하지 않는다.

## 11. 6일차에서 배운 핵심

- Keypoint CSV → 38-D Frame Feature → 30 Step Resampling → 1,140개 숫자로 Clip 하나를 표현하고 Random Forest로 분류했다.
- Train으로 학습, Validation으로 조건 비교, Test는 마지막에 한 번만 사용한다.
- 같은 실험도 데이터에 따라 결론이 달라진다 (교재 18개 vs 3팀 170개). 작은 Validation 결과를 일반화하지 않는다.
- 3팀 데이터에서는 팀 Feature(45-D)가 새로운 사람(S05)에서도 성능을 올렸다. 다만 Test 점수 자체는 누가 Test 사람이냐에 따라 10%p 넘게 달라져, 사람을 돌아가며 평가해야 한다.

## 12. 다음 수업

6일차에서는 완성된 Clip 전체를 이용하여 Action을 분류했다.

7일차에서는 실시간 Frame을 계속 모아 Sliding Window를 만들고
실시간 Action Recognition으로 연결한다.
