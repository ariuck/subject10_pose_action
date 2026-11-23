# Day 07 Sliding Window & IP Webcam Realtime Summary

## 1. 사용한 모델

| 항목 | 교재 실습 | 3팀 Mini Challenge |
|---|---|---|
| Model | `artifacts/day06/rf_action_baseline.joblib` | `artifacts/team03/day06/baseline_38d/rf_action_baseline.joblib` |
| Action Classes | 6개 (bend_return, leg_raise_lower, walk_turn_walk, drink_return, sit_stand, wave) | 17개 (무인 상점 이상행동 A001~A017) |
| Train | S01, 6 Clip | S01~S03, 102 Clip (정면·측면) |
| Sequence Steps | 30 | 30 |
| Frame Feature | 38-D (정규화 좌표 34 + 관절각도 4) | 38-D (같음) |
| Final Feature Count | 1,140 | 1,140 |


## 2. 오늘 만든 Realtime Pipeline

```text
Saved MP4 / IP Webcam
→ YOLO Pose
→ 38-D Frame Feature
→ Sliding Window (최근 5초, 80% 이상 차면 예측)
→ Sequence Resampling (30 Step)
→ Random Forest (0.25초마다)
→ Confidence Threshold (0.55 미만이면 UNKNOWN)
→ Stabilization (최근 5개 중 3표)
→ Runtime Result
```


## 3. 저장영상에서 확인한 핵심 결과

- Runtime Pipeline은 정상적으로 동작했는가:
  - 정상이었다.
  - 41번 입력 검사 결과는 `READY`였다.
  - Runtime-ready 영상은 5 / 6개였다. wave는 3.76초라 Warm-up 4초보다 짧다.
- Raw Prediction의 특징:
  - 교재 모델의 walk_turn_walk 영상은 Raw 21회 중 정답이 1회뿐이었다. 대부분 drink_return이었다.
  - wave는 Window를 3초·2초로 줄이자 예측이 나왔고 Raw도 대부분 wave였다.
- UNKNOWN이 나타난 주요 이유:
  - 교재 모델의 Confidence가 0.22~0.26으로 기준 0.55보다 훨씬 낮았다.
  - 모델이 한 사람의 6개 Clip으로만 학습해서 표가 갈라진 것이 원인이라고 본다.
- Window / Threshold / Stabilization 중 가장 눈에 띈 변화:
  - Window였다. 5초에서는 wave 예측이 0회였는데, 3초에서 5회, 2초에서 8회로 늘었다.
  - Threshold(0.40 / 0.70)와 Buffer(3 / 7)는 Confidence가 모두 0.40 아래라 결과가 바뀌지 않았다.
- 자세한 결과: `runtime_comparison.md`, `fps_test.md`


## 4. IP Webcam에서 확인한 핵심 결과

- Stream 연결:
  - 폰 DroidCam(`/video`, 1280×720)으로 연결했다.
  - 44번이 Stream에 연결(`Connected: 1280 x 720`)되는 것까지 확인했다.
- Target 기록: 아직 안 함
- Prediction 발생: 아직 안 함
- Raw / Threshold / Stable에서 관찰한 특징: 아직 안 함
- 저장영상과 달랐던 점: 아직 안 함
- 남은 일:
  - 교재 6개 동작을 Target 키 1~6으로 하나씩 수행하고 `Q`로 종료한다.
  - 45번으로 Log를 분석한다.


## 5. Mini Challenge

- 선택한 Action: 3팀 A007 `sit_floor` (바닥에 앉기)
- 직접 촬영한 MP4:
  - 새 촬영본 `S93_A007_T01.mp4`는 아직 없다.
  - 학습에 쓰지 않은 S05의 `S05_A007_T01.mp4`로 먼저 확인했다.
  - 팀에서 사람 ID·동작 이름을 통일한 뒤 다시 학습한 모델로 다시 실행한 결과다.
- 변경한 Runtime 변수: Confidence Threshold 0.55 → 0.20
- Baseline과 Experiment에서 달라진 점:
  - Raw는 26번 중 20번이 정답이었고, 두 실행에서 같았다. 틀린 6번은 앉기 전·일어난 뒤 자세가 섞인 앞뒤 구간이었다.
  - UNKNOWN 비율이 1.00에서 0.35로 줄었다.
  - Stable 일치 비율은 0.00에서 0.65가 됐다.
  - 틀린 Label 6개는 0.20도 넘지 못해서 통과하지 않았다.
- IP Webcam 결과와 저장영상 결과의 차이: 아직 안 함
- 자세한 결과: `challenge/team03/mini_challenge_summary.md`


## 6. 현재 Pipeline의 한계

- **Threshold가 모델에 따라 맞지 않는다**: 기본 Threshold 0.55는 두 모델 모두에 너무 높았다. 교재 모델은 Confidence가 0.2대이고, 3팀 모델은 17개 Class라 원래 낮게 나온다.
- **짧은 동작을 놓친다**: Window 5초에 80%를 채워야 하므로 5초보다 짧은 동작(3팀 S05 영상 34개 중 13개, 전체 170개 중 74개)은 앞뒤 자세와 섞이거나 예측이 아예 나오지 않는다.
- **3팀 최종 45-D 모델을 쓰지 못했다**: 실시간 코드는 38-D만 만든다. 팀 Feature 7개(속도 등)를 Frame마다 계산하는 부분이 없다.
- **사람 한 명만 가정한다**: Confidence가 가장 높은 사람 한 명만 본다. 무인 상점처럼 여러 사람이 나오면 누구의 동작인지 구분하지 못한다.
- **카메라 연결이 불안정하다**: 폰 화면이 꺼지거나 다른 프로그램이 Stream을 잡고 있으면 영상이 끊기거나 안내 화면이 들어온다.


## 7. 한 문장 결론

- 6일차 모델을 계속 들어오는 Frame에 연결하는 Realtime Pipeline은 정상으로 동작했다.
- 다만 Confidence가 낮아서 기본 설정에서는 대부분 UNKNOWN으로 보류됐다.
- 3팀 모델은 Threshold를 0.20으로 낮추자 맞는 동작이 Stable까지 올라왔다. 그래서 Threshold는 모델과 Class 수에 맞춰 정해야 한다는 것을 확인했다.
