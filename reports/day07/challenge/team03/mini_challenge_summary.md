# Day 07 Mini Challenge - 3팀 (무인 상점 이상행동 17개)

교재 Mini Challenge는 6개 동작 모델을 쓴다. 3팀은 6일차에 만든 **3팀 17개 동작 모델**로 같은 과정을 진행했다.

- 모델: `artifacts/team03/day06/baseline_38d/` (Random Forest 300 Tree, 38-D × 30 Step = 1,140 Feature, Train S01~S03)
- 45-D 모델은 실시간 코드에 팀 Feature 7개 계산이 없어서 쓰지 않았다. 교재 Runtime 입력과 모양이 같은 38-D 모델만 사용했다.
- 실행: `scripts/team03/team_day07_runtime.py`
  - 교재 42·44번 코드는 고치지 않는다.
  - 모델 경로, Action 이름표(`configs/team03_action_classes.csv`), 출력 폴더만 3팀 것으로 바꿔서 실행한다.
- 새 영상과 Webcam 입력은 Runtime 평가용이다. Train 데이터에 넣거나 모델을 다시 학습하지 않았다.


## 1. 내가 선택한 입력

Action ID: A007

Action Label: `sit_on_mat` (매트 위에 앉기)

선택 이유:
- 6일차 Test(S05)에서 정면·측면 모두 맞힌 동작이다.
- 영상이 길어서(8~10초) Warm-up 4초를 넘긴다.
- 3팀 동작은 짧은 것이 많아서(S05 34개 중 15개가 5초 미만) 기본 Window에서는 예측이 거의 안 나온다.
- 기절·넘어지기와 달리 IP Webcam 앞에서 안전하게 다시 할 수 있다.

직접 촬영한 MP4:
- 새 촬영본 `data/day07/challenge/team03/videos/S93_A007_T01.mp4`는 아직 녹화하지 않았다. 폰 Stream으로 녹화할 예정이다.
  ```bash
  python scripts/team03/team_record_clip.py --url "http://<폰 IP>:4747/video" --output data/day07/challenge/team03/videos/S93_A007_T01.mp4 --seconds 10
  ```
- 그 전에 Pipeline 연결을 확인하려고 **학습에 쓰지 않은 S05의 `S05_A007_T01.mp4`(정면)** 로 먼저 실행했다. 모델은 S01~S03으로만 학습했으므로 S05는 모델이 처음 보는 사람이다.

영상 길이: 10.21초 (60 FPS, 1080×1920, 609 Frame)


## 2. 저장영상 Baseline

```bash
python scripts/team03/team_day07_runtime.py video --source data/team_projects/team03/rgb/videos/S05/S05_A007_T01.mp4
```

Pipeline 확인: 영상 열림 → Pose 검출 → 4.0초에 WARMING_UP 끝 → 0.25초마다 예측해서 25회 발생

Raw Prediction 경향: `sit_on_mat` 23회, `vomit` 1회, `lie_back_side` 1회. 25번 중 23번이 정답이었다.

UNKNOWN이 많았는가: 25회 모두 UNKNOWN (`unknown_ratio = 1.0`)

Confidence 경향:
- 0.16 ~ 0.33, 평균 0.23
- 17개 Class라 아무렇게나 고르면 약 0.06이다. 정답 쪽으로 표가 꽤 모였지만, 기준 0.55에는 한 번도 닿지 않았다.

Stable 결과: 모두 UNKNOWN. 모델은 맞는 동작을 가장 비슷하게 봤지만 Threshold에서 전부 보류됐다.


## 3. 한 가지 Runtime 실험

변경한 값: Confidence Threshold

기본값: 0.55

실험값: 0.20

가설:
- 3팀 모델은 Class가 17개라 표가 여러 Class로 갈라진다. 그래서 Confidence가 원래 낮게 나온다.
- Threshold를 0.20으로 낮추면 UNKNOWN은 줄어들 것이다.
- 하지만 잘못된 Label(vomit, lie_back_side)도 함께 통과할 수 있다.

```bash
python scripts/team03/team_day07_runtime.py video --source data/team_projects/team03/rgb/videos/S05/S05_A007_T01.mp4 --confidence 0.20
python scripts/43_summarize_realtime_runs.py --input-dir reports/day07/challenge/team03/saved_video --output reports/day07/challenge/team03/saved_video_comparison.csv
```


## 4. Baseline VS Experiment

| 실행 | Threshold | Prediction | UNKNOWN 비율 | Raw 변경 | Stable 변경 | Stable 일치 비율 | 평균 Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.55 | 25 | 1.00 | 3 | 0 | 0.00 | 0.230 |
| Experiment | 0.20 | 25 | 0.32 | 3 | 2 | 0.64 | 0.230 |

실제로 달라진 점:
- UNKNOWN 비율이 1.00에서 0.32로 줄었다. Threshold 결과는 `sit_on_mat` 17회, UNKNOWN 8회였다.
- Stable은 5.5초부터 9.3초까지 `sit_on_mat`을 유지했다. 4.0~5.3초와 9.5초 이후는 Confidence가 0.20 아래라 UNKNOWN이었다.
- 잘못된 Label 2개(vomit 0.160, lie_back_side 0.187)는 Confidence가 0.20보다 낮아서 통과하지 못했다.
- Raw Prediction과 Confidence는 두 실행에서 똑같았다. Threshold는 모델의 판단을 바꾸지 않고 보류할지 말지만 바꾼다.

가설과 비슷했는가:
- UNKNOWN이 줄어든 것은 가설과 같았다.
- 이 영상에서는 틀린 Label이 통과하지 않았다. 다른 동작(예: 6일차에 서로 혼동한 push_two_hands ↔ table_strike)에서는 틀린 Label이 0.20을 넘을 수 있으니 따로 확인해야 한다.

한 문장 해석:
- 3팀 모델에는 교재 기준 0.55가 너무 높다. 0.20으로 낮추면 맞는 동작이 Stable까지 올라온다. 대신 이 값은 Class 수와 모델에 맞춰 다시 정해야 하는 값이다.
- 0.20으로 결과가 좋아진 것은 모델 성능이 좋아진 것이 아니다. Runtime 설정이 바뀐 결과다.


## 5. IP Webcam 결과

> 아직 하지 못했다. 같은 `sit_on_mat`을 폰 카메라 앞에서 기본 설정으로 한 번 수행할 예정이다.

```bash
python scripts/team03/team_day07_runtime.py webcam --url "http://<폰 IP>:4747/video"
# OpenCV 창 클릭 → 7 (A007 sit_on_mat) → 동작 → Q 종료
python scripts/45_analyze_ip_webcam_log.py --input-dir reports/day07/challenge/team03/ip_webcam --output reports/day07/challenge/team03/ip_webcam_summary.csv
```

3팀 Target 키:
- `1`~`9` → A001~A009
- `a`~`h` → A010~A017
- `0` Target 해제, `R` Buffer 초기화, `Q` 종료

Target: (아직 안 함)

Raw Prediction 경향: (아직 안 함)

UNKNOWN 경향: (아직 안 함)

Confidence 경향: (아직 안 함)

저장영상과 달랐던 점: (아직 안 함)


## 6. 오늘 이해한 내용

- Sliding Window는:
  - 계속 들어오는 Frame 중 최근 5초만 모아서 한 동작처럼 만든다.
  - 80%를 채워야 예측을 시작한다. 그래서 어떤 영상이든 첫 예측은 4초에 나왔다.
  - 3팀 동작 중 짧은 것(5초 미만)은 실시간에서 앞뒤 자세가 Window에 함께 섞인다.
- Confidence Threshold는:
  - 모델이 고른 Label을 믿을 만한지 거르는 기준이다.
  - Class가 17개면 Confidence가 원래 낮게 나오므로, 6개 Class용 기준(0.55)을 그대로 쓰면 맞는 예측도 전부 보류된다.
- Stabilization은:
  - 최근 5번 중 3번 이상 같은 Label이어야 Stable로 내보낸다.
  - 그래서 Threshold를 처음 넘은 5.0초가 아니라 5.5초에 Stable이 바뀌었다.
  - 끝날 때도 Threshold가 UNKNOWN이 된 9.0초보다 0.5초 늦게 UNKNOWN으로 돌아갔다.
- 저장영상과 실시간 입력의 차이는:
  - 저장영상은 같은 입력을 여러 번 넣어 설정 하나만 바꿔 비교할 수 있다.
  - 실시간은 매번 거리·각도·동작 속도가 달라진다.
  - 실시간은 처리 속도에 따라 Window 안의 Frame 수도 달라진다. (IP Webcam 테스트 후 결과로 확인 예정)

근거:
- `reports/day07/challenge/team03/saved_video_comparison.csv`
- `reports/day07/challenge/team03/saved_video/*.csv`
