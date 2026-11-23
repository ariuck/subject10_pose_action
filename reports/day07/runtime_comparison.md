# Day 07 Runtime Check

## 기본 Runtime 설정

Window Seconds: 5.0

Confidence Threshold: 0.55

Stabilization Buffer: 5 (최소 3표)

Prediction 주기: 0.25초 / Warm-up: Window의 80% = 4초


## 저장영상에서 확인한 내용

사용한 영상: `S02_A006_T01.mp4` (wave, 3.76초), `S02_A003_T01.mp4` (walk_turn_walk, 9.64초)

Prediction 발생 여부:
- wave는 기본 설정에서 Prediction 0회. 영상이 3.76초라 Warm-up 4초를 못 채웠다. (41번 검사에서도 `runtime_ready = 0`)
- walk_turn_walk는 4초부터 0.25초마다 예측해서 21회 발생했다.

UNKNOWN이 많았는가: 7번 실행 모두 `unknown_ratio = 1.0`. 한 번도 Threshold를 넘지 못했다.

관찰한 특징:
- walk_turn_walk 영상의 Raw는 drink_return 14회, leg_raise_lower 5회, wave 1회, walk_turn_walk 1회였다. 정답은 1번뿐이었다.
- Confidence는 0.22~0.26으로, 6개 Class일 때 아무렇게나 고른 값(약 0.17)보다 조금 높은 정도였다.
- Window를 3초·2초로 줄이자 wave에서도 예측이 5회·8회 나왔고 Raw는 대부분 wave였다. 그래도 Confidence가 0.25~0.26이라 UNKNOWN이었다.
- Threshold 0.40 / 0.70, Buffer 3 / 7로 바꿔도 결과는 모두 UNKNOWN이었다. Confidence가 0.40보다 낮아서 Threshold를 넘은 예측이 없었고, Buffer에는 UNKNOWN만 쌓였다.
- 원인은 모델 쪽이라고 생각한다. 6일차 모델은 S01 한 사람의 6개 Clip으로만 학습했다. 나무 300개의 표가 여러 Class로 갈라져서 Confidence가 낮게 나온다.

| 실행 | 바꾼 값 | Prediction | UNKNOWN 비율 | Raw 변경 | 평균 Confidence |
|---|---|---:|---:|---:|---:|
| wave 기본 | - | 0 | - | - | - |
| wave Window 3초 | window 3 | 5 | 1.0 | 2 | 0.249 |
| wave Window 2초 | window 2 | 8 | 1.0 | 2 | 0.260 |
| walk_turn_walk 기본 | - | 21 | 1.0 | 5 | 0.231 |
| walk_turn_walk Threshold 0.40 | conf 0.40 | 21 | 1.0 | 5 | 0.231 |
| walk_turn_walk Threshold 0.70 | conf 0.70 | 21 | 1.0 | 5 | 0.231 |
| walk_turn_walk Buffer 3 | buffer 3 | 21 | 1.0 | 5 | 0.231 |
| walk_turn_walk Buffer 7 | buffer 7 | 21 | 1.0 | 5 | 0.231 |

근거: `reports/day07/realtime_run_summary.csv`, `reports/day07/saved_video/*.csv`


## IP Webcam에서 확인한 내용

> 아직 6개 동작 테스트를 하지 못했다. 폰 카메라(DroidCam) 연결과 Stream 수신까지만 확인했다.

사용한 Action: (아직 안 함)

Raw Prediction: (아직 안 함)

Confidence 경향: (아직 안 함)

Threshold 결과: (아직 안 함)

Stable 결과: (아직 안 함)

연결하면서 확인한 것:
- DroidCam 영상 주소는 `http://<폰 IP>:4747/video`이고 1280×720으로 들어왔다.
- DroidCam은 한 번에 한 프로그램만 영상을 받는다. 브라우저에서 영상을 열어 둔 채 44번을 실행하면 사람 대신 "다른 곳에서 사용 중" 안내 화면이 들어온다.
- 44번은 `Q`로 끝내야 Log CSV가 저장된다. 터미널에서 강제로 멈추면 저장되지 않았다.


## 저장영상과 IP Webcam의 차이

- 저장영상(42번)은 영상 안의 시간(`frame_id / FPS`)으로 Window를 채운다. 처리가 느려도 5초 Window에는 5초 분량의 Frame이 모두 들어간다. IP Webcam(44번)은 실제로 흐른 시간(`perf_counter`)으로 채운다. 처리 속도가 느리면 5초 Window에 들어가는 Frame 수가 줄어든다.
- 저장영상은 같은 입력을 여러 번 넣을 수 있어 설정 비교가 쉽다. IP Webcam은 매번 사람 위치, 거리, 동작 속도가 달라진다.
- (IP Webcam 결과 비교는 테스트 후 추가)


## 한 줄 정리

- Runtime Pipeline 자체(Pose → 38-D → Window → 30 Step → RF → Threshold → Stable)는 정상으로 동작했다. 다만 6일차 모델의 Confidence가 0.2대라서 기본 Threshold 0.55에서는 모든 결과가 UNKNOWN으로 보류됐다.
