# Day 08 Summary - Multi-Person Pose Tracking

화면에 두 사람 이상이 있을 때 사람마다 Track ID를 붙이고, ID별로 Feature Buffer와 Action 상태를 따로 관리하는 구조를 만들었다.
새 Action 모델은 학습하지 않았다. 6일차 Random Forest와 7일차 Sliding Window Runtime을 그대로 재사용한다.

## 1. 입력 확인 (46번)

| 항목 | 결과 |
|---|---|
| 6일차 모델 입력 | 30 Step × 38-D = 1,140 Feature, 6개 Action |
| SBU 영상 | 3 / 3 (640×480, 약 15 FPS) |
| Tracker | ByteTrack (`configs/day08_multi_person.json`, Lost Timeout 2.0초) |
| 판정 | `Day08 input data: READY` |

- 01_sbu_shaking_hands: 30 Frame, 1.96초 (기본 2인 Tracking)
- 02_sbu_hugging: 33 Frame, 2.16초 (몸이 겹치는 Occlusion)
- 03_sbu_kicking_challenge: 18 Frame, 1.16초 (빠른 움직임)
- 세 영상 모두 7일차 5초 Window보다 짧다. 오늘은 Action 판정이 아니라 Frame 사이에 같은 사람이 같은 ID로 이어지는지만 본다.

## 2. SBU Tracking (47 → 48 → 51번)

| 영상 | Frame | Unique ID | Two-ID Ratio | Extra ID | 판단 |
|---|---:|---:|---:|---:|---|
| Shaking Hands | 30 | 2 | 1.0000 | 0 | 두 ID가 끝까지 유지 |
| Hugging | 33 | 4 | 0.5455 | 2 | 겹친 구간에서 ID 1 사라짐 + 중복 Track 2개 |
| Kicking | 18 | 2 | 1.0000 | 0 | 빠른 발차기에도 두 ID 유지 |

- Hugging에서 ID 1은 33 Frame 중 16 Frame만 잡혔다. 안고 있는 동안 흰 옷 사람(ID 2)의 BBox 하나가 두 사람을 덮었고, 검은 옷 사람은 가려져 따로 검출되지 않았다.
- 두 사람이 떨어진 Frame 25부터 검은 옷 사람에게 다시 원래 번호 ID 1이 붙었다. ByteTrack이 잠깐 사라진 Track을 기억해 두었다가 다시 이은 것이다.
- ID 3(Frame 6~11)과 ID 4(Frame 20~23)는 box_conf 0.3대의 짧은 중복 Track이었다.
- 처음과 끝에 같은 사람이 같은 번호였기 때문에, 두 사람의 번호가 서로 바뀐 ID Switch는 없었다고 판단했다.
- 이번 영상들에서는 빠른 움직임보다 **몸이 겹칠 때** Tracking이 더 흔들렸다.
- 자세한 판단 근거: `id_switch_cases.md`

교재와 다른 점: 교재의 Hugging은 ID 1이 24 Frame, 추가 ID가 4·5, Two-ID Ratio 0.6364였다. 추가 Track이 생긴 Frame 구간과 Confidence는 같았고, ID 1이 끊긴 길이만 달랐다.

## 3. 사람별 Runtime State (multi_person_utils · 49 · 50번)

```text
Frame → YOLO Pose + ByteTrack → Track ID
→ ID마다 38-D Feature → ID마다 TimedFeatureBuffer (5초) → 6일차 Random Forest
→ ID마다 PredictionStabilizer → ID별 Stable Action
```

- 새 Track ID가 나타나면 새 Buffer를 만들고 WARMING_UP부터 시작한다.
- 2초(Lost Timeout) 넘게 안 보인 ID는 상태를 지운다. 잠깐 가려진 사람의 Buffer는 바로 지우지 않는다.
- 코드 동작 확인: 카메라 대신 Hugging 저장영상을 49번에 넣어 끝까지 실행했다. ID 1~4가 각각 따로 기록됐고, 50번이 ID별 요약을 만들었다. 영상이 2초라 Window가 차지 않아 예측은 없었다 (정상). 이 실행은 확인용이라 결과로 쓰지 않았다.

## 4. 아직 하지 않은 것

- **IP Webcam 2인 실시간 실습 (교재 39~45번)**: 두 사람이 카메라 앞에서 서로 다른 동작, 교차, 화면 밖으로 나갔다 들어오기를 해야 한다.
- **Team Mini Challenge (교재 47~58번)**: 2~3명이 현장형 장면을 직접 촬영해야 한다.
- 두 가지 모두 사람과 폰 카메라가 필요해서 다음에 함께 진행한다.

## 5. 문제와 해결

- ByteTrack 실행 시 `No module named 'lap'` 오류가 났다. 교재에는 설치 단계가 없다. Ultralytics가 pip로 자동 설치하려 했지만 우리 가상환경은 `uv`로 만들어 pip가 없었다. `uv add lap`으로 설치하고 `pyproject.toml`에 기록했다.

## 6. 정리

- Track ID는 사람의 실제 신원이 아니라 한 영상 안에서 같은 사람을 잇는 임시 번호다.
- Unique ID 수가 늘었다고 바로 ID Switch는 아니다. Track Lost 후 재연결, 중복 Track, 실제 ID Switch를 결과 영상으로 구분해야 한다.
- 사람별 Action이 맞으려면 Tracking이 먼저 맞아야 한다. ID가 바뀌면 다른 사람의 Feature가 같은 Buffer에 섞인다.
