# Day 04 Action Clip & Sequence Summary

## 1. 오늘 한 일

6개의 Action MP4를 Frame 단위로 분석하여
3일차의 38-D Frame Feature를 시간 순서대로 연결한
Sequence CSV를 만들었다.

사용한 Action:

- bend_return
- leg_raise_lower
- walk_turn_walk
- drink_return
- sit_stand
- wave

Mini Challenge에서는 우리 팀(3팀 무인 상점 이상행동) 17개 동작을
정면·측면으로 찍은 영상 36개를 같은 Pipeline에 넣어 Sequence로 만들었다.

## 2. 오늘 만든 Pipeline

Action Clip
→ Frame
→ YOLO Pose
→ COCO 17 Keypoint
→ Normalization
→ Joint Angle
→ 38-D Frame Feature
→ 시간 순서 유지
→ Sequence
→ Action Label

## 3. 본 실습 Sequence 결과

입력 영상 확인 (`19_check_day04_data.py`): 6개 모두 640x480, 25fps, open_ok=1, read_ok=1

| Action | Frame 수 | 재생시간 |
|---|---|---|
| bend_return | 85 | 3.40초 |
| leg_raise_lower | 125 | 5.00초 |
| walk_turn_walk | 266 | 10.64초 |
| drink_return | 120 | 4.80초 |
| sit_stand | 102 | 4.08초 |
| wave | 126 | 5.04초 |

6개 영상에서 총 824 Frame을 처리했다.

- Pose 성공: 824 / 824
- 38-D Feature 성공: 824 / 824
- 6개 Action 모두 pose_ok_ratio = 1.0
- 6개 Action 모두 feature_ok_ratio = 1.0

### 대표 Frame 확인 (13번)

| Action | 동작 일치 여부 | 불필요한 구간·문제점 | 사용 구간(초) | 검수 결과 |
|---|---|---|---|---|
| `bend_return` | 서 있음 → 허리를 굽힘 → 다시 서 있음, 일치 | 없음 | 0~3.40 | PASS |
| `leg_raise_lower` | 서 있음 → 한쪽 다리를 옆으로 듦 → 다시 서 있음, 일치 | 다리를 앞이 아니라 옆으로 듦 | 0~5.00 | PASS |
| `walk_turn_walk` | 정면으로 서 있음 → 옆으로 걷는 중 → 다시 정면으로 서 있음, 일치 | 걷는 구간에서 몸이 옆으로 돌아 Confidence가 낮아짐 | 0~10.64 | PASS |
| `drink_return` | 정면 자세 → 오른손으로 병을 입에 댐 → 다시 정면 자세, 일치 | 없음 | 0~4.80 | PASS |
| `sit_stand` | **의자에 앉은 자세로 시작** → 일어나는 중 → 서 있음 | 서기 → 앉기 → 서기가 아니라 앉은 상태에서 일어나는 동작만 있음 | 0~4.08 | 수정 (Label 설명과 실제 동작이 다름) |
| `wave` | 팔 내림 → 오른팔을 들어 흔듦 → 팔을 내림, 일치 | 없음 | 0~5.04 | PASS |

- `sit_stand`는 설정 파일 설명("의자에 앉았다가 다시 일어선다")과 달리 영상이 앉은 자세로 시작했다. 24번 결과에서도 무릎 각도가 150~160° → 172~178°로 펴지기만 하고, Hip Center y가 286 → 250으로 올라가기만 했다. 예상에 맞추지 않고 실제 이미지에 보이는 대로 기록했다.

## 4. Sequence 품질에서 확인한 점

| Action | 평균 Confidence | 평균 LOW 수 | 인접 Row 평균 변화량 |
|---|---|---|---|
| bend_return | 0.9583 | 0.224 | 0.159 |
| drink_return | 0.9538 | 0.558 | 0.113 |
| leg_raise_lower | 0.9678 | 0.072 | 0.217 |
| sit_stand | 0.9743 | 0.000 | 0.101 |
| walk_turn_walk | 0.8883 | 1.677 | 17.478 |
| wave | 0.9724 | 0.000 | 0.178 |

walk_turn_walk은 다른 Action보다:

- 평균 Keypoint Confidence가 낮았다.
- LOW Keypoint 수가 많았다.
- 인접 Frame의 정규화 Feature 변화량이 매우 크게 나타났다.

하지만 Pose와 Feature 생성 자체는 모든 Frame에서 성공했다.

mean_adjacent_feature_change는
Action Recognition Accuracy가 아니라
Frame 사이 자세 변화 정도를 보기 위한 참고값이다.

## 5. Action별 Feature 변화

- wave:
  오른쪽 Wrist y가 -0.03 → -3.05 → -0.04로 올라갔다 내려왔고, 오른쪽 Elbow Angle이 176.9° → 115.9° → 173.1°로 변했다.

- drink_return:
  오른쪽 Wrist y가 -0.05 → -2.10 → -0.02로 올라갔다 돌아왔고, 오른쪽 Elbow Angle이 170.1° → 60.6° → 172.2°로 크게 작아졌다가 복귀했다.

- bend_return:
  중간 구간에서 양쪽 Wrist y가 약 0 → 0.95~0.98로 변하고 다시 0 근처로 돌아왔다. Hip Center y도 249 → 258 → 250으로 변했다.

- sit_stand:
  Knee Angle이 160° / 149° → 178° / 173°로 펴지고 Hip Center y가 286 → 250으로 올라가 일어서는 과정이 나타났다.

- walk_turn_walk:
  Hip Center x가 472 → 279 → 462로 이동했다 돌아와 화면 이동이 뚜렷하게 나타났다.

- leg_raise_lower:
  대표 5개 시점만으로는 다리를 든 순간이 잡히지 않았다. 9개로 늘리자 Frame 46(1.84초)에서 오른쪽 Ankle y가 2.4 → 1.81로 올라가고 x가 -0.40 → -3.23으로 옆으로 이동했으며, 오른쪽 Knee가 166.7°였다.

## 6. 오늘 이해한 핵심

Pose는 한 순간의 자세이고,
Action은 여러 Pose가 시간에 따라 변하는 과정이다.

한 Frame의 38-D Feature만으로는
Action 전체를 설명하기 어렵다.
wave와 drink_return은 시작과 끝의 값이 거의 같고, 중간에서만 크게 달랐다.

여러 Frame의 Feature를 원래 시간 순서대로 연결한
Sequence가 필요하다.

## 7. Mini Challenge

- 데이터: 3팀 17개 동작 × 정면·측면 = 영상 36개 (총 18,950 Frame)
- Pose / Feature 품질: Pose 성공 18,742 / 18,950 (98.9%). 32개 영상은 pose_ok_ratio = 1.0. 실패는 바닥에서 몸을 말거나 엎드린 장면뿐이었다.
- 정면 영상(평균 Confidence 중앙값 0.937)이 측면(0.829)보다 품질이 좋았다. 측면은 어깨 폭 정규화가 불안정해 인접 Row 변화량이 크게 부풀었다(중앙값 4.475).
- Action 1: `faint_back` (16 뒤로 기절하기, 측면, 808 Frame)
- Action 2: `trip_fall` (17 앞으로 넘어지기, 측면, 374 Frame)
- 시간 변화가 가장 잘 보였던 Feature:
  - faint_back: Hip Center y (1초 안에 약 190px 하강 후 8초 이상 유지), Knee Angle
  - trip_fall: Hip Center x (1초에 약 380px 앞으로 이동), Knee Angle (넘어진 뒤 70~100°)
- 두 Action에서 가장 크게 달랐던 점: 기절은 골반이 바닥까지 내려가 오래 머물렀고, 넘어지기는 골반이 앞으로 빠르게 이동한 뒤 높이는 약 40px만 내려가 손·무릎으로 버텼다.
- 휴대폰 영상은 교재 21번 원본이 End Frame에서 멈춰, Mini Challenge에서만 이를 처리한 사본으로 대표 Frame을 만들었다.
- 자세한 내용: `reports/day04/challenge/action_sequence_result.md`

## 8. 현재 데이터의 한계

현재 6개의 Clip은 Sequence 구조를 이해하기 위한
소규모 실습 데이터이다.

실제 Action Recognition 학습에는
여러 Subject와 Action별 여러 Take가 추가로 필요하다.

우리 팀 영상도 동작마다 한 사람이 한 번 찍은 영상이라, 학습용으로는 여러 사람·여러 Take가 더 필요하다.
또 측면 영상에서는 어깨 폭 정규화가 불안정해 정규화 좌표 변화량이 크게 부풀었다
(Mini Challenge 4.2 / 3.2, 교재 일반 Action 0.10~0.22).
