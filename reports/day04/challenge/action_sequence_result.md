# Day 04 Team Mini Challenge — 3팀 무인 상점 이상행동 인식

교재 조합(A~C) 대신 우리 팀 동작 중 두 개를 골랐다.
두 영상은 모션캡처 촬영 때 같은 측면 카메라(가로 영상, 1920x1080, 60fps)로 직접 찍은 영상이다.
같은 사람, 같은 장소, 카메라 고정, 전신이 보이는 조건이라 교재의 촬영 기준과 맞는다.

## 1. 촬영한 Action

- Action 1: `faint_back` — 서 있다가 뒤로 기절하듯 쓰러져 누워 있다가 일어난다 (3팀 16번 동작)
- 파일명: `challenge_faint_back.mp4` (원본 `20261113_165214.mp4`, 13.5초, 808 Frame)
- Action 2: `trip_fall` — 걷다가 장애물에 걸려 앞으로 넘어졌다가 일어난다 (3팀 17번 동작)
- 파일명: `challenge_trip_fall.mp4` (원본 `20261113_165930.mp4`, 6.3초, 374 Frame)
- 설정 파일: `configs/day04_challenge_actions.csv`

두 동작 모두 "사람이 바닥으로 쓰러지는" 이상행동이고 쓰러지는 방향만 다르다.
무인 상점에서는 둘 다 알림이 필요한 상황이라, 두 동작이 Sequence에서 어떻게 다르게 보이는지 확인했다.

## 2. 대표 Frame 확인

`21_extract_clip_keyframes.py` (`reports/day04/challenge/keyframes/`)

### Action 1 — faint_back
- Start: 옆으로 서 있음
- Middle: 바닥에 등을 대고 팔을 벌린 채 누워 있음
- End: 다시 일어나 서 있음

### Action 2 — trip_fall
- Start: 서 있음 (걷기 시작 전)
- Middle: 앞으로 넘어져 손과 무릎을 바닥에 짚고 엎드려 있음
- End: 다시 일어나 서 있음

> 처음 실행했을 때 21번 코드가 `Frame을 읽지 못했습니다: challenge_faint_back.mp4 frame=809`로 멈췄다.
> 휴대폰 영상은 영상 정보에 기록된 Frame 수(810, 379)가 실제로 읽을 수 있는 Frame 수(808, 374)보다 커서
> End 위치가 영상 끝을 넘었기 때문이다.
> 그래서 `read_frame_at()`이 이미 읽은 Frame이 있으면 실제 마지막 Frame을 대신 쓰도록 고쳤다.
> 교재 영상 6개는 수정 후에도 같은 결과(18장)가 나왔다.

## 3. Sequence 품질

`22_build_action_sequences.py` → `23_summarize_action_sequences.py`

| 항목 | faint_back | trip_fall |
|---|---:|---:|
| rows | 808 | 374 |
| pose_ok_ratio | 1.0 | 1.0 |
| feature_ok_ratio | 1.0 | 1.0 |
| mean_keypoint_conf | 0.8476 | 0.7856 |
| mean_low_keypoint_count | 1.778 | 3.096 |
| mean_adjacent_feature_change | 4.205 | 3.209 |

- Pose 실패 없음, Feature 생성 실패 없음.
- 평균 Confidence는 교재 영상(0.89~0.97)보다 낮았다. 측면이라 카메라 반대쪽 팔다리가 가려지고, 바닥에 누우면 얼굴 Keypoint가 잘 안 보인다.
- trip_fall은 넘어진 뒤 손·무릎을 짚은 구간(3~5초)에서 LOW가 5개까지 늘었다.
- `mean_adjacent_feature_change`는 교재 일반 Action(0.10~0.22)보다 20~40배 컸다. 이 값은 성능 점수가 아니다. 3일차에서 확인한 것처럼, 옆에서 보면 두 어깨가 겹쳐 어깨 폭이 9~175px로 크게 흔들리고, 그래서 어깨 폭으로 나눈 정규화 좌표가 크게 튄 영향이 크다.

## 4. 실제로 잘 변한 Feature

`24_inspect_sequence_stages.py --samples 5`, `--samples 9`

### Action 1 — faint_back
- Feature 1: `hip_center_y` (원본 Pixel, 클수록 화면 아래)
- Feature 2: `left_knee_angle` / `right_knee_angle`
- 변화 (0.5초 간격으로 확인):
  - 0~2.0초: hip_center_y 580~611 (서 있음), 무릎 157~166°
  - 2.5~3.0초: hip_center_y 763~772로 **약 190px 아래로 떨어짐**. 1초 안에 쓰러졌다.
  - 4.0~5.0초: 누운 채 무릎을 세움 (무릎 93~105°)
  - 5.5~11.0초: 다리를 대부분 펴고 누워 있음 (무릎 134~176°), hip_center_y 694~733 유지
  - 11.5~12.5초: 일어나는 중 hip_center_y 631→584, 무릎 51~72°까지 굽힘
  - 13.0초: hip_center_y 535, 무릎 162~170° (다시 서 있음)

### Action 2 — trip_fall
- Feature 1: `hip_center_x` (원본 Pixel, 화면 가로 위치)
- Feature 2: `left_knee_angle` / `right_knee_angle`
- 변화:
  - 0~1.5초: hip_center_x 591→651 (천천히 걸음), 무릎 151~169°
  - 2.0~3.0초: hip_center_x 790→1172로 **1초 동안 약 380px 앞으로 이동**하며 넘어짐, 무릎 115°→90°
  - 3.0~4.5초: hip_center_x 1139~1179로 멈춤, 무릎 70~98° (손·무릎을 짚은 자세)
  - 5.5~6.0초: hip_center_y 514~534로 올라가고 무릎 162~168° (일어남)
  - hip_center_y는 서 있을 때 626, 넘어진 뒤 최대 663으로 **약 40px만 내려갔다.** 바닥에 완전히 눕지 않고 손·무릎으로 버텼기 때문이다.

## 5. 두 Action 비교

- 두 Action에서 가장 다르게 보인 Feature:
  - faint_back은 **골반 높이(hip_center_y)가 크게 떨어진 뒤 오래 유지**되었다 (약 190px, 8초 이상).
  - trip_fall은 **골반이 앞으로 빠르게 이동(hip_center_x)**하고, 높이는 조금만 내려간 뒤 무릎이 70~100°로 굽은 상태가 유지되었다.
  - 3일차 팀 Feature로 보면 faint_back은 "눕기", trip_fall은 "숙이기"(손·무릎 짚음)로 나온 것과 같은 결과다.
- 시작과 마지막은 비슷했지만 중간이 달랐던 부분:
  - 두 영상 모두 시작과 끝은 서 있는 자세(무릎 160~170°)라 처음·마지막 Frame만 보면 차이가 없다.
  - 중간에 faint_back은 무릎을 폈다 굽혔다 하며 누워 있고, trip_fall은 무릎을 굽힌 채 엎드려 있다.
- 5개 Sample로 부족했다면 어떻게 확인했는가:
  - faint_back 5개 Sample은 3.4초, 6.7초처럼 누운 구간만 잡혀서 "누운 채 무릎을 세운 구간"(5.0초, 무릎 98~102°)이 빠졌다. 9개로 늘려서 확인했다.
  - 둘 다 쓰러지는 순간이 1초 이내라, 0.5초 간격(30 Frame마다)으로 hip_center와 무릎 각도를 다시 출력해 쓰러진 시점을 찾았다.
  - 누운 구간의 일부 정규화 값은 믿기 어려웠다. 예를 들어 faint_back 6.7초의 왼팔꿈치 8.6°, ankle_x 6.8이 있다. 그래서 Pixel 좌표인 hip_center와 관절각도를 중심으로 해석했다.

## 6. 최종 정리

- Pose와 Action의 차이: Pose는 한 순간의 자세이고, Action은 서기 → 쓰러짐 → 누워 있음 → 일어남처럼 자세가 시간에 따라 바뀌는 과정이다.
- Clip과 Sequence의 차이: Clip은 MP4 파일 하나(`challenge_faint_back.mp4`)이고, Sequence는 그 Clip의 Frame마다 만든 38-D Feature를 시간 순서대로 쌓은 CSV(`faint_back_sequence.csv`, 808행)이다.
- 한 Frame만으로 Action을 설명하기 어려운 이유:
  - "바닥에 누운 한 장면"만 보면 기절해서 쓰러진 것인지, 스스로 누운 것인지(3팀 10번 동작) 알 수 없다. 골반이 1초 안에 190px 떨어졌다는 **속도 정보**는 여러 Frame을 이어서 봐야만 나온다.
  - 넘어지기도 한 장면만 보면 "무릎 꿇고 엎드린 자세"(3팀 12번 인형빼기와 비슷)로 보인다. 앞으로 빠르게 이동하다 멈췄다는 흐름이 있어야 넘어짐이라고 말할 수 있다.
- 오늘 만든 데이터의 한계:
  - 동작마다 한 명, 한 번(Take 1개)만 찍은 영상이라 이 값이 다른 사람에게도 똑같이 나온다고 할 수 없다.
  - 측면 영상에서는 어깨 폭 정규화가 불안정해 정규화 좌표의 변화량이 크게 부풀었다. 3일차에서 확인한 상체 길이 기준 정규화나 Pixel 좌표(hip_center)를 함께 써야 한다.
  - 정면 영상(165152, 165917)은 이번에 쓰지 않았다. 정면에서 쓰러지는 동작이 어떻게 보이는지는 따로 확인이 필요하다.

## Mini Challenge 완료 체크

- [x] 두 개의 Action MP4를 직접 촬영했다. (모션캡처 촬영 때 찍은 측면 영상 사용)
- [x] 두 영상이 정상적으로 열리는지 확인했다. (open_ok=1, read_ok=1, 60fps, 1920x1080)
- [x] Action Label과 MP4를 설정 파일에 연결했다.
- [x] Start / Middle / End 대표 Frame을 만들었다.
- [x] 두 MP4를 Sequence CSV로 변환했다.
- [x] `pose_ok_ratio`와 `feature_ok_ratio`를 확인했다.
- [x] 평균 Confidence와 LOW Keypoint 수를 확인했다.
- [x] 각 Action의 대표 시점 Feature를 확인했다.
- [x] 실제로 잘 변한 Feature를 Action별로 2~3개 찾았다.
- [x] 5개 Sample이 부족하면 Sample 수를 늘려 다시 확인했다.
- [x] 두 Action의 시간 변화 차이를 설명했다.
- [x] `action_sequence_result.md`를 작성했다.
