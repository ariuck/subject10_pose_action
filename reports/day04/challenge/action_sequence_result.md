# Day 04 Team Mini Challenge — 3팀 무인 상점 이상행동 (17개 동작, 정면·측면)

## 1. 촬영한 Action

모션캡처 촬영 때 우리 팀(3팀) 이상행동 17개를 **정면·측면 카메라**로 찍은 영상 36개를 사용했다.
(9번은 정면 영상 2개, 16번은 측면 영상 2개)

- 설정 파일: `configs/day04_challenge_actions.csv` (36행)
- 영상: `data/day04/challenge/videos/challenge_<action_label>.mp4`
- 22번 코드는 Sequence 파일 이름을 `action_label`로 만들기 때문에, 같은 동작의 정면·측면 파일이 덮어쓰이지 않도록 Label 뒤에 `_front` / `_side`를 붙였다.
  - 예: `kiosk_tap_front`, `kiosk_tap_side`
- 영상은 모두 세로(1080x1920) 또는 가로(1920x1080) 60fps 휴대폰 영상이다. 교재 기준(가로 영상)과 다른 영상도 있다.

| # | 동작 | Action Label | 정면 | 측면 |
|---|---|---|---|---|
| 1 | 키오스크 두드리기 | kiosk_tap | 151425 | 151447 |
| 2 | 한손 물건 던지기 | throw_one_hand | 151737 | 151744 |
| 3 | 앞차기 | front_kick | 152526 | 152535 |
| 4 | 잡고 흔들기 | grab_shake | 152953 | 153015 |
| 5 | 밀치기 | push | 153253 | 153259 |
| 6 | 휘두르기 | swing | 153732 | 153746 |
| 7 | 바닥에 앉기 | sit_floor | 154206 | 154224 |
| 8 | 위에 앉기 | sit_on_object | 154727 | 154744 |
| 9 | 앞으로 눕기 | lie_forward | 160836, 161112 | 161202 |
| 10 | 뒤로 눕기 | lie_backward | 161645 | 161722 |
| 11 | 몸통 박치기 | body_slam | 162139 | 162147 |
| 12 | 쪼그려 물건 빼내기 | crouch_take_item | 162425 | 162450 |
| 13 | 샷건 | shotgun | 162728 | 162736 |
| 14 | 구토 | vomit | 164329 | 164347 |
| 15 | 옆으로 쓰러지기 | fall_sideways | 164838 | 164854 |
| 16 | 뒤로 넘어지기 | fall_backward | 165152 | 165054, 165214 |
| 17 | 앞으로 넘어지기 | fall_forward | 165917 | 165930 |

(영상 번호는 `20261113_xxxxxx.mp4`의 뒷부분)

비교할 두 Action은 교재 방식대로 하나를 골랐다.
**Action 1 = 16 뒤로 넘어지기 (`fall_backward_side2`)**, **Action 2 = 17 앞으로 넘어지기 (`fall_forward_side`)** 이다.
둘 다 같은 측면 카메라(가로 1920x1080)로 찍었고, 사람이 바닥으로 쓰러지는 이상행동이라 쓰러지는 방향만 다르다.

## 2. 대표 Frame 확인

`reports/day04/challenge/keyframes/` (36개 × Start / Middle / End = 108장)

```bash
python scripts/team3/team_extract_clip_keyframes.py \
  --action-file configs/day04_challenge_actions.csv \
  --video-dir data/day04/challenge/videos \
  --output-dir reports/day04/challenge/keyframes
```

- 교재 21번 원본은 휴대폰 영상에서 멈췄다. 영상 정보의 Frame 수가 실제로 읽히는 Frame 수보다 1~6개 많아서(예: 810 vs 808), End Frame을 읽지 못했기 때문이다.
- 그래서 Mini Challenge에서는 이 경우 실제 마지막 Frame을 쓰는 사본(`scripts/team3/team_extract_clip_keyframes.py`)으로 추출했다. 교재 본 실습은 21번 원본으로 실행했다.
- 36개 영상 모두 Start·End는 서 있는 자세이고, Middle에 동작이 들어 있었다.

### Action 1 — fall_backward (측면)
- Start: 옆으로 서 있음
- Middle: 바닥에 등을 대고 팔을 벌린 채 누워 있음
- End: 다시 일어나 서 있음

### Action 2 — fall_forward (측면)
- Start: 서 있음 (걷기 시작 전)
- Middle: 앞으로 넘어져 손과 무릎을 바닥에 짚고 엎드려 있음
- End: 다시 일어나 서 있음

## 3. Sequence 품질

```bash
python scripts/22_build_action_sequences.py \
  --action-file configs/day04_challenge_actions.csv \
  --video-dir data/day04/challenge/videos \
  --output-dir reports/day04/challenge/sequences

python scripts/23_summarize_action_sequences.py \
  --sequence-dir reports/day04/challenge/sequences \
  --output reports/day04/challenge/challenge_sequence_summary.csv
```

| 항목 | Action 1 fall_backward (측면) | Action 2 fall_forward (측면) |
|---|---:|---:|
| rows | 808 | 374 |
| pose_ok_ratio | 1.0 | 1.0 |
| feature_ok_ratio | 1.0 | 1.0 |
| mean_keypoint_conf | 0.8476 | 0.7856 |
| mean_low_keypoint_count | 1.778 | 3.096 |
| mean_adjacent_feature_change | 4.205 | 3.209 |

### 36개 전체

- 총 18,950 Frame 처리, Pose 성공 18,742 (98.9%), Feature 성공 18,742
- 32개 영상은 pose_ok_ratio = 1.0

| 구분 | 평균 Confidence (최소 / 중앙 / 최대) | 평균 LOW 수 (중앙값) | 인접 Row 변화량 (최소 / 중앙 / 최대) |
|---|---|---|---|
| 정면 18개 | 0.808 / 0.937 / 0.976 | 0.61 | 0.079 / 0.227 / 5.771 |
| 측면 18개 | 0.724 / 0.829 / 0.920 | 2.41 | 0.270 / 4.475 / 16.983 |

- 정면 영상이 측면보다 Confidence가 높고 LOW가 적었다. 측면에서는 카메라 반대쪽 팔다리와 눈·귀가 몸에 가려진다.
- 서서 하는 동작은 정면에서 인접 Row 변화량이 0.10~0.30으로 교재 영상(0.10~0.22)과 비슷했다. 같은 동작을 측면에서 찍으면 4.7~17.0으로 크게 뛰었다(몸통 박치기 측면만 0.27).
  - 예: 잡고 흔들기 정면 0.120 / 측면 16.983, 키오스크 두드리기 정면 0.129 / 측면 14.293
  - 측면에서는 두 어깨가 앞뒤로 겹쳐 어깨 폭이 작아지고, 어깨 폭으로 나눈 정규화 좌표가 크게 튄 영향이다. 실제로 더 많이 움직인 것이 아니다.

### 실패 Frame 확인 (교재 41번: 실패가 있을 때만 원인 확인)

| Sequence | 실패 Frame | 실패 구간 | 실제 영상에서 확인한 장면 |
|---|---|---|---|
| `lie_backward_front` | 171 / 1512 | 15.0~17.6초 | 카메라에서 먼 벽 쪽 바닥에 몸을 말고 옆으로 누움 (1일차 Challenge에서 사람 검출 실패한 장면과 같은 구간) |
| `lie_backward_side` | 21 / 1610 | 6.3~7.6초 | 바닥에 등을 대고 무릎을 세우고 손을 머리 뒤로 둔 자세 |
| `crouch_take_item_front` | 15 / 770 | 3.45~3.85초 | 카메라 쪽을 향해 엎드려 손·무릎으로 기는 자세, 머리가 아래를 향함 |
| `fall_forward_front` | 1 / 403 | 2.47초 | 카메라 쪽으로 앞으로 넘어지며 손을 짚는 순간 |

- 실패한 Frame도 삭제되지 않고 `pose_ok = 0`으로 시간 순서 안에 남아 있었다.
- 실패는 모두 바닥에서 몸을 말거나 엎드린 장면이었다. 서서 하는 동작에서는 실패가 없었다.

## 4. 실제로 잘 변한 Feature

`24_inspect_sequence_stages.py --samples 5`, `--samples 9` + 0.5초 간격 확인

### Action 1 — fall_backward (측면)
- Feature 1: `hip_center_y` (원본 Pixel, 클수록 화면 아래)
- Feature 2: `left_knee_angle` / `right_knee_angle`
- 변화 (0.5초 간격):
  - 0~2.0초: hip_center_y 580~611 (서 있음), 무릎 157~166°
  - 2.5~3.0초: hip_center_y 763~772로 **약 190px 아래로 떨어짐**. 1초 안에 쓰러졌다.
  - 4.0~5.0초: 누운 채 무릎을 세움 (무릎 93~105°)
  - 5.5~11.0초: 다리를 대부분 펴고 누워 있음 (무릎 134~176°), hip_center_y 694~733 유지
  - 11.5~12.5초: 일어나는 중 hip_center_y 631→584, 무릎 51~72°까지 굽힘
  - 13.0초: hip_center_y 535, 무릎 162~170° (다시 서 있음)

### Action 2 — fall_forward (측면)
- Feature 1: `hip_center_x` (원본 Pixel, 화면 가로 위치)
- Feature 2: `left_knee_angle` / `right_knee_angle`
- 변화:
  - 0~1.5초: hip_center_x 591→651 (천천히 걸음), 무릎 151~169°
  - 2.0~3.0초: hip_center_x 790→1172로 **1초 동안 약 380px 앞으로 이동**하며 넘어짐, 무릎 115°→90°
  - 3.0~4.5초: hip_center_x 1139~1179로 멈춤, 무릎 70~98° (손·무릎을 짚은 자세)
  - 5.5~6.0초: hip_center_y 514~534로 올라가고 무릎 162~168° (일어남)
  - hip_center_y는 서 있을 때 626, 넘어진 뒤 최대 663으로 **약 40px만 내려갔다.** 바닥에 완전히 눕지 않고 손·무릎으로 버텼기 때문이다.

### 다른 동작에서 함께 확인한 것 (정면)
- 1 키오스크 두드리기 (`kiosk_tap_front`, 5개 Sample): 오른쪽 Wrist y가 -2.09 → -2.36 → -1.70 → -0.51 → -0.07로, 손을 머리 위로 올렸다가 아래로 내려쳤다. 왼쪽 Wrist는 -0.27 ~ 0.04로 거의 그대로였다.
- 3 앞차기 (`front_kick_front`, 9개 Sample): 5개 Sample에서는 차는 순간이 잘 보이지 않아 9개로 늘렸다.
  - 1.03초에 양쪽 Wrist y가 -1.49 / -1.54로 크게 올라가고, 팔꿈치가 41° / 60°로 굽었다. 균형을 잡으려고 팔을 들었다.
  - 반면 Ankle y는 2.16 → 1.85~1.95로 조금만 줄었다. 정면에서는 발이 카메라 쪽으로 나오기 때문에 다리를 든 높이가 잘 보이지 않았다.

## 5. 두 Action 비교

- 두 Action에서 가장 다르게 보인 Feature:
  - fall_backward은 **골반 높이(hip_center_y)가 크게 떨어진 뒤 오래 유지**되었다 (약 190px, 8초 이상).
  - fall_forward은 **골반이 앞으로 빠르게 이동(hip_center_x)**하고, 높이는 조금만 내려간 뒤 무릎이 70~100°로 굽은 상태가 유지되었다.
- 시작과 마지막은 비슷했지만 중간이 달랐던 부분:
  - 두 영상 모두 시작과 끝은 서 있는 자세(무릎 160~170°)라 처음·마지막 Frame만 보면 차이가 없다.
  - 중간에 fall_backward은 무릎을 폈다 굽혔다 하며 누워 있고, fall_forward은 무릎을 굽힌 채 엎드려 있다.
- 5개 Sample로 부족했다면 어떻게 확인했는가:
  - fall_backward 5개 Sample은 3.4초, 6.7초, 10.1초처럼 누운 구간만 잡혀서 "누운 채 무릎을 세운 구간"(5.0초, 무릎 98~102°)이 빠졌다. 9개로 늘려서 확인했다.
  - 둘 다 쓰러지는 순간이 1초 이내라, 0.5초 간격(30 Frame마다)으로 hip_center와 무릎 각도를 다시 출력해 쓰러진 시점을 찾았다.
  - 누운 구간의 일부 정규화 값은 믿기 어려웠다. 예를 들어 fall_backward 6.7초의 왼팔꿈치 8.6°, ankle_x 6.8이 있다. 측면에서 어깨 폭이 작아진 영향이라, Pixel 좌표인 hip_center와 관절각도를 중심으로 해석했다.

## 6. 최종 정리

- Pose와 Action의 차이: Pose는 "바닥에 누운 한 순간"이고, Action은 서기 → 쓰러짐 → 누워 있음 → 일어남처럼 Pose가 시간에 따라 바뀌는 전체 흐름이다.
- Clip과 Sequence의 차이: Clip은 MP4 파일 하나(`challenge_fall_backward_side2.mp4`)이고, Sequence는 그 Clip의 Frame마다 만든 38-D Feature를 시간 순서대로 쌓은 CSV(`fall_backward_side2_sequence.csv`, 808행)이다.
- 한 Frame만으로 Action을 설명하기 어려운 이유:
  - "바닥에 누운 한 장면"만 보면 뒤로 넘어져 쓰러진 것인지(16번), 스스로 누운 것인지(10번) 알 수 없다. 골반이 1초 안에 190px 떨어졌다는 정보는 여러 Frame을 이어서 봐야만 나온다.
  - 넘어지기도 한 장면만 보면 "손·무릎을 짚고 엎드린 자세"(12번 쪼그려 물건 빼내기와 비슷)로 보인다. 앞으로 빠르게 이동하다 멈췄다는 흐름이 있어야 넘어짐이라고 말할 수 있다.
  - 키오스크 두드리기도 처음과 마지막에는 손이 내려가 있어, 손을 올렸다 내려친 중간 흐름을 봐야 동작을 알 수 있다.
- 오늘 만든 데이터의 한계:
  - 동작마다 한 사람이 한 번 찍은 영상이라, 이 Feature 변화가 다른 사람·다른 속도에서도 똑같이 나온다고 할 수 없다.
  - 측면 영상은 어깨 폭 정규화가 불안정해 정규화 좌표 변화량이 크게 부풀었다. 3일차 Mini Challenge에서 확인한 상체 길이 기준 정규화나 Pixel 좌표(hip_center)를 함께 써야 한다.
  - 바닥에서 몸을 만 장면은 Pose를 아예 찾지 못한 Frame이 있었다(최대 171 Frame).
  - 흔들기·밀치기처럼 물체와 닿는 동작은 Pose Sequence만으로는 상대(기계·사람)를 알 수 없다.

## Mini Challenge 완료 체크

- [x] 두 개의 Action MP4를 직접 촬영했다. (17개 동작 정면·측면 36개 영상, 비교는 16번 vs 17번 측면)
- [x] 두 영상이 정상적으로 열리는지 확인했다.
- [x] Action Label과 MP4를 설정 파일에 연결했다.
- [x] Start / Middle / End 대표 Frame을 만들었다. (108장)
- [x] 두 MP4를 Sequence CSV로 변환했다. (36개)
- [x] `pose_ok_ratio`와 `feature_ok_ratio`를 확인했다.
- [x] 평균 Confidence와 LOW Keypoint 수를 확인했다.
- [x] 각 Action의 대표 시점 Feature를 확인했다.
- [x] 실제로 잘 변한 Feature를 Action별로 2~3개 찾았다.
- [x] 5개 Sample이 부족하면 Sample 수를 늘려 다시 확인했다.
- [x] 두 Action의 시간 변화 차이를 설명했다.
- [x] `action_sequence_result.md`를 작성했다.
