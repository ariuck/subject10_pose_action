# Day 04 Team Mini Challenge

교재 **조합 B (bend_return VS sit_stand)** 를 선택했다.
두 영상은 모션캡처 촬영 때 같은 팀원을 정면 카메라로 직접 찍은 영상이다.

## 1. 촬영한 Action

- Action 1: `bend_return` — 서 있다가 허리를 앞으로 깊게 굽혔다가 다시 선다
- 파일명: `challenge_bend_return.mp4` (원본 `20261113_164329.mp4`, 7.7초, 1080x1920, 60fps)
- Action 2: `sit_stand` — 서 있다가 뒤의 책상 위로 올라앉았다가 다시 내려와 선다
- 파일명: `challenge_sit_stand.mp4` (원본 `20261113_154727.mp4`, 11.1초, 1080x1920, 60fps)
- 설정 파일: `configs/day04_challenge_actions.csv`

교재 촬영 기준과 다른 점:

- 두 영상 모두 세로 영상이다 (교재는 가로 영상 권장).
- 두 영상은 서로 다른 강의실에서 찍었다. 같은 사람, 같은 정면 카메라, 카메라 고정, 전신이 보이는 조건은 같다.
- `sit_stand`는 일반 의자가 아니라 높은 책상 위에 올라앉는 동작이라, 앉았을 때 골반이 오히려 화면 위쪽으로 올라갔다.

## 2. 대표 Frame 확인

`21_extract_clip_keyframes.py` (`reports/day04/challenge/keyframes/`)

### Action 1 — bend_return
- Start: 정면을 보고 팔을 내린 채 서 있음
- Middle: 허리를 앞으로 깊게 숙이고 한 팔을 앞으로 뻗음
- End: 다시 똑바로 서 있음

### Action 2 — sit_stand
- Start: 정면을 보고 서 있음
- Middle: 책상 위에 걸터앉아 휴대폰을 보고 있음
- End: 내려와서 다시 서 있음

> 21번 코드는 휴대폰 영상의 마지막 Frame에서 멈추지 않도록 고친 버전으로 실행했다.
> 영상 정보의 Frame 수(463 / 665)보다 실제로 읽히는 Frame 수(459 / 662)가 적어,
> End는 실제 마지막 Frame(458 / 661)으로 저장되었다.

## 3. Sequence 품질

`22_build_action_sequences.py` → `23_summarize_action_sequences.py`

| 항목 | bend_return | sit_stand |
|---|---:|---:|
| rows | 459 | 662 |
| pose_ok_ratio | 1.0 | 1.0 |
| feature_ok_ratio | 1.0 | 1.0 |
| mean_keypoint_conf | 0.9048 | 0.9652 |
| mean_low_keypoint_count | 1.296 | 0.115 |
| mean_adjacent_feature_change | 0.129393 | 0.079080 |

- Pose 실패 없음, Feature 생성 실패 없음.
- bend_return은 허리를 깊게 숙인 3.5~4.5초에 평균 Confidence가 0.75까지 내려갔다. 얼굴이 아래를 향해 눈·귀가 잘 보이지 않았기 때문이다.
- 인접 Row 변화량은 교재 본 실습(0.10~0.22)과 비슷한 범위였다. 60fps 영상이라 Frame 사이 간격이 짧아서 조금 작게 나왔다.

## 4. 실제로 잘 변한 Feature

`24_inspect_sequence_stages.py --samples 5`, `--samples 9` + 0.5초 간격 확인

### Action 1 — bend_return
- Feature 1: `nose_y` (정규화 좌표, 골반 중심 기준)
- Feature 2: `hip_center_y` (원본 Pixel)
- Feature 3: `right_wrist_y`
- 변화:
  - nose_y: 0초 -2.16 → 3.0초 -0.22 → **4.5초 +0.70** → 5.5초 -0.66 → 7.5초 -2.18
    - 머리가 골반보다 한참 위에 있다가, 허리를 숙이면서 골반 높이 아래까지 내려갔다가, 다시 처음 높이로 돌아왔다.
  - right_wrist_y: -0.10 → 0.37~0.47 (3.8~4.8초) → -0.06
  - 무릎 각도는 처음부터 끝까지 147~180°로 거의 펴져 있었다. 무릎을 굽히지 않고 허리만 숙였다.

### Action 2 — sit_stand
- Feature 1: `hip_center_y` (원본 Pixel)
- Feature 2: `right_knee_angle`
- Feature 3: `left_wrist_y`
- 변화:
  - hip_center_y: 0~1.5초 1117~1129 → 2.5초 1046 → 3.0~10.0초 1028~1045 유지 → 11.0초 1118
    - 책상 위로 올라앉아 골반이 약 85px 화면 위로 올라갔다가, 내려오면서 원래 높이로 돌아왔다.
  - right_knee_angle: 177° → 앉아 있는 동안 155~166° → 177°
  - left_wrist_y: -0.1 → 앉아서 휴대폰을 보는 동안 -0.78 ~ -1.05 → -0.07

## 5. 두 Action 비교

- 두 Action에서 가장 다르게 보인 Feature:
  - bend_return은 **nose_y**(머리 높이)가 크게 변했다. -2.2 → +0.7 → -2.2로, 몸통이 앞으로 접혔다가 펴졌다.
  - sit_stand는 nose_y가 -2.4 → -1.7 → -2.4로 조금만 변했다. 대신 **hip_center_y와 무릎 각도**가 앉아 있는 7초 동안 다른 값으로 유지되었다.
  - bend_return은 무릎이 거의 그대로였고, sit_stand는 무릎이 굽었다. 같은 "자세를 낮추는" 동작이라도 바뀌는 관절이 달랐다.
- 시작과 마지막은 비슷했지만 중간이 달랐던 부분:
  - 두 영상 모두 시작과 끝은 정면으로 선 자세다. nose_y는 -2.16 / -2.18, -2.37 / -2.41이고, 무릎은 약 174~180°였다.
  - 처음·마지막 Frame만 보면 두 Action을 구분할 수 없고, 중간 구간에서만 다르게 보였다.
- 5개 Sample로 부족했다면 어떻게 확인했는가:
  - 24번 코드 출력에는 nose가 나오지 않아서, bend_return에서 머리가 내려가는 모습은 5개·9개 Sample로 직접 볼 수 없었다. 그래서 Sequence CSV에서 nose_y를 0.5초 간격으로 따로 출력해 확인했다.
  - sit_stand는 정면에서 찍어 굽힌 무릎이 카메라 쪽을 향했다. 그래서 무릎 각도가 실제보다 펴진 값(155~166°)으로 나왔다. 9개 Sample로 늘려 앉아 있는 구간(2.7~9.6초) 전체에서 값이 유지되는 것을 확인했다.
  - 교재 `sit_stand`처럼 "앉으면 골반이 내려간다"고 생각했지만, 이번 영상은 높은 책상에 올라앉아 골반이 올라갔다. 예상에 맞추지 않고 실제 값대로 기록했다.

## 6. 최종 정리

- Pose와 Action의 차이: Pose는 "허리를 숙인 한 순간"이고, Action은 서기 → 숙이기 → 다시 서기처럼 Pose가 시간에 따라 변하는 전체 흐름이다.
- Clip과 Sequence의 차이: Clip은 MP4 파일 하나(`challenge_bend_return.mp4`)이고, Sequence는 그 Clip의 Frame마다 만든 38-D Feature를 시간 순서대로 쌓은 CSV(`bend_return_sequence.csv`, 459행)이다.
- 한 Frame만으로 Action을 설명하기 어려운 이유: 두 Action 모두 시작·끝 Frame은 거의 같은 서 있는 자세였다. 숙였다가 돌아온 것인지, 앉았다가 일어난 것인지는 중간 Frame들의 변화를 순서대로 봐야만 알 수 있었다.
- 오늘 만든 데이터의 한계:
  - Action마다 한 사람이 한 번 찍은 영상 하나뿐이라, 이 Feature 변화가 다른 사람·다른 속도에서도 똑같이 나온다고 할 수 없다.
  - 두 영상의 장소가 다르고 세로 영상이라 교재 촬영 기준과 완전히 같지 않다.
  - 정면 영상에서는 앞뒤 방향 움직임(무릎을 앞으로 굽힘, 허리를 앞으로 숙임)이 짧게 보여 관절각도가 실제보다 작게 변했다.

## Mini Challenge 완료 체크

- [x] 두 개의 Action MP4를 직접 촬영했다. (모션캡처 촬영 때 찍은 정면 영상 사용)
- [x] 두 영상이 정상적으로 열리는지 확인했다. (open_ok=1, read_ok=1, 60fps, 1080x1920)
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

> 우리 팀(3팀) 동작으로 따로 진행한 추가 과제(뒤로 기절하기 vs 앞으로 넘어지기)는 `reports/day04/team3/fall_sequence_result.md`에 있다.
