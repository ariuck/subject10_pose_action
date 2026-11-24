# Day 08 Tracking Cases

같은 코드(`47_track_sbu_video.py`, YOLO11n-pose + ByteTrack)로 SBU 영상 3개를 추적하고 `48`번으로 Log를 요약했다.
숫자로 문제 구간을 찾은 뒤, 결과 MP4의 해당 Frame을 직접 보고 판단했다.

## 1. Shaking Hands

- 전체 Frame: 30
- Unique Track IDs: 2
- Two-ID Frame Ratio: 1.0000
- Extra ID: 0
- ID 1: 30 / 30 Frame (box_conf 0.92~0.94)
- ID 2: 30 / 30 Frame (box_conf 0.90~0.95)
- Log 해석: 두 Track이 전체 영상에서 유지됨
- 결과영상 확인: Frame 0·15·29 모두 오른쪽 흰 옷 사람이 ID 1, 왼쪽 검은 옷 사람이 ID 2였다. 악수하며 손이 맞닿은 Frame 15에서도 두 BBox가 따로 유지됐다.
- 최종 판단: ID 유지, Track Lost 없음, ID Switch 없음

## 2. Hugging

- 전체 Frame: 33
- Unique Track IDs: 4
- Two-ID Frame Ratio: 0.5455
- Extra ID: 2
- ID 1: 16 / 33 Frame (Frame 0~6, 20, 25~32)
- ID 2: 32 / 33 Frame (Frame 5에만 빠짐)
- ID 3: 4 Frame (Frame 6~11, box_conf 평균 0.387)
- ID 4: 3 Frame (Frame 20~23, box_conf 평균 0.298)
- Log 해석: 두 사람이 안는 Frame 7~24 동안 ID 1이 거의 사라지고, 낮은 Confidence의 짧은 Track 2개가 생겼다.
- ID 3 / ID 4가 나타난 장면:
  - ID 3 (Frame 6~11): 두 사람이 막 겹치기 시작할 때, 겹친 몸 부분에 생긴 BBox
  - ID 4 (Frame 20~23): 안은 상태에서 앞쪽 검은 옷 사람의 상체 일부에 생긴 BBox
- 결과영상에서 본 현상:
  - 안고 있는 동안에는 흰 옷 사람(ID 2)의 BBox 하나가 두 사람을 거의 다 덮었다. 검은 옷 사람은 흰 옷 사람 몸에 가려져 따로 검출되지 않았다.
  - Frame 25에 두 사람이 떨어지자 검은 옷 사람에게 다시 **원래 번호 ID 1**이 붙었다. 처음(Frame 0)과 끝(Frame 32)에 검은 옷 = ID 1, 흰 옷 = ID 2로 같았다.
- 최종 판단:
  - ID 1: Track Lost 후 같은 ID로 다시 연결됨 (ByteTrack이 잠깐 사라진 Track을 기억해 둠)
  - ID 3·ID 4: 겹친 구간의 낮은 Confidence **중복 Track**
  - 두 사람의 ID가 서로 바뀌는 ID Switch는 보이지 않았다

## 3. Kicking

- 전체 Frame: 18
- Unique Track IDs: 2
- Two-ID Frame Ratio: 1.0000
- Extra ID: 0
- ID 1: 18 / 18 Frame (box_conf 0.91~0.94)
- ID 2: 18 / 18 Frame (box_conf 0.93~0.94)
- Log 해석: 빠른 움직임에서도 두 Track 유지
- 결과영상 확인: 발을 차는 Frame 9에서 ID 1(검은 옷)의 BBox가 다리를 따라 오른쪽으로 크게 넓어졌지만 ID는 그대로였다. 두 사람 사이 거리가 떨어져 있어 BBox가 겹치지 않았다.
- 최종 판단: ID 유지, Track Lost 없음, ID Switch 없음

## 4. 세 영상 비교

| 영상 | Frame | Unique ID | Two-ID Ratio | Extra ID | 판단 |
|---|---:|---:|---:|---:|---|
| Shaking Hands | 30 | 2 | 1.0000 | 0 | 안정적 |
| Hugging | 33 | 4 | 0.5455 | 2 | 겹침 구간에서 Track Lost + 중복 Track |
| Kicking | 18 | 2 | 1.0000 | 0 | 안정적 |

- 기본 Tracking: Shaking Hands → 안정적
- Occlusion: Hugging → 가장 큰 Tracking 변화 발생
- 빠른 움직임: Kicking → 이번 영상에서는 안정적
- 가장 눈에 띈 문제: Hugging에서 ID 1이 18 Frame 동안 사라지고 ID 3·4가 잠깐 생김
- 교재 결과와 차이: 교재는 ID 1이 24 Frame, 추가 ID 번호가 4·5, Two-ID Ratio 0.6364였다. 추가 Track이 생긴 구간(Frame 6~11, 20~23)과 Confidence(0.387, 0.298)는 같았고, ID 1이 끊긴 Frame 수만 달랐다. 겹친 구간에서는 작은 계산 차이(GPU·라이브러리 버전)에도 결과가 흔들린다는 뜻으로 본다.

## 5. Action Recognition과의 연결

- ID가 유지되면 같은 사람의 Pose를 같은 Buffer에 계속 쌓을 수 있다.
- Hugging의 ID 1처럼 1초 넘게 사라지면 그 사람의 Buffer에는 그 구간 Feature가 비고, 중복 Track(ID 3·4)은 새 사람처럼 WARMING_UP부터 시작한다.
- ID가 서로 바뀌면 다른 사람의 Feature가 섞인다. 이번 SBU 영상에서는 이 경우는 없었다.
