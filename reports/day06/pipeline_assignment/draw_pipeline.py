"""4~6일차 Action 학습 파이프라인 그림 만들기 (PNG + draw.io).

같은 노드·연결 정의로 두 파일을 만든다.
- 3팀_Action_학습파이프라인.png    : 제출용 그림
- 3팀_Action_학습파이프라인.drawio : diagrams.net(draw.io)에서 열어 직접 수정할 수 있는 원본

실행: python reports/day06/pipeline_assignment/draw_pipeline.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
NAME = "3팀_Action_학습파이프라인"
W_CANVAS, H_CANVAS = 2460, 1450

# ---------------------------------------------------------------- 스타일
KIND = {
    #          fill       edge      matplotlib 모양  draw.io style
    "config": ("#eeeeee", "#555555", "note", "shape=note;size=14;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#555555;"),
    "raw": ("#ede9fe", "#7c3aed", "rect", "rounded=0;whiteSpace=wrap;html=1;fillColor=#ede9fe;strokeColor=#7c3aed;strokeWidth=2;"),
    "code": ("#dbeafe", "#2563eb", "round", "rounded=1;whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;strokeWidth=2;"),
    "lib": ("#e0f2fe", "#0284c7", "round_dash", "rounded=1;dashed=1;whiteSpace=wrap;html=1;fillColor=#e0f2fe;strokeColor=#0284c7;"),
    "csv": ("#fef3c7", "#d97706", "cyl", "shape=cylinder3;boundedLbl=1;size=8;whiteSpace=wrap;html=1;fillColor=#fef3c7;strokeColor=#d97706;"),
    "model": ("#dcfce7", "#16a34a", "hex", "shape=hexagon;perimeter=hexagonPerimeter2;size=0.08;whiteSpace=wrap;html=1;fillColor=#dcfce7;strokeColor=#16a34a;strokeWidth=2;"),
    "human": ("#fce7f3", "#db2777", "ellipse", "ellipse;whiteSpace=wrap;html=1;fillColor=#fce7f3;strokeColor=#db2777;strokeWidth=2;"),
}
RED = "#dc2626"
GRAY = "#6b7280"

# ---------------------------------------------------------------- 노드 (x, y, w, h, 종류, 글자)
N = {
    # ---- 5일차
    "project": (40, 250, 230, 70, "config", "configs/project.json\n(1일차 작성)\nYOLO 모델·imgsz·conf·device"),
    "mp4": (40, 96, 230, 86, "raw", "data/day05/videos/\nS01·S02·S03 MP4 18개\n(강사 제공 원본)"),
    "action": (40, 450, 230, 86, "config", "configs/action_classes.csv\n(4일차 9번 작성)\nA001~A006 → action_label"),
    "c25": (330, 110, 230, 56, "code", "25_check_day05_data.py\n영상·파일명 검사"),
    "inv": (620, 104, 230, 68, "csv", "reports/day05/tables/\nvideo_inventory.csv"),
    "c26": (330, 330, 230, 64, "code", "26_extract_keypoint_dataset.py\nFrame별 YOLO Pose"),
    "kp": (620, 270, 250, 96, "csv", "data/day05/keypoints/\nS0x/<label>/S0x_A00n_T01.csv × 18\n(메타 10 + 관절 85 = 95열)"),
    "kpman": (620, 430, 250, 68, "csv", "data/day05/manifests/\nkeypoint_dataset_manifest.csv"),
    "c27": (930, 432, 220, 64, "code", "27_keypoint_qa.py\nAUTO_PASS 13 / CHECK 5"),
    "qa": (1210, 330, 230, 68, "csv", "data/day05/qa/\nkeypoint_clip_qa.csv"),
    "review": (1210, 450, 230, 68, "csv", "data/day05/qa/\nkeypoint_clip_review.csv"),
    "human": (1205, 585, 240, 96, "human", "Human QA (사람)\n원본 MP4 확인 →\nhuman_decision 입력"),
    "c28": (1500, 360, 240, 64, "code", "28_validate_human_review.py\n입력 완료 검사"),
    "c29": (1500, 480, 240, 64, "code", "29_build_approved_manifest.py\nPASS만 승인"),
    "appr": (1790, 470, 230, 84, "csv", "data/day05/manifests/\napproved_clip_manifest.csv\n(18개 승인)"),
    "split_cfg": (1790, 620, 230, 86, "config", "configs/day05_subject_split.json\n(5일차 28번 작성)\nS01 Train / S02 Val / S03 Test"),
    "c30": (2060, 560, 200, 64, "code", "30_build_subject_split.py"),
    "s_train": (2290, 430, 150, 70, "csv", "splits/train.csv\nS01 · 6 Clip"),
    "s_val": (2290, 530, 150, 70, "csv", "splits/val.csv\nS02 · 6 Clip"),
    "s_test": (2290, 630, 150, 70, "csv", "splits/test.csv\nS03 · 6 Clip"),
    "c31": (2060, 760, 220, 64, "code", "31_check_split_leakage.py\nLeakage check: OK"),
    # ---- 6일차
    "base_cfg": (40, 960, 230, 86, "config", "configs/day06_baseline.json\n(6일차 11번 작성)\n30 Step · 300 Tree · seed 42"),
    "lib": (40, 1080, 230, 86, "lib", "src/sequence_utils.py\n(공통 함수)\n38-D Feature + 30 Step 보간"),
    "c34": (330, 1030, 230, 64, "code", "34_build_sequence_dataset.py\n1,140-D Sequence 생성"),
    "q_train": (620, 930, 270, 70, "csv", "data/day06/sequences/train.csv\nS01 · 6행 × (메타 8 + 1,140)"),
    "q_val": (620, 1040, 270, 70, "csv", "data/day06/sequences/val.csv\nS02 · 6행 × (메타 8 + 1,140)"),
    "q_test": (620, 1150, 270, 70, "csv", "data/day06/sequences/test.csv\nS03 · 6행 × (메타 8 + 1,140)"),
    "c35": (960, 930, 250, 64, "code", "35_train_rf_baseline.py\nmodel.fit(X_train, y_train)"),
    "model": (1280, 900, 250, 60, "model", "artifacts/day06/\nrf_action_baseline.joblib"),
    "fjson": (1280, 990, 250, 60, "model", "artifacts/day06/\nrf_feature_columns.json"),
    "c36": (1590, 1040, 230, 64, "code", "36_evaluate_validation.py\nS02 Validation"),
    "vout": (1880, 1020, 290, 100, "csv", "reports/day06/metrics/\nvalidation_predictions.csv\nvalidation_metrics.txt\nconfusion/validation_confusion_matrix.png"),
    "c37": (2210, 905, 220, 64, "code", "37_analyze_misclassified.py\n틀린 Clip 찾기"),
    "mis": (2210, 1020, 220, 84, "csv", "reports/day06/misclassified/\nvalidation_misclassified.csv"),
    "human2": (2205, 1140, 230, 84, "human", "사람: 오분류 Clip의\n원본 MP4 확인"),
    "c40": (1590, 1220, 230, 64, "code", "40_evaluate_test.py\nS03 Test (마지막 1회)"),
    "tout": (1880, 1200, 290, 100, "csv", "reports/day06/metrics/\ntest_predictions.csv\ntest_metrics.txt\nconfusion/test_confusion_matrix.png"),
}

# ---------------------------------------------------------------- 연결 (출발, 도착, 전달 데이터, action_label 경로?, 경유점, 점선?)
E = [
    ("project", "c26", "YOLO 설정", False, [], False),
    ("project", "c27", "keypoint_confidence 0.50 (저신뢰도 기준)", False, [(300, 232), (1040, 232)], False),
    ("mp4", "c25", "MP4 18개", False, [], False),
    ("mp4", "c26", "Frame", False, [], False),
    ("action", "c26", "A006 → wave", True, [], False),
    ("c25", "inv", "검사 결과", False, [], False),
    ("c26", "kp", "관절 좌표·Conf\n+ action_label", True, [], False, (590, 300)),
    ("c26", "kpman", "Clip 목록", True, [], False, (590, 418)),
    ("kpman", "c27", "", True, [], False),
    ("kp", "c27", "관절 Conf", False, [], True, (905, 395)),
    ("c27", "qa", "자동 QA", False, [], False, (1170, 385)),
    ("c27", "review", "작업표", True, [], False, (1180, 470)),
    ("review", "human", "검토 대상", False, [(1250, 560)], False, (1185, 556)),
    ("human", "review", "PASS·RESHOOT·EXCLUDE\n입력", False, [(1400, 560)], False, (1475, 590)),
    ("review", "c28", "", False, [], True),
    ("review", "c29", "", True, [], False),
    ("c29", "appr", "", True, [], False),
    ("appr", "c30", "", True, [], False),
    ("split_cfg", "c30", "", False, [], False),
    ("c30", "s_train", "", True, [], False),
    ("c30", "s_val", "", True, [], False),
    ("c30", "s_test", "", True, [], False),
    ("s_test", "c31", "3개 Split 검사", False, [], True, (2240, 725)),
    # 5일차 → 6일차
    ("s_test", "c34", "splits/train·val·test.csv (keypoint_csv 경로 + action_label)", True,
     [(2365, 870), (445, 870)], False),
    ("base_cfg", "c34", "Step·최소 Frame", False, [], False, (300, 1000)),
    ("lib", "c34", "함수 호출", False, [], False, (300, 1110)),
    ("c34", "q_train", "", True, [], False),
    ("c34", "q_val", "", True, [], False),
    ("c34", "q_test", "", True, [], False),
    ("q_train", "c35", "X_train·y_train", True, [], False, (925, 945)),
    ("base_cfg", "c35", "n_estimators · max_depth · random_state", False, [(155, 912), (1085, 912)], False),
    ("c35", "model", "", False, [], False),
    ("c35", "fjson", "", False, [], False),
    ("q_val", "c36", "X_val · y_true", True, [], False),
    ("model", "c36", "", False, [], False),
    ("fjson", "c36", "", False, [], False),
    ("c36", "vout", "", True, [], False),
    ("vout", "c37", "예측 결과", False, [(2025, 937)], False, (2110, 925)),
    ("s_val", "c37", "5일차 val.csv의\nvideo_path", False, [(2452, 565), (2452, 937)], False, (2395, 850)),
    ("c37", "mis", "", False, [], False),
    ("mis", "human2", "", False, [], False),
    ("q_test", "c40", "X_test · y_true", True, [], False),
    ("model", "c40", "", False, [(1480, 930), (1480, 1252)], False),
    ("fjson", "c40", "", False, [(1450, 1020), (1450, 1240)], False),
    ("c40", "tout", "", True, [], False),
]


def center(k):
    x, y, w, h, *_ = N[k]
    return x + w / 2, y + h / 2


def clip(k, toward):
    """노드 k의 테두리에서 toward 방향으로 나가는 점."""
    x, y, w, h, *_ = N[k]
    cx, cy = x + w / 2, y + h / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def edge_points(src, dst, way):
    pts_mid = list(way)
    first = pts_mid[0] if pts_mid else center(dst)
    last = pts_mid[-1] if pts_mid else center(src)
    return [clip(src, first)] + pts_mid + [clip(dst, last)]


# ---------------------------------------------------------------- PNG
def find_font():
    for name in ("Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Noto Serif CJK KR", "Noto Serif CJK JP"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return None


def draw_png():
    font = find_font()
    if font:
        plt.rcParams["font.family"] = font
    fig = plt.figure(figsize=(W_CANVAS / 100, H_CANVAS / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_CANVAS)
    ax.set_ylim(H_CANVAS, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # 구역 배경
    ax.add_patch(Rectangle((15, 45), W_CANVAS - 30, 805, facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.2, zorder=0))
    ax.add_patch(Rectangle((15, 885), W_CANVAS - 30, 440, facecolor="#f0fdf4", edgecolor="#bbf7d0", lw=1.2, zorder=0))
    ax.text(30, 66, "5일차 — 18개 MP4 → Keypoint Dataset → 자동 QA → Human QA → 승인 → Subject-wise Split",
            fontsize=15, weight="bold", color="#334155", va="center")
    ax.text(1000, 1300, "6일차 — Sequence Feature (30 Step × 38-D = 1,140) → Random Forest → Validation(S02) → Test(S03)",
            fontsize=15, weight="bold", color="#166534", va="center", ha="center")

    # 연결선
    for src, dst, label, red, way, dashed, *pos in E:
        pts = edge_points(src, dst, way)
        color = RED if red else GRAY
        lw = 2.4 if red else 1.4
        ls = (0, (4, 3)) if dashed else "-"
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="round")
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16), zorder=2)
        if label:
            if pos:
                mx, my = pos[0]
            else:
                segs = list(zip(pts[:-1], pts[1:]))
                (a, b) = max(segs, key=lambda s: (s[0][0] - s[1][0]) ** 2 + (s[0][1] - s[1][1]) ** 2)
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ax.text(mx, my, label, fontsize=8.8, color=color if red else "#374151", ha="center", va="center",
                    zorder=8, linespacing=1.2,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color if red else "#d1d5db", lw=0.6, alpha=0.95))

    # 노드
    for k, (x, y, w, h, kind, text) in N.items():
        fill, edge, shape, _ = KIND[kind]
        if shape in ("round", "round_dash"):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc=fill, ec=edge,
                                        lw=2, ls="--" if shape == "round_dash" else "-", zorder=4))
        elif shape == "rect":
            ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=edge, lw=2, zorder=4))
        elif shape == "note":
            f = 14
            ax.add_patch(Polygon([(x, y), (x + w - f, y), (x + w, y + f), (x + w, y + h), (x, y + h)], closed=True,
                                 fc=fill, ec=edge, lw=1.6, zorder=4))
            ax.plot([x + w - f, x + w - f, x + w], [y, y + f, y + f], color=edge, lw=1.2, zorder=5)
        elif shape == "cyl":
            e = 10
            ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=fill, ec="none", zorder=4))
            ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc=fill, ec=edge, lw=1.6, zorder=4))
            ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=fill, ec="none", zorder=4.1))
            ax.plot([x, x], [y + e / 2, y + h - e / 2], color=edge, lw=1.6, zorder=4.2)
            ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=edge, lw=1.6, zorder=4.2)
            ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=fill, ec=edge, lw=1.6, zorder=4.3))
        elif shape == "hex":
            d = 14
            ax.add_patch(Polygon([(x + d, y), (x + w - d, y), (x + w, y + h / 2), (x + w - d, y + h), (x + d, y + h),
                                  (x, y + h / 2)], closed=True, fc=fill, ec=edge, lw=2, zorder=4))
        elif shape == "ellipse":
            ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, fc=fill, ec=edge, lw=2, zorder=4))
        ax.text(x + w / 2, y + h / 2 + (3 if shape == "cyl" else 0), text, fontsize=9.6, ha="center", va="center",
                zorder=6, linespacing=1.35, weight="bold" if kind == "code" else "normal")

    # 범례
    lx, ly = 40, 1370
    ax.add_patch(Rectangle((lx - 10, ly - 18), 2390, 50, fc="white", ec="#d1d5db", lw=1, zorder=4))
    items = [("config", "직접 작성한 설정"), ("raw", "원본 영상(강사 제공)"), ("code", "Python 코드(번호)"),
             ("lib", "공통 함수 파일"), ("csv", "자동 생성 CSV·결과"), ("model", "학습 모델·모델 정보"),
             ("human", "사람이 하는 단계")]
    for i, (kind, text) in enumerate(items):
        cx = lx + i * 285
        cy = ly + 8
        fill, edge, *_ = KIND[kind]
        ax.add_patch(FancyBboxPatch((cx, cy - 8), 28, 16, boxstyle="round,pad=0,rounding_size=3", fc=fill, ec=edge,
                                    lw=1.6, zorder=5))
        ax.text(cx + 36, cy, text, fontsize=9.4, va="center", zorder=6)
    cx, cy = lx + 7 * 285 - 20, ly + 8
    ax.plot([cx, cx + 28], [cy, cy], color=RED, lw=2.4, zorder=6)
    ax.text(cx + 36, cy, "빨간 선: action_label이 함께 이동", fontsize=9.4, va="center", color=RED, zorder=6)
    ax.plot([cx, cx + 28], [cy + 0, cy + 0], color=RED, lw=2.4, zorder=6)
    ax.text(lx, ly - 32, "점선 = 검사·참고용으로 읽기   |   화살표 위 글자 = 전달되는 데이터", fontsize=9.4,
            color="#374151", va="center", zorder=6)

    path = OUT / f"{NAME}.png"
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return path, font


# ---------------------------------------------------------------- draw.io
def draw_drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    cells.append('<mxCell id="zone5" value="5일차 — Keypoint Dataset → QA → Split" style="rounded=0;whiteSpace=wrap;'
                 'html=1;fillColor=#f8fafc;strokeColor=#cbd5e1;align=left;verticalAlign=top;fontStyle=1;spacingLeft=10;" '
                 f'vertex="1" parent="1"><mxGeometry x="15" y="45" width="{W_CANVAS - 30}" height="805" as="geometry"/></mxCell>')
    cells.append('<mxCell id="zone6" value="6일차 — Sequence → Random Forest → Validation / Test" style="rounded=0;'
                 'whiteSpace=wrap;html=1;fillColor=#f0fdf4;strokeColor=#bbf7d0;align=left;verticalAlign=bottom;'
                 'fontStyle=1;spacingLeft=10;" vertex="1" parent="1"><mxGeometry x="15" y="885" '
                 f'width="{W_CANVAS - 30}" height="425" as="geometry"/></mxCell>')
    for k, (x, y, w, h, kind, text) in N.items():
        style = KIND[kind][3] + ("fontStyle=1;" if kind == "code" else "")
        value = escape(text).replace("\n", "&lt;br&gt;")
        cells.append(f'<mxCell id="{k}" value="{value}" style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, (src, dst, label, red, way, dashed, *_pos) in enumerate(E):
        color = RED if red else GRAY
        style = (f"edgeStyle=none;html=1;endArrow=block;endFill=1;strokeColor={color};"
                 f"strokeWidth={2.4 if red else 1.4};fontSize=10;fontColor={color if red else '#374151'};"
                 + ("dashed=1;" if dashed else ""))
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in way)
        geo = (f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>' if way
               else '<mxGeometry relative="1" as="geometry"/>')
        cells.append(f'<mxCell id="e{i}" value="{escape(label).replace(chr(10), ' ')}" style="{style}" edge="1" parent="1" '
                     f'source="{src}" target="{dst}">{geo}</mxCell>')
    xml = ('<mxfile host="app.diagrams.net"><diagram name="4-6일차 파이프라인">'
           f'<mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" pageWidth="{W_CANVAS}" pageHeight="{H_CANVAS}">'
           f'<root>{"".join(cells)}</root></mxGraphModel></diagram></mxfile>')
    path = OUT / f"{NAME}.drawio"
    path.write_text(xml, encoding="utf-8")
    return path


if __name__ == "__main__":
    png, font = draw_png()
    print("font:", font)
    print("[SAVED]", png)
    print("[SAVED]", draw_drawio())
