"""Generate architecture and workflow diagrams for the paper using PIL (white background)."""
from PIL import Image, ImageDraw, ImageFont

OUT_W = 2200
LRG = 200
FONT_PATH = r"C:\Windows\Fonts\arialbd.ttf"
FONT_PATH_N = r"C:\Windows\Fonts\arial.ttf"


def font(size, bold=True):
    try:
        return ImageFont.truetype(FONT_PATH if bold else FONT_PATH_N, size)
    except OSError:
        return ImageFont.load_default()


def text_wrap(draw, text, fs, max_w):
    f = font(fs)
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=f) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_box(draw, cx, cy, w, h, lines, fs, fc, ec, text_fill="#111111", bold=True, lw=3, round_r=14):
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=round_r, fill=fc, outline=ec, width=lw)
    f = font(fs, bold)
    total = sum(draw.textbbox((0, 0), ln, font=f)[3] - draw.textbbox((0, 0), ln, font=f)[1] for ln in lines)
    y = cy - total / 2
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=f)
        draw.text((cx - (bbox[2] - bbox[0]) / 2, y - bbox[1]), ln, font=f, fill=text_fill)
        y += (bbox[3] - bbox[1]) + 4


def draw_arrow(draw, p1, p2, width=4, color="#333333", head=16):
    x1, y1 = p1
    x2, y2 = p2
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    p = math.pi / 6
    for s in (p, -p):
        draw.line([(x2, y2), (x2 - head * math.cos(ang - s), y2 - head * math.sin(ang - s))], fill=color, width=width)


def layer_label(draw, y, text):
    f = font(60, bold=True)
    bbox = draw.textbbox((0, 0), text, font=f)
    draw.text(((OUT_W - (bbox[2] - bbox[0])) / 2, y), text, font=f, fill="#555555")


def make(width, height):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    return img, draw


# ─────────────── Architecture ───────────────
H = 2300
img, draw = make(OUT_W, H)

boxes = [
    # (cx, cy, w, h, text, fc, ec)
    (330, 200, 560, 300, "Synthetic cohort\ngenerator", "#fff1e6", "#c75b12"),
    (900, 200, 520, 300, "Pima Indians\n(real public)", "#fff1e6", "#c75b12"),
    (1500, 200, 560, 300, "User-provided CSV\n(path / URL / upload)", "#fff1e6", "#c75b12"),
    (860, 620, 760, 300, "Validation & standardisation\n(prepare_dataset)", "#e8f4f8", "#12707f"),
    (2000, 620, 640, 300, "Knowledge graph\n(observed + assumed edges)", "#f3e8ee", "#7a1f53"),
    (870, 1010, 760, 300, "Feature engineering\n(mean / std / slope / curvature)", "#e8f4f8", "#12707f"),
    (1990, 1010, 660, 300, "Patient trajectory clustering\n(PTC) — cluster-ID feature", "#f3e8ee", "#7a1f53"),
    (400, 1400, 620, 240, "Baselines\nLR · RF · XGB · MLP", "#e6f4ea", "#1e7d38"),
    (1020, 1400, 560, 240, "Graph models\nGNN · GNN+KG · +longitudinal", "#e6f4ea", "#1e7d38"),
    (1630, 1400, 620, 240, "Novel models\nTAGNN · MHFIN", "#fff7e0", "#a86d00"),
    (2300, 1400, 620, 240, "CCF — calibrated\nstacked fusion", "#fff7e0", "#a86d00"),
    (1400, 1780, 1900, 260, "Evaluation & artefacts\npatient-level split · accuracy · precision · recall · F1 · ROC-AUC · PR-AUC · Brier · log loss", "#fafafa", "#444444"),
    (850, 2070, 1050, 240, "FastAPI REST backend\n(src/api.py)", "#e8f0fe", "#1a3f8f"),
    (1900, 2070, 1000, 240, "React dashboard\n(Vite)", "#e8f0fe", "#1a3f8f"),
]

for (cx, cy, w, h, text, fc, ec) in boxes:
    lines = text_wrap(draw, text, 58, int(w * 0.9))
    while len(lines) > 2:
        w += 120
        lines = text_wrap(draw, text, 58, int(w * 0.9))
    draw_box(draw, cx, cy, w, h, lines, 58, fc, ec)

# arrows
for x in range(900 - 280, 900 - 10, 60):
    draw_arrow(draw, (330 + 280, 350), (x, 470))  # synth → validation
for x in range(900 + 10, 900 + 280, 60):
    draw_arrow(draw, (900, 350), (x, 470))  # pima → validation
draw_arrow(draw, (1500 - 280, 350), (900 + 280, 470))  # csv → validation
draw_arrow(draw, (1500 + 280, 350), (2000 - 320, 470))  # csv → KG
draw_arrow(draw, (860 + 380, 770), (2000 - 320, 770))  # standardise → KG
draw_arrow(draw, (860, 770), (860, 860))  # down
draw_arrow(draw, (860 + 380, 920 + 200), (870, 1160 - 1), )  # standardise → feats
draw_arrow(draw, (2000, 770), (2000, 860))
draw_arrow(draw, (1990 + 330, 1150), (1700, 1280))
draw_arrow(draw, (870 + 380, 1160), (1100, 1280))
draw_arrow(draw, (500, 1520), (700, 1640))
draw_arrow(draw, (1100, 1520), (1150, 1650))
draw_arrow(draw, (1700, 1520), (1600, 1650))
draw_arrow(draw, (2300, 1520), (1800, 1650))
draw_arrow(draw, (700, 1780 - 1), (850, 1780 - 1))
draw_arrow(draw, (1150, 1780 - 1), (1100, 1780 - 1))
draw_arrow(draw, (1600, 1660), (1550, 1780 - 1))
draw_arrow(draw, (1800, 1660), (1750, 1780 - 1))
draw_arrow(draw, (900, 1910), (850 - 100, 1950))
draw_arrow(draw, (1800, 1910), (1800, 1950))
draw_arrow(draw, (1425, 1950), (1450, 1950), width=4)

layer_label(draw, 30, "Data layer")
layer_label(draw, 480, "Pre-processing & knowledge layer")
layer_label(draw, 1080, "Modelling layer")
layer_label(draw, 1600, "Evaluation & serving layer")

img.save(r"C:\Users\subal\OneDrive\Desktop\project\paper\figures\architecture.png")
print("architecture.png written")

# ─────────────── Workflow ───────────────
H = 900
img, draw = make(OUT_W, H)
steps = [
    ("1. Data loading", "#fff1e6", "#c75b12"),
    ("2. Knowledge graph", "#f3e8ee", "#7a1f53"),
    ("3. Feature engineering", "#e8f4f8", "#12707f"),
    ("4. PTC cluster features", "#f3e8ee", "#7a1f53"),
    ("5. Patient-level split 75/25", "#e8f4f8", "#12707f"),
    ("6. Train 10 models", "#e6f4ea", "#1e7d38"),
    ("7. CCF fusion", "#fff7e0", "#a86d00"),
    ("8. Evaluate & save artefacts", "#fafafa", "#444444"),
    ("9. Serve via REST + React", "#e8f0fe", "#1a3f8f"),
]
n = len(steps)
W = 2000 / n - 30
x = 100 + W / 2
cy = 450
for label, fc, ec in steps:
    lines = text_wrap(draw, label, 48, int(W * 0.9))
    draw_box(draw, x, cy, W, 340, lines, 48, fc, ec)
    if x < OUT_W - 120:
        draw_arrow(draw, (x + W / 2, cy), (x + W / 2 + 30, cy), width=5, head=22)
    x += W + 30

img.save(r"C:\Users\subal\OneDrive\Desktop\project\paper\figures\workflow.png")
print("workflow.png written")
