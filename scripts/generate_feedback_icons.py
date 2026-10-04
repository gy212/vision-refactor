"""Generate local static PNG icons for student feedback states (Style A, geometric line)."""
from pathlib import Path
import math
from PIL import Image, ImageDraw

COLORS = {
    "candidate": (146, 64, 14, 255),      # #92400E
    "not_observed": (22, 101, 52, 255),   # #166534
    "unable": (71, 85, 105, 255),         # #475569
    "pending_rule": (29, 78, 216, 255),   # #1D4ED8
}

SIZES = [24, 32, 48]
SCALE = 16  # 24 * 16 = 384x384 canvas
CANVAS_SIZE = 24 * SCALE


def draw_capsule_line(draw: ImageDraw.ImageDraw, p1, p2, width, color):
    """Draw line with round end caps."""
    draw.line([p1, p2], fill=color, width=int(width))
    r = width / 2.0
    draw.ellipse([p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r], fill=color)
    draw.ellipse([p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r], fill=color)


def draw_candidate(draw: ImageDraw.ImageDraw, color):
    # Rounded triangle: (12, 3.5), (3.5, 19.5), (20.5, 19.5)
    sw = 2.0 * SCALE
    # Vertices
    v_top = (12.0 * SCALE, 4.0 * SCALE)
    v_bl = (4.0 * SCALE, 19.5 * SCALE)
    v_br = (20.0 * SCALE, 19.5 * SCALE)

    draw_capsule_line(draw, v_top, v_bl, sw, color)
    draw_capsule_line(draw, v_bl, v_br, sw, color)
    draw_capsule_line(draw, v_br, v_top, sw, color)

    # Exclamation mark
    p_ex_top = (12.0 * SCALE, 8.5 * SCALE)
    p_ex_bottom = (12.0 * SCALE, 13.5 * SCALE)
    draw_capsule_line(draw, p_ex_top, p_ex_bottom, sw, color)

    # Exclamation dot
    dot_center = (12.0 * SCALE, 16.5 * SCALE)
    dot_r = 1.0 * SCALE
    draw.ellipse([dot_center[0] - dot_r, dot_center[1] - dot_r,
                  dot_center[0] + dot_r, dot_center[1] + dot_r], fill=color)


def draw_not_observed(draw: ImageDraw.ImageDraw, color):
    sw = 2.0 * SCALE
    # Outer circle
    cx, cy, cr = 12.0 * SCALE, 12.0 * SCALE, 9.5 * SCALE
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=color, width=int(sw))

    # Checkmark inside
    p1 = (7.0 * SCALE, 12.0 * SCALE)
    p2 = (10.5 * SCALE, 15.5 * SCALE)
    p3 = (17.0 * SCALE, 9.0 * SCALE)
    draw_capsule_line(draw, p1, p2, sw, color)
    draw_capsule_line(draw, p2, p3, sw, color)


def draw_unable(draw: ImageDraw.ImageDraw, color):
    sw = 2.0 * SCALE
    # Outer circle
    cx, cy, cr = 12.0 * SCALE, 12.0 * SCALE, 9.5 * SCALE
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], outline=color, width=int(sw))

    # Question mark hook
    # Sample points along smooth question mark arc
    points = []
    # arc from angle ~ 180 to 0
    qcx, qcy, qr = 12.0 * SCALE, 9.0 * SCALE, 2.5 * SCALE
    for deg in range(180, -20, -10):
        rad = math.radians(deg)
        points.append((qcx + qr * math.cos(rad), qcy - qr * math.sin(rad)))
    # curve down to stem
    points.append((12.5 * SCALE, 11.5 * SCALE))
    points.append((12.0 * SCALE, 12.5 * SCALE))
    points.append((12.0 * SCALE, 13.5 * SCALE))

    for i in range(len(points) - 1):
        draw_capsule_line(draw, points[i], points[i + 1], sw, color)

    # Dot
    dot_center = (12.0 * SCALE, 16.5 * SCALE)
    dot_r = 1.0 * SCALE
    draw.ellipse([dot_center[0] - dot_r, dot_center[1] - dot_r,
                  dot_center[0] + dot_r, dot_center[1] + dot_r], fill=color)


def draw_pending_rule(draw: ImageDraw.ImageDraw, color):
    sw = 2.0 * SCALE
    # Clipboard outer frame
    # Main board: left=4.5, top=5.5, right=19.5, bottom=20.5, rx=2.0
    x0, y0 = 4.5 * SCALE, 5.5 * SCALE
    x1, y1 = 19.5 * SCALE, 20.5 * SCALE
    rx = 2.0 * SCALE
    draw.rounded_rectangle([x0, y0, x1, y1], radius=int(rx), outline=color, width=int(sw))

    # Top clip
    cx0, cy0 = 8.5 * SCALE, 2.5 * SCALE
    cx1, cy1 = 15.5 * SCALE, 5.5 * SCALE
    crx = 1.2 * SCALE
    # Erase top part of board under clip
    draw.rectangle([cx0 - 0.5 * SCALE, y0 - sw / 2, cx1 + 0.5 * SCALE, y0 + sw / 2], fill=(0, 0, 0, 0))
    draw.rounded_rectangle([cx0, cy0, cx1, cy1], radius=int(crx), outline=color, width=int(sw))

    # Inner checkmark / task lines
    p1 = (7.5 * SCALE, 12.5 * SCALE)
    p2 = (10.5 * SCALE, 15.5 * SCALE)
    p3 = (16.5 * SCALE, 9.5 * SCALE)
    draw_capsule_line(draw, p1, p2, sw, color)
    draw_capsule_line(draw, p2, p3, sw, color)


DRAW_FUNCS = {
    "candidate": draw_candidate,
    "not_observed": draw_not_observed,
    "unable": draw_unable,
    "pending_rule": draw_pending_rule,
}


def generate_all(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for state, color in COLORS.items():
        # Render high-res image
        img_hr = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img_hr)
        DRAW_FUNCS[state](draw, color)

        for size in SIZES:
            # Resize with LANCZOS for high-quality downsampling
            resized = img_hr.resize((size, size), Image.Resampling.LANCZOS)
            # Save both naming conventions: status_<state>_<size>.png and <state>_<size>.png
            p1 = out_dir / f"status_{state}_{size}.png"
            p2 = out_dir / f"{state}_{size}.png"
            resized.save(p1, "PNG")
            resized.save(p2, "PNG")
            generated.extend([p1, p2])

    print(f"Generated {len(generated)} PNG icons in {out_dir}")
    return generated


if __name__ == "__main__":
    generate_all(Path(__file__).resolve().parent.parent / "assets" / "feedback")
