#!/usr/bin/env python3
"""Generate a static SVG figure for Flow-SDE noise injection behavior.

No third-party Python plotting packages are required. The script writes a
vector SVG that can be converted to PNG/PDF with ImageMagick.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent
SVG_PATH = OUT_DIR / "flow_sde_noise_injection.svg"

W, H = 2400, 1350

COLORS = {
    "bg": "#F6F4EF",
    "ink": "#20232A",
    "muted": "#6D737C",
    "grid": "#D7D6CE",
    "panel": "#FFFFFF",
    "panel2": "#ECEBE4",
    "single": "#D0604C",
    "every": "#2D6F9F",
    "accent": "#2F8F6B",
    "gold": "#D8A441",
    "soft_single": "#F2D1C8",
    "soft_every": "#C9DDEB",
    "soft_accent": "#CFE5D8",
}

FONT = "DejaVu Sans, Nimbus Sans, Arial, sans-serif"
MONO = "DejaVu Sans Mono, monospace"


def fmt_attrs(**attrs: object) -> str:
    parts = []
    for key, value in attrs.items():
        if value is None:
            continue
        key = key.replace("_", "-")
        parts.append(f'{key}="{escape(str(value))}"')
    return " ".join(parts)


def tag(name: str, content: str | None = None, **attrs: object) -> str:
    attr = fmt_attrs(**attrs)
    if content is None:
        return f"<{name} {attr}/>"
    return f"<{name} {attr}>{content}</{name}>"


def text(
    x: float,
    y: float,
    s: str,
    size: int = 28,
    fill: str = COLORS["ink"],
    weight: int | str = 400,
    anchor: str = "start",
    family: str = FONT,
    opacity: float | None = None,
    rotate: float | None = None,
) -> str:
    attrs = {
        "x": f"{x:.1f}",
        "y": f"{y:.1f}",
        "font_family": family,
        "font_size": size,
        "font_weight": weight,
        "fill": fill,
        "text_anchor": anchor,
        "opacity": opacity,
    }
    if rotate is not None:
        attrs["transform"] = f"rotate({rotate:.1f} {x:.1f} {y:.1f})"
    return tag("text", escape(s), **attrs)


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = COLORS["ink"],
    width: float = 2,
    opacity: float | None = None,
    dash: str | None = None,
) -> str:
    return tag(
        "line",
        None,
        x1=f"{x1:.1f}",
        y1=f"{y1:.1f}",
        x2=f"{x2:.1f}",
        y2=f"{y2:.1f}",
        stroke=stroke,
        stroke_width=width,
        stroke_linecap="round",
        opacity=opacity,
        stroke_dasharray=dash,
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str | None = None,
    width: float = 1,
    rx: float = 0,
    opacity: float | None = None,
) -> str:
    return tag(
        "rect",
        None,
        x=f"{x:.1f}",
        y=f"{y:.1f}",
        width=f"{w:.1f}",
        height=f"{h:.1f}",
        rx=f"{rx:.1f}",
        fill=fill,
        stroke=stroke,
        stroke_width=width if stroke else None,
        opacity=opacity,
    )


def circle(
    x: float,
    y: float,
    r: float,
    fill: str,
    stroke: str | None = None,
    width: float = 1,
    opacity: float | None = None,
) -> str:
    return tag(
        "circle",
        None,
        cx=f"{x:.1f}",
        cy=f"{y:.1f}",
        r=f"{r:.1f}",
        fill=fill,
        stroke=stroke,
        stroke_width=width if stroke else None,
        opacity=opacity,
    )


def polyline(points: list[tuple[float, float]], stroke: str, width: float, dash: str | None = None) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return tag(
        "polyline",
        None,
        points=pts,
        fill="none",
        stroke=stroke,
        stroke_width=width,
        stroke_linejoin="round",
        stroke_linecap="round",
        stroke_dasharray=dash,
    )


def path(d: str, stroke: str, width: float = 2, fill: str = "none", opacity: float | None = None) -> str:
    return tag(
        "path",
        None,
        d=d,
        fill=fill,
        stroke=stroke,
        stroke_width=width,
        stroke_linecap="round",
        stroke_linejoin="round",
        opacity=opacity,
    )


def arrow(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 3) -> str:
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    head = 15
    left = (
        x2 - head * math.cos(ang - math.pi / 6),
        y2 - head * math.sin(ang - math.pi / 6),
    )
    right = (
        x2 - head * math.cos(ang + math.pi / 6),
        y2 - head * math.sin(ang + math.pi / 6),
    )
    d = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f} M {left[0]:.1f} {left[1]:.1f} L {x2:.1f} {y2:.1f} L {right[0]:.1f} {right[1]:.1f}"
    return path(d, color, width)


def plot_points(
    xs: list[float], ys: list[float | None], map_x, map_y
) -> list[list[tuple[float, float]]]:
    segs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if y is None:
            if current:
                segs.append(current)
                current = []
            continue
        current.append((map_x(x), map_y(y)))
    if current:
        segs.append(current)
    return segs


def make_svg() -> str:
    xs = [0.5, 1, 2, 2.5, 3, 3.5, 4, 5]
    single = [0.006, 0.012, 0.047, 0.119, 0.228, 0.353, 0.493, 0.846]
    every: list[float | None] = [0.017, 0.024, 0.056, None, 0.137, None, 0.255, 0.298]

    # Canvas and headline.
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        tag("rect", None, x=0, y=0, width=W, height=H, fill=COLORS["bg"]),
        text(96, 118, "Flow-SDE noise injection", 54, COLORS["ink"], 700),
        text(98, 166, "Why every-step sampling can end with lower action std than single-step sampling", 28, COLORS["muted"]),
        text(2180, 118, "TACO / RLinf", 24, COLORS["muted"], 500, "end", MONO),
    ]

    # Main plot card.
    px, py, pw, ph = 95, 230, 1390, 880
    svg.append(rect(px, py, pw, ph, COLORS["panel"], stroke="#E0DED5", width=2, rx=8))
    svg.append(text(px + 45, py + 62, "Observed final action std", 34, COLORS["ink"], 700))
    svg.append(text(px + 45, py + 102, "single stochastic denoise step vs stochastic transition at every denoise step", 23, COLORS["muted"]))

    left, right = px + 110, px + pw - 75
    top, bottom = py + 150, py + ph - 115
    x_min, x_max = 0.5, 5.0
    y_min, y_max = 0.0, 0.9

    def mx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (right - left)

    def my(y: float) -> float:
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    # Regime shading.
    svg.append(rect(mx(0.5), top, mx(2.25) - mx(0.5), bottom - top, COLORS["soft_every"], opacity=0.35))
    svg.append(rect(mx(2.75), top, mx(5.0) - mx(2.75), bottom - top, COLORS["soft_accent"], opacity=0.32))
    svg.append(line(mx(2.55), top - 20, mx(2.55), bottom + 18, COLORS["accent"], 3, dash="10 12", opacity=0.8))
    svg.append(text(mx(1.35), top + 36, "+ noise dominates", 22, COLORS["every"], 700, "middle"))
    svg.append(text(mx(3.9), top + 36, "eta^2 correction dominates", 22, COLORS["accent"], 700, "middle"))
    svg.append(text(mx(2.55) + 12, top + 70, "crossover", 19, COLORS["accent"], 500, "start", MONO))

    # Grid and axes.
    for yt in [0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
        y = my(yt)
        svg.append(line(left, y, right, y, COLORS["grid"], 1.2))
        label = "0" if yt == 0 else f"{yt:.2f}"
        svg.append(text(left - 26, y + 8, label, 20, COLORS["muted"], 400, "end", MONO))
    for xt in xs:
        x = mx(xt)
        svg.append(line(x, top, x, bottom, COLORS["grid"], 0.9, opacity=0.55))
        svg.append(text(x, bottom + 44, f"{xt:g}", 22, COLORS["ink"], 500, "middle", MONO))
    svg.append(line(left, bottom, right, bottom, COLORS["ink"], 2.2))
    svg.append(line(left, top, left, bottom, COLORS["ink"], 2.2))
    svg.append(text((left + right) / 2, bottom + 86, "noise level eta", 24, COLORS["ink"], 600, "middle"))
    svg.append(text(left - 83, (top + bottom) / 2, "final action std", 24, COLORS["ink"], 600, "middle", rotate=-90))

    # Curves.
    single_pts = [(mx(x), my(y)) for x, y in zip(xs, single)]
    svg.append(polyline(single_pts, COLORS["single"], 5.5))
    for seg in plot_points(xs, every, mx, my):
        svg.append(polyline(seg, COLORS["every"], 5.5))

    # Missing markers for every-step.
    for x in [2.5, 3.5]:
        svg.append(line(mx(x) - 12, my(0.085) - 12, mx(x) + 12, my(0.085) + 12, COLORS["every"], 3, opacity=0.55))
        svg.append(line(mx(x) - 12, my(0.085) + 12, mx(x) + 12, my(0.085) - 12, COLORS["every"], 3, opacity=0.55))

    # Markers and value labels.
    for x, y in zip(xs, single):
        svg.append(circle(mx(x), my(y), 11, COLORS["panel"], COLORS["single"], 4))
        if x in [2.5, 3, 4, 5]:
            svg.append(text(mx(x), my(y) - 20, f"{y:.3f}", 19, COLORS["single"], 700, "middle", MONO))
    for x, y in zip(xs, every):
        if y is None:
            continue
        svg.append(circle(mx(x), my(y), 10, COLORS["every"], "#FFFFFF", 3))
        if x in [0.5, 2, 3, 4, 5]:
            svg.append(text(mx(x), my(y) + 34, f"{y:.3f}", 18, COLORS["every"], 700, "middle", MONO))

    # Legend.
    lx, ly = left + 24, bottom - 30
    svg.append(rect(lx - 20, ly - 40, 505, 62, COLORS["panel"], stroke="#E6E3DA", width=1.5, rx=6, opacity=0.92))
    svg.append(line(lx, ly, lx + 70, ly, COLORS["single"], 6))
    svg.append(circle(lx + 35, ly, 8, COLORS["panel"], COLORS["single"], 3))
    svg.append(text(lx + 88, ly + 8, "single-step faithful train", 22, COLORS["ink"], 600))
    svg.append(line(lx + 325, ly, lx + 395, ly, COLORS["every"], 6))
    svg.append(circle(lx + 360, ly, 8, COLORS["every"], "#FFFFFF", 2.5))
    svg.append(text(lx + 413, ly + 8, "every-step", 22, COLORS["ink"], 600))

    # Callout arrow.
    svg.append(path(f"M {mx(3.05):.1f} {my(0.23):.1f} C {mx(3.35):.1f} {my(0.42):.1f}, {mx(4.15):.1f} {my(0.43):.1f}, {mx(4.85):.1f} {my(0.31):.1f}", COLORS["accent"], 3.5, opacity=0.9))
    svg.append(text(mx(4.1), my(0.43) - 12, "repeated correction bends the chain down", 21, COLORS["accent"], 700, "middle"))

    # Right mechanism panel.
    rx, ry, rw, rh = 1540, 230, 765, 880
    svg.append(rect(rx, ry, rw, rh, COLORS["panel"], stroke="#E0DED5", width=2, rx=8))
    svg.append(text(rx + 48, ry + 62, "Mechanism in one transition", 32, COLORS["ink"], 700))
    svg.append(text(rx + 48, ry + 104, "Flow-SDE is not just additive noise", 23, COLORS["muted"]))

    eqy = ry + 182
    svg.append(rect(rx + 44, eqy - 58, rw - 88, 122, COLORS["panel2"], stroke="#DDD8CC", width=1.4, rx=7))
    svg.append(text(rx + 78, eqy - 10, "x_next = ODE_next", 28, COLORS["ink"], 700, family=MONO))
    svg.append(text(rx + 405, eqy - 10, "- c_i · noise_pred", 28, COLORS["accent"], 700, family=MONO))
    svg.append(text(rx + 78, eqy + 34, "+ alpha_i · eps", 28, COLORS["every"], 700, family=MONO))
    svg.append(text(rx + 405, eqy + 34, "alpha_i ∝ eta     c_i ∝ eta²", 22, COLORS["muted"], 600, family=MONO))

    # Two force bars.
    bx, by = rx + 70, ry + 330
    svg.append(text(bx, by - 34, "competing terms", 24, COLORS["ink"], 700))
    svg.append(rect(bx, by, 560, 34, COLORS["soft_every"], rx=4))
    svg.append(rect(bx, by, 250, 34, COLORS["every"], rx=4))
    svg.append(text(bx + 280, by + 24, "+ injected noise grows linearly", 20, COLORS["ink"], 600))
    svg.append(rect(bx, by + 66, 560, 34, COLORS["soft_accent"], rx=4))
    svg.append(rect(bx, by + 66, 455, 34, COLORS["accent"], rx=4))
    svg.append(text(bx + 486, by + 90, "- mean correction grows quadratically", 20, COLORS["ink"], 600, "end"))

    # Chain diagrams.
    cy = ry + 540
    svg.append(text(rx + 48, cy - 70, "sampling paths", 24, COLORS["ink"], 700))
    svg.append(text(rx + 48, cy - 31, "single-step", 21, COLORS["single"], 700))
    svg.append(text(rx + 48, cy + 122, "every-step", 21, COLORS["every"], 700))

    start_x = rx + 190
    step_gap = 82
    for row, yy, color, mode in [
        ("single", cy - 38, COLORS["single"], "single"),
        ("every", cy + 115, COLORS["every"], "every"),
    ]:
        for i in range(6):
            x = start_x + i * step_gap
            fill = COLORS["panel2"]
            stroke = "#C8C4BA"
            if (mode == "single" and i == 4) or mode == "every":
                fill = COLORS["soft_single"] if mode == "single" else COLORS["soft_every"]
                stroke = color
            svg.append(circle(x, yy, 20, fill, stroke, 3))
            if i < 5:
                svg.append(arrow(x + 24, yy, x + step_gap - 25, yy, COLORS["muted"], 2.2))
        svg.append(circle(start_x + 6 * step_gap + 16, yy, 24, COLORS["ink"], None))
        svg.append(text(start_x + 6 * step_gap + 16, yy + 8, "a", 22, "#FFFFFF", 700, "middle", family=MONO))

    svg.append(text(start_x + 4 * step_gap, cy - 78, "late impulse can survive", 18, COLORS["single"], 700, "middle"))
    svg.append(text(start_x + 2.7 * step_gap, cy + 70, "noise is repeatedly reprocessed", 18, COLORS["every"], 700, "middle"))
    svg.append(text(start_x + 5.0 * step_gap, cy + 70, "correction repeats", 18, COLORS["accent"], 700, "middle"))

    # Bottom conclusion strip.
    sy = 1165
    svg.append(rect(95, sy, 2210, 112, "#20232A", rx=8))
    svg.append(text(140, sy + 46, "Takeaway", 26, "#FFFFFF", 700))
    svg.append(text(140, sy + 84, "Every-step adds more stochastic transitions, but Flow-SDE also applies an eta² mean correction at every transition.", 25, "#F4F1E8", 500))
    svg.append(text(1375, sy + 84, "At large eta, correction can dominate and lower the final action std.", 25, "#CFE5D8", 700))

    # Tiny footer.
    svg.append(text(96, 1320, "Data from eta sweep table. Missing every-step entries shown with crosses.", 20, COLORS["muted"], 400, family=MONO))
    svg.append(text(2305, 1320, "generated: flow_sde_noise_figure", 20, COLORS["muted"], 400, "end", MONO))

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    SVG_PATH.write_text(make_svg(), encoding="utf-8")
    print(SVG_PATH)


if __name__ == "__main__":
    main()
