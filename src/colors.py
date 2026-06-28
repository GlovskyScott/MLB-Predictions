"""Team-color helpers for the dark dashboard."""


def hex_to_rgb_str(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def bar_color(primary: str, secondary: str) -> str:
    """Return a legible team color for dark backgrounds.

    Picks the brighter of primary/secondary, then blends toward white until the
    result meets the minimum readable luminance.
    """
    MIN_LUM = 0.28

    def _lum(hex_color: str) -> float:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
        return 0.299 * r + 0.587 * g + 0.114 * b

    color = primary if _lum(primary) >= _lum(secondary) else secondary
    if _lum(color) >= MIN_LUM:
        return color

    # Blend toward white in 5% steps until readable
    h = color.lstrip('#')
    r0, g0, b0 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    for step in range(5, 100, 5):
        t = step / 100.0
        r = min(int(r0 + (255 - r0) * t), 255)
        g = min(int(g0 + (255 - g0) * t), 255)
        b = min(int(b0 + (255 - b0) * t), 255)
        if 0.299 * (r / 255) + 0.587 * (g / 255) + 0.114 * (b / 255) >= MIN_LUM:
            return f"#{r:02x}{g:02x}{b:02x}"
    return '#888888'
