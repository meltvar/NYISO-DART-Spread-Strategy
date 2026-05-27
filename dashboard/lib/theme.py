"""
Light theme: CSS polish + helper components.

Pages call `apply_theme()` once at the top, before any other Streamlit calls.
Helpers `kpi_tile()`, `status_strip()`, `section_header()`, and `trade_tape()`
emit the custom-styled markup used across the dashboard.
"""
from __future__ import annotations

import textwrap

import streamlit as st


# ── Palette (light) ─────────────────────────────────────────────────────────
BG          = "#ffffff"
BG_PANEL    = "#fafafa"
BG_PANEL_HI = "#f3f6f9"
BORDER      = "#e1e4e8"
GRID        = "#eeeeee"

TEXT        = "#1a1a1a"
TEXT_DIM    = "#586069"
TEXT_FAINT  = "#959da5"

GREEN       = "#2e7d32"
GREEN_DIM   = "#81c784"
RED         = "#c62828"
RED_DIM     = "#ef5350"
AMBER       = "#f57c00"
BLUE        = "#1565c0"
PURPLE      = "#6a1b9a"
PRIMARY_BLUE = "#1f4e79"


CSS = f"""
<style>
/* ── KPI tiles ───────────────────────────────────────────────────────── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 10px;
    margin: 8px 0 4px 0;
}}

.kpi-tile {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
}}

.kpi-tile::before {{
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: {GREEN_DIM};
}}

.kpi-tile.red::before  {{ background: {RED_DIM}; }}
.kpi-tile.amber::before {{ background: {AMBER}; }}
.kpi-tile.blue::before  {{ background: {BLUE}; }}
.kpi-tile.dim::before   {{ background: {TEXT_FAINT}; }}

.kpi-label {{
    font-size: 0.68rem;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
    font-weight: 500;
}}

.kpi-value {{
    font-size: 1.4rem;
    font-weight: 600;
    color: {TEXT};
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}}

.kpi-value.green {{ color: {GREEN}; }}
.kpi-value.red   {{ color: {RED}; }}
.kpi-value.amber {{ color: {AMBER}; }}

.kpi-sub {{
    font-size: 0.72rem;
    color: {TEXT_DIM};
    margin-top: 2px;
}}

/* ── Status strip ────────────────────────────────────────────────────── */
.status-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    padding: 8px 14px;
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    font-size: 0.78rem;
    color: {TEXT_DIM};
    margin-bottom: 14px;
}}

.status-item {{
    display: flex;
    align-items: center;
    gap: 6px;
}}

.status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: {GREEN};
}}

.status-dot.amber {{ background: {AMBER}; }}
.status-dot.red   {{ background: {RED}; }}

.status-key {{
    color: {TEXT_FAINT};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.7rem;
}}

.status-val {{
    color: {TEXT};
    font-variant-numeric: tabular-nums;
}}

/* ── Section header ──────────────────────────────────────────────────── */
.section-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 22px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid {BORDER};
}}

.section-header .title {{
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {TEXT_DIM};
    font-weight: 600;
}}

.section-header .sub {{
    font-size: 0.74rem;
    color: {TEXT_FAINT};
}}

/* ── Trade tape ──────────────────────────────────────────────────────── */
.tape {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 8px 12px;
    font-family: 'Menlo', 'Consolas', monospace;
    font-size: 0.78rem;
    max-height: 280px;
    overflow-y: auto;
}}

.tape-row {{
    display: grid;
    grid-template-columns: 140px 80px 50px 1fr 100px;
    gap: 12px;
    padding: 4px 0;
    border-bottom: 1px dotted {BORDER};
    font-variant-numeric: tabular-nums;
}}

.tape-row:last-child {{ border-bottom: none; }}

.tape-ts    {{ color: {TEXT_DIM}; }}
.tape-zone  {{ color: {TEXT}; font-weight: 500; }}
.tape-side  {{ font-weight: 600; }}
.tape-side.dec {{ color: {GREEN}; }}
.tape-side.inc {{ color: {AMBER}; }}
.tape-proba {{ color: {TEXT_DIM}; }}
.tape-pnl.win  {{ color: {GREEN}; text-align: right; }}
.tape-pnl.loss {{ color: {RED};   text-align: right; }}

/* Tighter metric styling on Streamlit defaults */
.stMetric {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
}}

[data-testid="stMetricValue"] {{
    font-variant-numeric: tabular-nums;
}}
</style>
"""


def apply_theme() -> None:
    """Inject CSS polish. Call once at the top of each page."""
    st.markdown(CSS, unsafe_allow_html=True)


def kpi_tile(label: str, value: str, sub: str | None = None,
             color: str | None = None, accent: str = "green") -> str:
    """Return HTML for one KPI tile. Use `st.markdown(..., unsafe_allow_html=True)`.

    color: 'green', 'red', 'amber' for value tint; None = default text color.
    accent: left-border color, 'green' | 'red' | 'amber' | 'blue' | 'dim'.
    """
    accent_cls = "" if accent == "green" else f" {accent}"
    color_cls = f" {color}" if color else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return textwrap.dedent(f"""
    <div class="kpi-tile{accent_cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value{color_cls}">{value}</div>
      {sub_html}
    </div>
    """)


def kpi_grid(tiles: list[str]) -> None:
    """Render a list of kpi_tile() HTML strings as a responsive grid."""
    html = '<div class="kpi-grid">' + "".join(tiles) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def status_strip(items: list[tuple[str, str, str]]) -> None:
    """Render the top status bar. Each item is (key, value, dot_color).
    dot_color: 'green', 'amber', 'red'.
    """
    pieces = []
    for key, val, dot in items:
        dot_cls = "status-dot" if dot == "green" else f"status-dot {dot}"
        pieces.append(
            f'<div class="status-item"><span class="{dot_cls}"></span>'
            f'<span class="status-key">{key}</span>'
            f'<span class="status-val">{val}</span></div>'
        )
    html = '<div class="status-strip">' + "".join(pieces) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, sub: str = "") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="section-header">'
        f'<div class="title">{title}</div>{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def trade_tape(rows: list[dict]) -> None:
    """Render the trade tape. Each row dict: ts, zone, side, proba, payoff."""
    lines = []
    for r in rows:
        side_cls = "dec" if r["side"] == "pos" else "inc"
        side_label = "DEC" if r["side"] == "pos" else "INC"
        pnl_cls = "win" if r["payoff"] >= 0 else "loss"
        ts = r["ts"].strftime("%Y-%m-%d %H:%M") if hasattr(r["ts"], "strftime") else str(r["ts"])
        pnl_str = f"+${r['payoff']:,.2f}" if r["payoff"] >= 0 else f"-${abs(r['payoff']):,.2f}"
        lines.append(
            f'<div class="tape-row">'
            f'<span class="tape-ts">{ts}</span>'
            f'<span class="tape-zone">{r["zone"]}</span>'
            f'<span class="tape-side {side_cls}">{side_label}</span>'
            f'<span class="tape-proba">p={r["proba"]:.3f}  dart={r["dart"]:+.2f}</span>'
            f'<span class="tape-pnl {pnl_cls}">{pnl_str}</span>'
            f'</div>'
        )
    html = '<div class="tape">' + "".join(lines) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
