"""
Daedalus — Scientific Data Analysis Platform
Dark modern UI · Sidebar navigation · Plotly visualizations
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Ensure project root is on sys.path so analytics/ and functions/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics import eda  # noqa: E402
from functions.data_prep_f.data_helpers import column_transform  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daedalus — Scientific Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Dark Netflix-inspired theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #141414;
        color: #e5e5e5;
    }

    /* ── Main content area ── */
    .main .block-container {
        background-color: #141414;
        padding-top: 2rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }

    /* ── Section header ── */
    .section-header {
        background: #1f1f1f;
        border-left: 4px solid #e50914;
        color: #ffffff;
        padding: 10px 18px;
        border-radius: 0 6px 6px 0;
        margin: 24px 0 12px 0;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* ── Surface cards ── */
    .card {
        background: #1f1f1f;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 22px 26px;
        margin-bottom: 16px;
    }
    .card h4 { margin-top: 0; margin-bottom: 10px; color: #fff !important; }

    /* ── Upload placeholder ── */
    .upload-placeholder {
        background: #1f1f1f;
        border: 2px dashed #3a3a3a;
        border-radius: 12px;
        padding: 56px 24px;
        text-align: center;
        color: #777;
        margin-top: 16px;
    }
    .upload-placeholder h3 { color: #aaa !important; margin-bottom: 8px; }

    /* ── Error box ── */
    .error-box {
        background: #250a0a;
        border: 1px solid #e50914;
        border-radius: 8px;
        padding: 12px 18px;
        color: #ff7070;
        font-size: 0.9rem;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #1f1f1f;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { color: #888 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }

    /* ── Streamlit radio in sidebar ── */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        color: #ccc !important;
        padding: 6px 0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 4px;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background-color: #1f1f1f !important;
        color: #e5e5e5 !important;
        border-radius: 6px;
    }
    .streamlit-expanderContent {
        background-color: #1a1a1a !important;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #2a2a2a; margin: 24px 0; }

    /* ── Landing hero ── */
    .landing-hero {
        text-align: center;
        padding: 72px 40px 48px;
    }
    .landing-hero h1 {
        font-size: 3.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 30%, #e50914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    .landing-hero .tagline {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #e50914;
        margin-bottom: 20px;
    }
    .landing-hero .subtitle {
        font-size: 1.1rem;
        color: #999;
        max-width: 580px;
        margin: 0 auto 36px;
        line-height: 1.75;
    }
    .get-started-badge {
        display: inline-block;
        background: #1f1f1f;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 14px 32px;
        color: #ccc;
        font-size: 0.95rem;
    }
    .get-started-badge .accent { color: #e50914; font-weight: 700; }

    /* ── Download button row ── */
    .dl-row { display: flex; gap: 10px; margin-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMATIONS = ["None", "log", "boxcox", "yeojohnson", "sqrt"]
_NONE_SENTINEL  = "— none —"

# Color palettes for qualitative (categorical) plots
QUALITATIVE_PALETTES: dict[str, list[str]] = {
    "Daedalus Red":  ["#e50914", "#ff6b6b", "#ff9999", "#c0392b", "#922b21", "#7b241c"],
    "Dark24":        px.colors.qualitative.Dark24,
    "Bold":          px.colors.qualitative.Bold,
    "Vivid":         px.colors.qualitative.Vivid,
    "Safe":          px.colors.qualitative.Safe,
    "Pastel":        px.colors.qualitative.Pastel,
}

# Colorscales for continuous / matrix plots
SEQUENTIAL_SCALES = ["Reds", "Blues", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"]
DIVERGING_SCALES  = ["RdBu", "Spectral", "RdYlGn", "PiYG", "PRGn"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _friendly_error(exc: Exception) -> str:
    """Convert a raw exception into a plain-language message."""
    msg = str(exc)
    if "could not convert string to float" in msg or "could not be interpreted" in msg:
        return "One of the selected columns contains text values. Please choose a numeric column."
    if "all values must be positive" in msg or "strictly positive" in msg:
        return (
            "The selected transformation requires all values to be positive. "
            "Try 'yeojohnson' instead, or remove zero/negative values first."
        )
    if "not enough values to unpack" in msg or "only integers" in msg:
        return "Not enough data to draw this plot. Check that the column has sufficient values."
    if "KeyError" in type(exc).__name__:
        return f"Column not found: {msg}. Please re-select your columns."
    return f"An unexpected error occurred: {msg}"


def _load_data(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type '.{ext}'. Please upload a CSV or Excel file.")


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _opt(value: str):
    """Return None when the sentinel 'none' option is selected."""
    return None if value == _NONE_SENTINEL else value


def _apply_transform(df: pd.DataFrame, col: str, transformation: str) -> pd.DataFrame:
    """Apply a mathematical transformation to a column; no-op when 'None'."""
    if transformation and transformation != "None":
        df = column_transform(data=df, column=col, transformation=transformation)
    return df


def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme Plotly styling
# ─────────────────────────────────────────────────────────────────────────────

def _dark(fig: go.Figure, title="", xlabel="", ylabel="") -> go.Figure:
    """Apply the Daedalus dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#1f1f1f",
        plot_bgcolor="#1f1f1f",
        font=dict(color="#e5e5e5", family="Inter, Segoe UI, sans-serif", size=12),
        title=dict(
            text=title,
            font=dict(color="#ffffff", size=15),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(color="#bbb")),
            gridcolor="#2a2a2a",
            linecolor="#444",
            zerolinecolor="#444",
            tickcolor="#666",
            tickfont=dict(color="#999"),
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(color="#bbb")),
            gridcolor="#2a2a2a",
            linecolor="#444",
            zerolinecolor="#444",
            tickcolor="#666",
            tickfont=dict(color="#999"),
        ),
        legend=dict(
            bgcolor="#1f1f1f",
            bordercolor="#333",
            borderwidth=1,
            font=dict(color="#ccc"),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        hoverlabel=dict(
            bgcolor="#2a2a2a",
            bordercolor="#444",
            font=dict(color="#e5e5e5"),
        ),
    )
    return fig


def _download_buttons(fig: go.Figure, key: str) -> None:
    """Render HTML and PNG download buttons for a Plotly figure."""
    col_html, col_png = st.columns(2)
    html_bytes = pio.to_html(fig, include_plotlyjs="cdn", full_html=True).encode("utf-8")
    with col_html:
        st.download_button(
            "⬇ Download HTML",
            data=html_bytes,
            file_name=f"{key}.html",
            mime="text/html",
            key=f"dl_html_{key}",
        )
    with col_png:
        try:
            png_bytes = pio.to_image(fig, format="png", width=900, height=520, scale=2)
            st.download_button(
                "⬇ Download PNG",
                data=png_bytes,
                file_name=f"{key}.png",
                mime="image/png",
                key=f"dl_png_{key}",
            )
        except Exception:
            st.caption("PNG export: install `kaleido` to enable.")


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Plot Functions — all return go.Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(
    df, x, y,
    color_col=None, transformation="None",
    title="", xlabel="", ylabel="",
    palette="Dark24",
) -> go.Figure:
    df = _apply_transform(df, x, transformation)
    colors = QUALITATIVE_PALETTES.get(palette, QUALITATIVE_PALETTES["Dark24"])
    fig = px.scatter(
        df, x=x, y=y, color=color_col,
        color_discrete_sequence=colors,
        opacity=0.78,
    )
    fig.update_traces(marker=dict(size=6))
    return _dark(fig, title or f"Scatter: {x} vs {y}", xlabel or x, ylabel or y)


def plot_histogram(
    df, x,
    color_col=None, transformation="None",
    title="", xlabel="", ylabel="",
    palette="Dark24",
) -> go.Figure:
    df = _apply_transform(df, x, transformation)
    colors = QUALITATIVE_PALETTES.get(palette, QUALITATIVE_PALETTES["Dark24"])
    fig = px.histogram(
        df, x=x, color=color_col,
        color_discrete_sequence=colors,
        nbins=30, opacity=0.85,
    )
    fig.update_layout(bargap=0.04)
    return _dark(fig, title or f"Histogram of {x}", xlabel or x, ylabel or "Count")


def plot_boxplot(
    df, x,
    group_col=None, color_col=None, transformation="None",
    title="", xlabel="", ylabel="",
    palette="Dark24",
) -> go.Figure:
    df = _apply_transform(df, x, transformation)
    colors = QUALITATIVE_PALETTES.get(palette, QUALITATIVE_PALETTES["Dark24"])
    if group_col:
        fig = px.box(df, x=group_col, y=x, color=color_col, color_discrete_sequence=colors)
    else:
        fig = px.box(df, y=x, color=color_col, color_discrete_sequence=colors)
    return _dark(fig, title or f"Box Plot of {x}", xlabel or (group_col or ""), ylabel or x)


def plot_violin(
    df, x, group_col,
    color_col=None, transformation="None",
    title="", xlabel="", ylabel="",
    palette="Dark24",
) -> go.Figure:
    df = _apply_transform(df, x, transformation)
    colors = QUALITATIVE_PALETTES.get(palette, QUALITATIVE_PALETTES["Dark24"])
    fig = px.violin(
        df, x=group_col, y=x, color=color_col,
        color_discrete_sequence=colors,
        box=True, points="outliers",
    )
    return _dark(fig, title or f"Violin: {x} by {group_col}", xlabel or group_col, ylabel or x)


def plot_bar(
    df, x,
    color_col=None,
    title="", xlabel="", ylabel="",
    palette="Dark24",
) -> go.Figure:
    colors = QUALITATIVE_PALETTES.get(palette, QUALITATIVE_PALETTES["Dark24"])
    if color_col and color_col != x:
        cross = (
            pd.crosstab(df[x], df[color_col])
            .reset_index()
            .melt(id_vars=x, var_name=color_col, value_name="count")
        )
        fig = px.bar(
            cross, x=x, y="count", color=color_col,
            barmode="group", color_discrete_sequence=colors,
        )
    else:
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, "count"]
        fig = px.bar(counts, x=x, y="count", color_discrete_sequence=[colors[0]])
    return _dark(fig, title or f"Bar Chart of {x}", xlabel or x, ylabel or "Count")


def plot_ecdf(
    df, x,
    color_col=None, transformation="None",
    title="", xlabel="", ylabel="",
    palette="Dark24",
) -> go.Figure:
    df = _apply_transform(df, x, transformation)
    colors = QUALITATIVE_PALETTES.get(palette, QUALITATIVE_PALETTES["Dark24"])
    fig = px.ecdf(df, x=x, color=color_col, color_discrete_sequence=colors)
    return _dark(fig, title or f"ECDF of {x}", xlabel or x, ylabel or "Cumulative Probability")


def plot_heatmap(
    df, x, y,
    title="", xlabel="", ylabel="",
    colorscale="Reds",
) -> go.Figure:
    cross = pd.crosstab(df[x], df[y])
    fig = px.imshow(
        cross,
        color_continuous_scale=colorscale,
        text_auto=True,
        aspect="auto",
    )
    fig = _dark(fig, title or f"Heatmap: {x} vs {y}", xlabel or y, ylabel or x)
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color="#888"),
            title=dict(font=dict(color="#bbb")),
        )
    )
    return fig


def plot_correlation_matrix(
    df, cols,
    title="",
    colorscale="RdBu",
) -> go.Figure:
    corr = df[cols].corr()
    fig = px.imshow(
        corr,
        color_continuous_scale=colorscale,
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig = _dark(fig, title or "Correlation Matrix")
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color="#888"),
            title=dict(font=dict(color="#bbb")),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Auto-overview (max 2 plots)
# ─────────────────────────────────────────────────────────────────────────────

def _overview_plots(df: pd.DataFrame) -> None:
    """Auto-generate up to 2 overview charts on data upload."""
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    candidates: list[tuple[str, str]] = []
    if num_cols:
        candidates.append((num_cols[0], "num"))
    if cat_cols:
        candidates.append((cat_cols[0], "cat"))
    elif len(num_cols) >= 2:
        candidates.append((num_cols[1], "num"))
    candidates = candidates[:2]

    if not candidates:
        st.info("No columns available to preview.")
        return

    cols = st.columns(len(candidates))
    for idx, (col_name, kind) in enumerate(candidates):
        with cols[idx]:
            try:
                if kind == "num":
                    fig = plot_histogram(df, x=col_name)
                else:
                    fig = plot_bar(df, x=col_name)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not generate preview for **{col_name}**: {_friendly_error(exc)}")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Plot Builder
# ─────────────────────────────────────────────────────────────────────────────

def _plot_customization_widgets(prefix: str) -> tuple[str, str, str, str]:
    """Render shared customization widgets; return (title, xlabel, ylabel, palette)."""
    user_title  = st.text_input("Plot title (optional)",   key=f"{prefix}_title")
    user_xlabel = st.text_input("X-axis label (optional)", key=f"{prefix}_xlabel")
    user_ylabel = st.text_input("Y-axis label (optional)", key=f"{prefix}_ylabel")
    palette     = st.selectbox("Color palette", list(QUALITATIVE_PALETTES.keys()), key=f"{prefix}_palette")
    return user_title, user_xlabel, user_ylabel, palette


def _custom_plot_builder(df: pd.DataFrame) -> None:
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    PLOT_TYPES = [
        "Scatter",
        "Histogram",
        "Box Plot",
        "Violin",
        "Bar Chart",
        "ECDF",
        "Heatmap",
        "Correlation Matrix",
    ]

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("##### Configure Plot")
        plot_type = st.selectbox("Plot type", PLOT_TYPES, key="plot_type_select")
        st.markdown("---")

        # ── Scatter ───────────────────────────────────────────────────────────
        if plot_type == "Scatter":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns.")
                return
            x           = st.selectbox("X axis", num_cols, key="sc_x")
            y           = st.selectbox("Y axis", [c for c in num_cols if c != x], key="sc_y")
            color_col   = st.selectbox("Color by (optional)", [_NONE_SENTINEL] + cat_cols, key="sc_color")
            transf      = st.selectbox("Transform X", TRANSFORMATIONS, key="sc_trans")
            t, xl, yl, pal = _plot_customization_widgets("sc")
            generate = st.button("Generate", key="btn_scatter", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            fig = plot_scatter(df, x, y, _opt(color_col), transf, t, xl, yl, pal)
                            st.plotly_chart(fig, use_container_width=True)
                            _download_buttons(fig, f"scatter_{x}_{y}")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )

        # ── Histogram ─────────────────────────────────────────────────────────
        elif plot_type == "Histogram":
            if not num_cols:
                st.warning("No numeric columns found.")
                return
            x         = st.selectbox("Variable", num_cols, key="hist_x")
            color_col = st.selectbox("Split by (optional)", [_NONE_SENTINEL] + cat_cols, key="hist_color")
            transf    = st.selectbox("Transform", TRANSFORMATIONS, key="hist_trans")
            t, xl, yl, pal = _plot_customization_widgets("hist")
            generate = st.button("Generate", key="btn_histogram", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            fig = plot_histogram(df, x, _opt(color_col), transf, t, xl, yl, pal)
                            st.plotly_chart(fig, use_container_width=True)
                            _download_buttons(fig, f"histogram_{x}")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )

        # ── Box Plot ──────────────────────────────────────────────────────────
        elif plot_type == "Box Plot":
            if not num_cols:
                st.warning("No numeric columns found.")
                return
            x         = st.selectbox("Numeric variable", num_cols, key="box_x")
            group_col = st.selectbox("Group by (optional)", [_NONE_SENTINEL] + cat_cols, key="box_group")
            color_col = st.selectbox("Color by (optional)", [_NONE_SENTINEL] + cat_cols, key="box_color")
            transf    = st.selectbox("Transform", TRANSFORMATIONS, key="box_trans")
            t, xl, yl, pal = _plot_customization_widgets("box")
            generate = st.button("Generate", key="btn_boxplot", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            fig = plot_boxplot(df, x, _opt(group_col), _opt(color_col), transf, t, xl, yl, pal)
                            st.plotly_chart(fig, use_container_width=True)
                            _download_buttons(fig, f"boxplot_{x}")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )

        # ── Violin ────────────────────────────────────────────────────────────
        elif plot_type == "Violin":
            if not num_cols:
                st.warning("No numeric columns found.")
                return
            if not cat_cols:
                st.warning(
                    "A violin plot needs at least one categorical column to define the groups. "
                    "Check that your dataset includes a text or category column."
                )
                return
            x         = st.selectbox("Numeric variable", num_cols, key="vio_x")
            group_col = st.selectbox("Grouping variable", cat_cols, key="vio_group")
            color_col = st.selectbox("Color by (optional)", [_NONE_SENTINEL] + cat_cols, key="vio_color")
            transf    = st.selectbox("Transform", TRANSFORMATIONS, key="vio_trans")
            t, xl, yl, pal = _plot_customization_widgets("vio")
            generate = st.button("Generate", key="btn_violin", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            fig = plot_violin(df, x, group_col, _opt(color_col), transf, t, xl, yl, pal)
                            st.plotly_chart(fig, use_container_width=True)
                            _download_buttons(fig, f"violin_{x}_{group_col}")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )

        # ── Bar Chart ─────────────────────────────────────────────────────────
        elif plot_type == "Bar Chart":
            if not cat_cols:
                st.warning("No categorical columns found.")
                return
            x         = st.selectbox("Categorical variable", cat_cols, key="bar_x")
            color_col = st.selectbox("Color by (optional)", [_NONE_SENTINEL] + cat_cols, key="bar_color")
            t, xl, yl, pal = _plot_customization_widgets("bar")
            generate = st.button("Generate", key="btn_bar", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            fig = plot_bar(df, x, _opt(color_col), t, xl, yl, pal)
                            st.plotly_chart(fig, use_container_width=True)
                            _download_buttons(fig, f"bar_{x}")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )

        # ── ECDF ──────────────────────────────────────────────────────────────
        elif plot_type == "ECDF":
            if not num_cols:
                st.warning("No numeric columns found.")
                return
            x         = st.selectbox("Variable", num_cols, key="ecdf_x")
            color_col = st.selectbox("Color by (optional)", [_NONE_SENTINEL] + cat_cols, key="ecdf_color")
            transf    = st.selectbox("Transform", TRANSFORMATIONS, key="ecdf_trans")
            t, xl, yl, pal = _plot_customization_widgets("ecdf")
            generate = st.button("Generate", key="btn_ecdf", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            fig = plot_ecdf(df, x, _opt(color_col), transf, t, xl, yl, pal)
                            st.plotly_chart(fig, use_container_width=True)
                            _download_buttons(fig, f"ecdf_{x}")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )

        # ── Heatmap ───────────────────────────────────────────────────────────
        elif plot_type == "Heatmap":
            if len(cat_cols) < 2:
                st.warning(
                    "A heatmap needs at least two categorical columns. "
                    "Please check your dataset."
                )
                return
            x          = st.selectbox("Row variable", cat_cols, key="heat_x")
            y          = st.selectbox("Column variable", [c for c in cat_cols if c != x], key="heat_y")
            colorscale = st.selectbox("Color scale", SEQUENTIAL_SCALES, key="heat_scale")
            t, xl, yl, _ = _plot_customization_widgets("heat")
            generate = st.button("Generate", key="btn_heatmap", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            fig = plot_heatmap(df, x, y, t, xl, yl, colorscale)
                            st.plotly_chart(fig, use_container_width=True)
                            _download_buttons(fig, f"heatmap_{x}_{y}")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )

        # ── Correlation Matrix ────────────────────────────────────────────────
        elif plot_type == "Correlation Matrix":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns for a correlation matrix.")
                return
            selected_cols = st.multiselect(
                "Columns to include",
                num_cols,
                default=num_cols[:min(8, len(num_cols))],
                key="corr_cols",
            )
            colorscale = st.selectbox("Color scale", DIVERGING_SCALES, key="corr_scale")
            t, _, _, _ = _plot_customization_widgets("corr")
            generate = st.button("Generate", key="btn_corr", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Rendering…"):
                        try:
                            if len(selected_cols) < 2:
                                st.warning("Please select at least 2 columns.")
                            else:
                                fig = plot_correlation_matrix(df, selected_cols, t, colorscale)
                                st.plotly_chart(fig, use_container_width=True)
                                _download_buttons(fig, "correlation_matrix")
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

def page_home() -> None:
    st.markdown(
        """
        <div class="landing-hero">
            <div class="tagline">Scientific Data Analysis Platform</div>
            <h1>Daedalus</h1>
            <p class="subtitle">
                Upload your dataset and instantly unlock interactive visualizations,
                statistical summaries, and a fully customizable plot workspace —
                no code required.
            </p>
            <div class="get-started-badge">
                👈 Select a page from the <span class="accent">sidebar</span> to get started
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="card">
                <h4>📊 Data Explorer</h4>
                <p style="color:#888;font-size:0.9rem;line-height:1.65;">
                    Upload CSV or Excel files. Get instant summary statistics,
                    missing-value reports, and auto-generated overview charts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <h4>🎨 8 Interactive Plot Types</h4>
                <p style="color:#888;font-size:0.9rem;line-height:1.65;">
                    Scatter, Histogram, Box, Violin, Bar, ECDF, Heatmap, and
                    Correlation Matrix — all fully interactive and dark-themed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
                <h4>⬇ Export Ready</h4>
                <p style="color:#888;font-size:0.9rem;line-height:1.65;">
                    Every plot ships with one-click download as an interactive
                    HTML file or a high-resolution PNG image.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_data_explorer() -> None:
    st.markdown(
        """
        <div style="margin-bottom:24px;">
            <h2 style="margin-bottom:4px;">Data Explorer</h2>
            <p style="color:#888;font-size:0.95rem;">
                Upload your dataset to explore statistics and build interactive visualizations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── File upload ──────────────────────────────────────────────────────────
    _section("📂  Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="explorer_uploader",
        help=(
            "Accepted formats: .csv (comma-separated values) or .xlsx / .xls (Microsoft Excel). "
            "The first row must contain column names."
        ),
    )

    if uploaded_file is not None:
        if (
            "explorer_filename" not in st.session_state
            or st.session_state["explorer_filename"] != uploaded_file.name
        ):
            with st.spinner("Loading file…"):
                try:
                    df = _load_data(uploaded_file)
                    st.session_state["explorer_data"]     = df
                    st.session_state["explorer_filename"] = uploaded_file.name
                    st.success(
                        f"✅ **{uploaded_file.name}** loaded — "
                        f"{df.shape[0]:,} rows × {df.shape[1]:,} columns."
                    )
                except Exception as exc:
                    st.markdown(
                        f'<div class="error-box">⚠️ Could not read file: {_friendly_error(exc)}</div>',
                        unsafe_allow_html=True,
                    )
                    return

    if "explorer_data" not in st.session_state:
        st.markdown(
            """
            <div class="upload-placeholder">
                <h3>👆 Upload a file to get started</h3>
                <p>Once you upload a CSV or Excel file, a full data overview,
                summary statistics, and interactive charts will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df: pd.DataFrame = st.session_state["explorer_data"]

    # ── Dataset overview ─────────────────────────────────────────────────────
    _section("📊  Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",        f"{df.shape[0]:,}",           help="Total number of data rows.")
    c2.metric("Columns",     f"{df.shape[1]:,}",           help="Total number of columns.")
    c3.metric("Numeric",     len(_numeric_cols(df)),        help="Columns with numerical values.")
    c4.metric("Categorical", len(_categorical_cols(df)),    help="Columns with text or category values.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Interactive data table** — scroll, sort, and search your data below.")
    st.dataframe(df, use_container_width=True, height=300)

    missing_pct = df.isnull().mean().mul(100).round(2)
    if missing_pct.max() > 0:
        with st.expander("⚠️  Missing value summary", expanded=False):
            miss_df = (
                missing_pct[missing_pct > 0]
                .reset_index()
                .rename(columns={"index": "Column", 0: "Missing (%)"})
            )
            st.dataframe(miss_df, use_container_width=True)

    # ── Summary statistics ───────────────────────────────────────────────────
    _section("📋  Summary Statistics")
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    if num_cols:
        st.markdown("**Numerical columns**")
        try:
            summary_df = eda.num_summary(data=df, columns=num_cols)
            st.dataframe(summary_df.style.format(precision=4), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not compute numerical summary: {_friendly_error(exc)}")

    if cat_cols:
        st.markdown("**Categorical columns**")
        try:
            summary_df = eda.categorical_summary(data=df, columns=cat_cols)
            st.dataframe(summary_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not compute categorical summary: {_friendly_error(exc)}")

    if not num_cols and not cat_cols:
        st.info("No numeric or categorical columns detected in this dataset.")

    # ── Auto-generated overview plots (max 2) ────────────────────────────────
    _section("🖼️  Quick Preview Charts")
    st.markdown("Automatically generated overview of your first columns — up to 2 charts.")
    _overview_plots(df)

    # ── Custom plot builder ──────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    _section("🎨  Plot Workspace")
    st.markdown(
        "Build interactive, publication-ready charts. Configure the plot type, variables, "
        "and styling below, then click **Generate**."
    )
    _custom_plot_builder(df)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 14px 0 22px 0;">
                <div style="font-size:1.4rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">
                    🔬 Daedalus
                </div>
                <div style="font-size:0.72rem;color:#555;margin-top:3px;letter-spacing:0.08em;text-transform:uppercase;">
                    Scientific Analysis Platform
                </div>
            </div>
            <hr style="border-color:#2a2a2a;margin-bottom:20px;">
            """,
            unsafe_allow_html=True,
        )

        nav = st.radio(
            "Navigation",
            options=["🏠  Home", "📊  Data Explorer"],
            label_visibility="collapsed",
        )

    page_map = {
        "🏠  Home":          "home",
        "📊  Data Explorer": "data_explorer",
    }
    return page_map[nav]


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    page = _sidebar()

    if page == "home":
        page_home()
    elif page == "data_explorer":
        page_data_explorer()


if __name__ == "__main__":
    main()
