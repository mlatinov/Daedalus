"""
Daedalus — Scientific Data Analysis Platform
Tab 1: Data Explorer
"""

import io
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so analytics/ and functions/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics import eda  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daedalus — Scientific Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global Styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Typography ── */
    html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* ── Top banner ── */
    .app-banner {
        background: linear-gradient(135deg, #0f2044, #1d4ed8);
        color: white;
        padding: 18px 28px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .app-banner h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
    .app-banner p  { margin: 4px 0 0 0; font-size: 0.95rem; opacity: 0.85; }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        color: #64748b;
        padding: 8px 18px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #1d4ed8;
        background: #eff6ff;
        border-bottom: 3px solid #1d4ed8;
    }

    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(90deg, #0f2044, #1d4ed8);
        color: white;
        padding: 9px 16px;
        border-radius: 6px;
        margin: 20px 0 10px 0;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* ── Tab description box ── */
    .tab-description {
        background: #f0f7ff;
        border-left: 4px solid #1d4ed8;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 22px;
        color: #1e3a5f;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* ── Upload placeholder ── */
    .upload-placeholder {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 48px 24px;
        text-align: center;
        color: #64748b;
        margin-top: 16px;
    }
    .upload-placeholder h3 { color: #475569; margin-bottom: 8px; }

    /* ── User-friendly error box ── */
    .error-box {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 8px;
        padding: 12px 18px;
        color: #b91c1c;
        font-size: 0.9rem;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* ── TODO / placeholder box ── */
    .todo-box {
        background: #fffbeb;
        border: 1px solid #fbbf24;
        border-radius: 8px;
        padding: 14px 18px;
        color: #92400e;
        font-size: 0.9rem;
    }

    /* ── Subtle divider ── */
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMATIONS = ["None", "log", "boxcox", "yeojohnson", "sqrt"]
BAR_STATS       = ["percent", "count", "probability", "proportion", "density"]
_NONE_SENTINEL  = "— none —"


def _render_figure(fig_or_tuple) -> None:
    """
    Render a figure returned by an analytics function.

    Handles:
      • (matplotlib.Figure, Axes) tuple  → st.pyplot(fig)
      • seaborn grid (PairGrid / JointGrid) → st.pyplot(grid.fig)
      • bare matplotlib.Figure            → st.pyplot(fig)
    """
    if isinstance(fig_or_tuple, tuple):
        fig = fig_or_tuple[0]
    else:
        fig = fig_or_tuple

    # Seaborn grid objects expose the underlying Figure via .fig
    mpl_fig = getattr(fig, "fig", fig)
    st.pyplot(mpl_fig)
    plt.close("all")


def _friendly_error(exc: Exception) -> str:
    """Convert a raw exception into a plain-language message."""
    msg = str(exc)
    # Map common pandas/numpy/seaborn messages to friendlier text
    if "could not convert string to float" in msg or "could not be interpreted" in msg:
        return "One of the selected columns contains text values. Please choose a numeric column."
    if "all values must be positive" in msg or "strictly positive" in msg:
        return (
            "The selected transformation requires all values to be positive. "
            "Try 'yeojohnson' instead, or remove zero / negative values first."
        )
    if "not enough values to unpack" in msg or "only integers" in msg:
        return "Not enough data to draw this plot. Check that the column has sufficient values."
    if "hue" in msg.lower() and "palette" in msg.lower():
        return "Could not apply the selected color grouping. Make sure the color column is categorical."
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


def _numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _all_cols(df: pd.DataFrame):
    return df.columns.tolist()


def _opt(value: str):
    """Return None when the sentinel 'none' option is selected."""
    return None if value == _NONE_SENTINEL else value


def _transformation_arg(value: str):
    """Return None when no transformation is selected."""
    return None if value == "None" else value


# ─────────────────────────────────────────────────────────────────────────────
# Section header helper
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Data Explorer
# ─────────────────────────────────────────────────────────────────────────────

def _overview_plots(df: pd.DataFrame) -> None:
    """
    Auto-generate a quick visual overview when data is first uploaded.

    Shows histograms for numeric columns and bar charts for categorical columns,
    up to 8 plots total, arranged in a two-column grid.
    """
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    # Limit overview to avoid overwhelming the page
    selected_num = num_cols[:6]
    selected_cat = cat_cols[:4]
    all_selected  = [(c, "num") for c in selected_num] + [(c, "cat") for c in selected_cat]
    all_selected   = all_selected[:8]

    if not all_selected:
        st.info("No columns available to preview.")
        return

    grid_cols = st.columns(2)
    for idx, (col_name, kind) in enumerate(all_selected):
        with grid_cols[idx % 2]:
            try:
                if kind == "num":
                    result = eda.viz_histogram(data=df, x=col_name)
                else:
                    result = eda.viz_bar(data=df, x=col_name)
                _render_figure(result)
            except Exception as exc:
                st.warning(f"Could not generate overview plot for **{col_name}**: {_friendly_error(exc)}")
                plt.close("all")


def _summary_section(df: pd.DataFrame) -> None:
    """Display numerical and categorical summary statistics."""
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


def _custom_plot_builder(df: pd.DataFrame) -> None:
    """Interactive widget that lets the user choose a plot type and configure it."""
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    all_cols = _all_cols(df)

    PLOT_TYPES = [
        "Histogram",
        "Boxplot",
        "Violin",
        "Scatter",
        "Bar Chart",
        "ECDF",
        "Heatmap (Cross-tabulation)",
        "Correlation Matrix",
    ]

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("##### Configure your plot")
        plot_type = st.selectbox(
            "Plot type",
            PLOT_TYPES,
            help="Choose the kind of chart you want to create.",
        )

        # ── Histogram ────────────────────────────────────────────────────────
        if plot_type == "Histogram":
            if not num_cols:
                st.warning("No numeric columns found in the dataset.")
                return
            x = st.selectbox(
                "Variable (X axis)",
                num_cols,
                help="The numeric column whose distribution you want to visualise.",
            )
            transformation = st.selectbox(
                "Transformation",
                TRANSFORMATIONS,
                help=(
                    "Optional: apply a mathematical transformation to make skewed data "
                    "easier to visualise. 'log' is most common for right-skewed data."
                ),
            )
            color = st.selectbox(
                "Split by group (optional)",
                [_NONE_SENTINEL] + cat_cols,
                help="Colour-code bars by a categorical variable to compare distributions across groups.",
            )
            generate = st.button("Generate Plot", key="btn_histogram", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Drawing plot…"):
                        try:
                            result = eda.viz_histogram(
                                data=df,
                                x=x,
                                transformation=_transformation_arg(transformation),
                                color=_opt(color),
                            )
                            _render_figure(result)
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )
                            plt.close("all")

        # ── Boxplot ──────────────────────────────────────────────────────────
        elif plot_type == "Boxplot":
            if not num_cols:
                st.warning("No numeric columns found in the dataset.")
                return
            x = st.selectbox(
                "Numeric variable (X axis)",
                num_cols,
                help="The numeric column to show as a box-and-whisker plot.",
            )
            y = st.selectbox(
                "Group by (optional)",
                [_NONE_SENTINEL] + cat_cols,
                help="Optional categorical column to split the boxplot into groups.",
            )
            transformation = st.selectbox(
                "Transformation",
                TRANSFORMATIONS,
                help="Apply an optional transformation to the numeric column.",
            )
            color = st.selectbox(
                "Colour group (optional)",
                [_NONE_SENTINEL] + cat_cols,
                help="Additional categorical column to colour-code the boxes.",
            )
            generate = st.button("Generate Plot", key="btn_boxplot", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Drawing plot…"):
                        try:
                            result = eda.viz_boxplot(
                                data=df,
                                x=x,
                                y=_opt(y),
                                transformation=_transformation_arg(transformation),
                                color=_opt(color),
                            )
                            _render_figure(result)
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )
                            plt.close("all")

        # ── Violin ───────────────────────────────────────────────────────────
        elif plot_type == "Violin":
            if not num_cols:
                st.warning("No numeric columns found in the dataset.")
                return
            if not cat_cols:
                st.warning(
                    "A violin plot needs at least one categorical column to define the groups. "
                    "Please make sure your dataset includes a text/category column."
                )
                return
            x = st.selectbox(
                "Numeric variable",
                num_cols,
                help="The numeric column whose distribution you want to compare across groups.",
            )
            y = st.selectbox(
                "Grouping variable",
                cat_cols,
                help="Categorical column that defines the groups (violins).",
            )
            transformation = st.selectbox(
                "Transformation",
                TRANSFORMATIONS,
                help="Apply an optional transformation to the numeric column.",
            )
            color = st.selectbox(
                "Colour group (optional)",
                [_NONE_SENTINEL] + cat_cols,
                help="Additional categorical column for colour coding.",
            )
            generate = st.button("Generate Plot", key="btn_violin", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Drawing plot…"):
                        try:
                            result = eda.viz_violin(
                                data=df,
                                x=x,
                                y=y,
                                transformation=_transformation_arg(transformation),
                                color=_opt(color),
                            )
                            _render_figure(result)
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )
                            plt.close("all")

        # ── Scatter ──────────────────────────────────────────────────────────
        elif plot_type == "Scatter":
            if len(num_cols) < 2:
                st.warning("A scatter plot needs at least two numeric columns.")
                return
            x = st.selectbox(
                "X axis",
                num_cols,
                help="Numeric column for the horizontal axis.",
            )
            y_opts = [c for c in num_cols if c != x]
            y = st.selectbox(
                "Y axis",
                y_opts,
                help="Numeric column for the vertical axis.",
            )
            transformation = st.selectbox(
                "Transformation (applied to X)",
                TRANSFORMATIONS,
                help="Optional transformation applied to the X-axis variable.",
            )
            color = st.selectbox(
                "Colour group (optional)",
                [_NONE_SENTINEL] + cat_cols,
                help="Colour-code points by a categorical variable.",
            )
            generate = st.button("Generate Plot", key="btn_scatter", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Drawing plot…"):
                        try:
                            result = eda.viz_scatter(
                                data=df,
                                x=x,
                                y=y,
                                color=_opt(color),
                                transformation=_transformation_arg(transformation),
                            )
                            _render_figure(result)
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )
                            plt.close("all")

        # ── Bar Chart ────────────────────────────────────────────────────────
        elif plot_type == "Bar Chart":
            if not cat_cols:
                st.warning("No categorical columns found in the dataset.")
                return
            x = st.selectbox(
                "Categorical variable",
                cat_cols,
                help="The column whose category frequencies you want to display.",
            )
            stat = st.selectbox(
                "Bar height shows",
                BAR_STATS,
                help=(
                    "'percent' — percentage of total; 'count' — raw number of rows; "
                    "'probability' / 'proportion' — fraction of total; 'density' — density estimate."
                ),
            )
            color = st.selectbox(
                "Colour group (optional)",
                [_NONE_SENTINEL] + cat_cols,
                help="Split bars by an additional categorical variable.",
            )
            generate = st.button("Generate Plot", key="btn_bar", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Drawing plot…"):
                        try:
                            result = eda.viz_bar(
                                data=df,
                                x=x,
                                color=_opt(color),
                                stat=stat,
                            )
                            _render_figure(result)
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )
                            plt.close("all")

        # ── ECDF ─────────────────────────────────────────────────────────────
        elif plot_type == "ECDF":
            if not num_cols:
                st.warning("No numeric columns found in the dataset.")
                return
            x = st.selectbox(
                "Variable",
                num_cols,
                help=(
                    "Numeric column to plot. The curve shows what fraction of values are "
                    "less than or equal to each point on the X axis."
                ),
            )
            transformation = st.selectbox(
                "Transformation",
                TRANSFORMATIONS,
                help="Optional transformation applied before plotting.",
            )
            color = st.selectbox(
                "Colour group (optional)",
                [_NONE_SENTINEL] + cat_cols,
                help="Draw a separate ECDF curve for each group.",
            )
            generate = st.button("Generate Plot", key="btn_ecdf", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Drawing plot…"):
                        try:
                            result = eda.viz_ECDF(
                                data=df,
                                x=x,
                                transformation=_transformation_arg(transformation),
                                color=_opt(color),
                            )
                            _render_figure(result)
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )
                            plt.close("all")

        # ── Heatmap (Cross-tabulation) ────────────────────────────────────────
        elif plot_type == "Heatmap (Cross-tabulation)":
            if len(cat_cols) < 2:
                st.warning(
                    "A cross-tabulation heatmap needs at least two categorical columns. "
                    "Please check your data."
                )
                return
            x = st.selectbox(
                "Row variable",
                cat_cols,
                help="Categorical column to display as rows of the heatmap.",
            )
            y_opts = [c for c in cat_cols if c != x]
            y = st.selectbox(
                "Column variable",
                y_opts,
                help="Categorical column to display as columns of the heatmap.",
            )
            generate = st.button("Generate Plot", key="btn_heatmap", type="primary")
            if generate:
                with col_right:
                    with st.spinner("Drawing plot…"):
                        try:
                            result = eda.viz_cross_tab(data=df, x=x, y=y)
                            _render_figure(result)
                        except Exception as exc:
                            st.markdown(
                                f'<div class="error-box">⚠️ {_friendly_error(exc)}</div>',
                                unsafe_allow_html=True,
                            )
                            plt.close("all")

        # ── Correlation Matrix ────────────────────────────────────────────────
        elif plot_type == "Correlation Matrix":
            with col_right:
                # TODO: analytics/eda.py does not yet expose a correlation matrix function.
                # Action required (analytics team): add viz_correlation(data, columns) to
                # analytics/eda.py, then replace the placeholder below with a call to it.
                st.markdown(
                    """
                    <div class="todo-box">
                    🔧 <strong>Coming soon</strong> — The Correlation Matrix plot is not yet
                    available. A <code>viz_correlation()</code> function needs to be added to
                    <code>analytics/eda.py</code> before this chart can be displayed.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def tab_data_explorer() -> None:
    """Render the complete Data Explorer tab."""

    # ── Tab description ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="tab-description">
        <strong>Data Explorer</strong> — Upload your dataset (CSV or Excel) to instantly see a
        summary of its contents, key statistics, and automatically generated charts. You can also
        build custom plots by choosing a chart type and the variables you want to explore.
        No coding required.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── File upload ──────────────────────────────────────────────────────────
    _section("📂  Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="tab1_uploader",
        help=(
            "Accepted formats: .csv (comma-separated values) or .xlsx / .xls (Microsoft Excel). "
            "The first row must contain column names."
        ),
    )

    # ── Load into session state so the data persists across reruns ───────────
    if uploaded_file is not None:
        if (
            "tab1_filename" not in st.session_state
            or st.session_state["tab1_filename"] != uploaded_file.name
        ):
            with st.spinner("Loading file…"):
                try:
                    df = _load_data(uploaded_file)
                    st.session_state["tab1_data"]     = df
                    st.session_state["tab1_filename"] = uploaded_file.name
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

    # ── Guard: nothing uploaded yet ──────────────────────────────────────────
    if "tab1_data" not in st.session_state:
        st.markdown(
            """
            <div class="upload-placeholder">
                <h3>👆  Upload a file to get started</h3>
                <p>Once you upload a CSV or Excel file, a full data overview,
                summary statistics, and interactive charts will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df: pd.DataFrame = st.session_state["tab1_data"]

    # ── Dataset overview ─────────────────────────────────────────────────────
    _section("📊  Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}", help="Total number of data rows.")
    c2.metric("Columns", f"{df.shape[1]:,}", help="Total number of columns (variables).")
    c3.metric(
        "Numeric columns",
        len(_numeric_cols(df)),
        help="Columns with numerical (continuous or integer) values.",
    )
    c4.metric(
        "Categorical columns",
        len(_categorical_cols(df)),
        help="Columns with text or category values.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Interactive data table** — scroll, sort, and search your data below.")
    st.dataframe(df, use_container_width=True, height=320)

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
    _summary_section(df)

    # ── Auto-generated overview plots ────────────────────────────────────────
    _section("🖼️  Automatic Overview Charts")
    st.markdown(
        "A quick visual overview of all columns in your dataset. "
        "Numeric columns are shown as histograms; categorical columns as bar charts."
    )
    _overview_plots(df)

    # ── Custom plot builder ──────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    _section("🎨  Custom Plot Builder")
    st.markdown(
        "Build your own chart by selecting a plot type and the columns you want to explore. "
        "All settings include plain-language descriptions to guide you."
    )
    _custom_plot_builder(df)


# ─────────────────────────────────────────────────────────────────────────────
# App Layout
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Top banner
    st.markdown(
        """
        <div class="app-banner">
            <h1>🔬 Daedalus</h1>
            <p>Scientific Data Analysis Platform for Molecular Biology Research</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    (tab1,) = st.tabs(["📊  Data Explorer"])

    with tab1:
        tab_data_explorer()


if __name__ == "__main__":
    main()
