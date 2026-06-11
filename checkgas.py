"""
Biogas Analyzer — Multi-File Log Dashboard
Supports any number of semicolon-delimited log CSVs from your analyzer instrument.

Column layout (14 fields, semicolon-delimited):
  0  timestamp          DD-MM-YYYY HH:MM:SS
  1  CH4 [%]
  2  CO2 [%]
  3  O2 [%]
  4  H2S High [ppm]
  5  H2S Low [ppm]
  6  Dew Point [°C]
  7  Flow [l/min]
  8  Balance [%]
  9  GCV [Kcal/SCM]
  10 NCV [Kcal/SCM]
  11 R.Density [kg/SCM]  — always numeric
  12 Point label         — "Point 1" / "Point 2" / "Point 3" or empty
  13 Alarm text          — free-text alarm string or empty
"""

import io
import math
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Biogas Log Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette (one per point + neutral)
POINT_COLOURS = {
    "Point 1": "#3B82F6",   # blue  — before compressor
    "Point 2": "#10B981",   # green — sweet gas
    "Point 3": "#F59E0B",   # amber — after purification
    "No Point": "#94A3B8",  # slate — untagged
}

# All numeric parameters we want to plot, grouped logically
PARAM_GROUPS = {
    "Gas Composition (%)": ["CH4 [%]", "CO2 [%]", "O2 [%]", "Balance [%]"],
    "H₂S (ppm)": ["H2S High [ppm]", "H2S Low [ppm]"],
    "Energy Values (Kcal/SCM)": ["GCV [Kcal/SCM]", "NCV [Kcal/SCM]"],
    "Physical / Flow": [
        "Flow [l/min]",
        "Dew Point [°C]",
        "R.Density [kg/SCM]",
    ],
    "Derived Parameters": [
        "CH4:CO2 Ratio",
        "H2S Total [ppm]",
        "Methane Quality Index",
        "Cumulative Flow [l]",
    ],
}

# CH4 thresholds for average calculations (applied per point)
CH4_AVG_THRESHOLDS = {
    "Point 1": {"min": 80.0},   # only rows with CH4 >= 80% count toward averages
    "Point 2": {"max": 65.0},   # only rows with CH4 <= 65% count toward averages
    "Point 3": {"min": 80.0},   # only rows with CH4 >= 80% count toward averages
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

RAW_COLS = [
    "timestamp",
    "CH4 [%]",
    "CO2 [%]",
    "O2 [%]",
    "H2S High [ppm]",
    "H2S Low [ppm]",
    "Dew Point [°C]",
    "Flow [l/min]",
    "Balance [%]",
    "GCV [Kcal/SCM]",
    "NCV [Kcal/SCM]",
    "R.Density [kg/SCM]",
    "Point",
    "Alarm",
]


@st.cache_data(show_spinner=False)
def load_file(uploaded_file) -> pd.DataFrame:
    """
    Parse a single log CSV (semicolon-delimited, 14 fields).
    Returns a clean DataFrame with derived columns added.
    """
    content = uploaded_file.read()
    # Try to detect encoding
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    df = pd.read_csv(
        io.StringIO(text),
        sep=";",
        header=None,
        skiprows=1,      # skip the original header row
        names=RAW_COLS,
        on_bad_lines="skip",
    )

    # ── Timestamp ────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Numeric coercion ─────────────────────────────────────
    numeric_cols = [c for c in RAW_COLS if c not in ("timestamp", "Point", "Alarm")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Point label normalisation ─────────────────────────────
    df["Point"] = df["Point"].astype(str).str.strip().replace("nan", pd.NA)
    df["Point"] = df["Point"].where(
        df["Point"].isin(["Point 1", "Point 2", "Point 3"]), other=pd.NA
    )

    # ── Alarm normalisation ───────────────────────────────────
    df["Alarm"] = df["Alarm"].astype(str).str.strip().replace("nan", pd.NA)
    df["Has Alarm"] = df["Alarm"].notna() & (df["Alarm"] != "")

    # ── Derived parameters ────────────────────────────────────
    # CH4:CO2 ratio — useful purity indicator
    df["CH4:CO2 Ratio"] = (
        df["CH4 [%]"] / df["CO2 [%]"].replace(0, float("nan"))
    ).round(3)

    # Combined H2S (High sensor includes Low readings; pick max)
    df["H2S Total [ppm]"] = df[["H2S High [ppm]", "H2S Low [ppm]"]].max(axis=1)

    # Methane Quality Index: normalised 0–100
    # High CH4, low CO2, low O2, low H2S = good quality
    # MQI = CH4% * (1 - O2/21) * (1 - H2S_total/2000)
    df["Methane Quality Index"] = (
        df["CH4 [%]"].clip(0, 100)
        * (1 - (df["O2 [%]"].clip(0, 21) / 21))
        * (1 - (df["H2S Total [ppm]"].clip(0, 2000) / 2000))
    ).round(2)

    # Cumulative flow (trapezoidal integration over time in minutes)
    dt_min = df["timestamp"].diff().dt.total_seconds().fillna(0) / 60.0
    df["Cumulative Flow [l]"] = (df["Flow [l/min]"].fillna(0) * dt_min).cumsum().round(2)

    # ── Source metadata ───────────────────────────────────────
    df["source_file"] = uploaded_file.name
    df["date"] = df["timestamp"].dt.date

    return df


def compute_working_time(df: pd.DataFrame) -> timedelta | None:
    """Working time = last − first timestamp where a Point label is present."""
    point_rows = df[df["Point"].notna()]
    if len(point_rows) < 2:
        return None
    return point_rows["timestamp"].max() - point_rows["timestamp"].min()


def apply_ch4_threshold(df: pd.DataFrame, point: str | None) -> pd.DataFrame:
    """
    Filter rows to those meeting the CH4 threshold for average calculations.
    Only applied when a specific point is given and that point has a defined threshold.
    Returns the dataframe unchanged when point is None or has no threshold.
    """
    if point is None or point not in CH4_AVG_THRESHOLDS:
        return df
    thresh = CH4_AVG_THRESHOLDS[point]
    mask = pd.Series([True] * len(df), index=df.index)
    if "min" in thresh:
        mask &= df["CH4 [%]"].ge(thresh["min"]) | df["CH4 [%]"].isna()
    if "max" in thresh:
        mask &= df["CH4 [%]"].le(thresh["max"]) | df["CH4 [%]"].isna()
    return df[mask]



    """Return a dict of key stats for a given point (or all data if None)."""
    sub = df if point_filter is None else df[df["Point"] == point_filter]
    if sub.empty:
        return {}
    stats = {}
    for col in [
        "CH4 [%]", "CO2 [%]", "O2 [%]",
        "H2S High [ppm]", "H2S Low [ppm]",
        "Dew Point [°C]", "Flow [l/min]",
        "GCV [Kcal/SCM]", "NCV [Kcal/SCM]",
        "R.Density [kg/SCM]", "Methane Quality Index",
    ]:
        vals = sub[col].dropna()
        if len(vals):
            stats[col] = {
                "mean": vals.mean(),
                "min": vals.min(),
                "max": vals.max(),
                "std": vals.std(),
                "count": len(vals),
            }
    return stats


# ─────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────

def time_series_chart(
    df: pd.DataFrame,
    params: list[str],
    title: str,
    colour_by_point: bool = True,
) -> go.Figure:
    """Multi-line time-series chart, optionally colour-coded by Point label."""
    fig = go.Figure()
    point_labels = df["Point"].fillna("No Point").unique()

    for param in params:
        for pt in sorted(point_labels):
            sub = df[df["Point"].fillna("No Point") == pt] if colour_by_point else df
            sub = sub.dropna(subset=[param])
            if sub.empty:
                continue
            colour = POINT_COLOURS.get(pt, "#94A3B8") if colour_by_point else None
            label = f"{param} — {pt}" if colour_by_point and len(point_labels) > 1 else param
            fig.add_trace(
                go.Scatter(
                    x=sub["timestamp"],
                    y=sub[param],
                    mode="lines",
                    name=label,
                    line=dict(color=colour, width=1.5),
                    hovertemplate=(
                        f"<b>{param}</b><br>"
                        "%{x|%H:%M:%S}<br>"
                        "Value: %{y:.3f}<extra></extra>"
                    ),
                )
            )
            if not colour_by_point:
                break  # only one pass if not splitting by point

    _style_fig(fig, title)
    return fig


def bar_comparison_chart(
    dfs: list[pd.DataFrame],
    param: str,
    file_names: list[str],
) -> go.Figure:
    """Bar chart comparing mean values of one parameter across multiple files."""
    rows = []
    for df, name in zip(dfs, file_names):
        for pt in ["Point 1", "Point 2", "Point 3"]:
            sub = df[df["Point"] == pt][param].dropna()
            if not sub.empty:
                rows.append(
                    dict(
                        file=name,
                        point=pt,
                        mean=sub.mean(),
                        std=sub.std(),
                    )
                )
    if not rows:
        return go.Figure()

    bar_df = pd.DataFrame(rows)
    fig = px.bar(
        bar_df,
        x="file",
        y="mean",
        color="point",
        barmode="group",
        error_y="std",
        color_discrete_map=POINT_COLOURS,
        title=f"{param} — Mean by File & Point",
        labels={"mean": param, "file": "Log File"},
    )
    _style_fig(fig, f"{param} — Mean by File & Point")
    return fig


def alarm_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of alarm distribution."""
    alarm_df = df[df["Has Alarm"]]["Alarm"].value_counts().reset_index()
    alarm_df.columns = ["Alarm", "Count"]
    if alarm_df.empty:
        return go.Figure()
    fig = px.pie(
        alarm_df,
        names="Alarm",
        values="Count",
        title="Alarm Distribution",
        hole=0.4,
    )
    _style_fig(fig, "Alarm Distribution")
    return fig


def density_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: Density vs CH4% coloured by Point."""
    sub = df[df["Point"].notna() & df["R.Density [kg/SCM]"].notna() & df["CH4 [%]"].notna()]
    if sub.empty:
        return go.Figure()
    fig = px.scatter(
        sub,
        x="CH4 [%]",
        y="R.Density [kg/SCM]",
        color="Point",
        color_discrete_map=POINT_COLOURS,
        title="R.Density vs CH4% (by Point)",
        trendline="ols",
        labels={"R.Density [kg/SCM]": "Density (kg/SCM)", "CH4 [%]": "CH4 (%)"},
        hover_data=["timestamp", "CO2 [%]", "GCV [Kcal/SCM]"],
    )
    _style_fig(fig, "R.Density vs CH4% (by Point)")
    return fig


def _style_fig(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=60, b=50),
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        hovermode="x unified",
    )


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def metric_row(label: str, val: float | str, unit: str = "", delta: float | None = None):
    """Render a single metric tile."""
    if isinstance(val, float):
        formatted = f"{val:.3g}"
    else:
        formatted = str(val)
    st.metric(label=f"{label} {unit}", value=formatted, delta=f"{delta:.3g}" if delta else None)


def render_stats_table(stats: dict):
    rows = []
    for param, s in stats.items():
        rows.append(
            {
                "Parameter": param,
                "Mean": f"{s['mean']:.4g}",
                "Min": f"{s['min']:.4g}",
                "Max": f"{s['max']:.4g}",
                "Std Dev": f"{s['std']:.4g}",
                "N": s["count"],
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔥 Biogas Analyzer")
    st.caption("Upload one or more log files")

    uploaded_files = st.file_uploader(
        "Log CSV files",
        accept_multiple_files=True,
        type=["csv"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Display options")
    colour_by_point = st.toggle("Colour traces by Point", value=True)
    show_alarms = st.toggle("Highlight alarm rows", value=True)
    resample_freq = st.selectbox(
        "Resample / smooth",
        ["None (raw)", "1 min", "5 min", "15 min"],
        index=0,
    )
    st.divider()
    st.caption(
        "Point 1 = Before Compressor\n"
        "Point 2 = Sweet Gas\n"
        "Point 3 = After Purification"
    )

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if not uploaded_files:
    st.title("Biogas Log Analyzer")
    st.info(
        "👈 Upload one or more analyzer log CSV files from the sidebar to begin.\n\n"
        "Each file is a semicolon-delimited daily log with columns:\n"
        "`timestamp · CH4 · CO2 · O2 · H2S_High · H2S_Low · DewPoint · Flow · "
        "Balance · GCV · NCV · Density · Point · Alarm`"
    )
    st.stop()

# ── Load all files ─────────────────────────────────────────
with st.spinner("Loading files…"):
    all_dfs = []
    file_names = []
    load_errors = []

    for uf in uploaded_files:
        try:
            df_loaded = load_file(uf)
            all_dfs.append(df_loaded)
            file_names.append(uf.name)
        except Exception as exc:
            load_errors.append(f"{uf.name}: {exc}")

if load_errors:
    for err in load_errors:
        st.error(f"Failed to load: {err}")

if not all_dfs:
    st.warning("No files loaded successfully.")
    st.stop()

# ── Combine ────────────────────────────────────────────────
combined = pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")

# ── Optional resampling ────────────────────────────────────
RESAMPLE_MAP = {"1 min": "1min", "5 min": "5min", "15 min": "15min"}
if resample_freq != "None (raw)":
    freq = RESAMPLE_MAP[resample_freq]
    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
    combined_resampled = (
        combined.set_index("timestamp")[numeric_cols]
        .resample(freq)
        .mean()
        .reset_index()
    )
    # Re-attach categorical columns via nearest merge
    cat_cols = ["Point", "Alarm", "Has Alarm", "source_file", "date"]
    for c in cat_cols:
        if c in combined.columns:
            combined_resampled[c] = pd.NA
    combined_display = combined_resampled
else:
    combined_display = combined.copy()

# ─────────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────────

tab_overview, tab_gas, tab_energy, tab_flow, tab_derived, tab_points, tab_alarms, tab_raw = st.tabs(
    [
        "📊 Overview",
        "🧪 Gas Composition",
        "⚡ Energy",
        "💧 Flow & Physical",
        "📐 Derived",
        "🔵 Point Compare",
        "🚨 Alarms",
        "📋 Raw Data",
    ]
)


# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Overview")

    # ── Working time per file ───────────────────────────────
    st.subheader("Working Time Summary")
    wt_cols = st.columns(max(len(all_dfs), 1))
    for i, (df_i, fname) in enumerate(zip(all_dfs, file_names)):
        wt = compute_working_time(df_i)
        with wt_cols[i % len(wt_cols)]:
            if wt:
                h, rem = divmod(int(wt.total_seconds()), 3600)
                m, s = divmod(rem, 60)
                st.metric(
                    label=fname,
                    value=f"{h:02d}h {m:02d}m {s:02d}s",
                    help="Time span from first to last tagged measurement (Point 1/2/3)",
                )
            else:
                st.metric(label=fname, value="No Point data")

    st.divider()

    # ── Key averages across all data ───────────────────────
    st.subheader("Fleet Average (all files, all points)")
    kpi_params = [
        ("CH4 [%]", "CH4", "%"),
        ("CO2 [%]", "CO2", "%"),
        ("O2 [%]", "O2", "%"),
        ("H2S Total [ppm]", "H2S Total", "ppm"),
        ("GCV [Kcal/SCM]", "GCV", "kcal/SCM"),
        ("NCV [Kcal/SCM]", "NCV", "kcal/SCM"),
        ("R.Density [kg/SCM]", "Density", "kg/SCM"),
        ("Flow [l/min]", "Flow", "l/min"),
        ("Methane Quality Index", "Quality Index", ""),
    ]
    kpi_row = st.columns(len(kpi_params))
    for col, (param, label, unit) in zip(kpi_row, kpi_params):
        vals = combined[param].dropna()
        with col:
            if not vals.empty:
                st.metric(f"{label} ({unit})" if unit else label, f"{vals.mean():.3g}")
            else:
                st.metric(label, "—")

    st.divider()

    # ── CH4 overview chart ──────────────────────────────────
    st.subheader("CH4 — Full Timeline")
    fig_ch4 = time_series_chart(
        combined_display, ["CH4 [%]"], "CH4 % over Time", colour_by_point=colour_by_point
    )
    st.plotly_chart(fig_ch4, use_container_width=True)

    # ── Files summary table ─────────────────────────────────
    st.subheader("Files Loaded")
    summary_rows = []
    for df_i, fname in zip(all_dfs, file_names):
        wt = compute_working_time(df_i)
        wt_str = (
            str(wt).split(".")[0] if wt else "—"
        )
        summary_rows.append(
            {
                "File": fname,
                "Rows": len(df_i),
                "Start": df_i["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
                "End": df_i["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
                "Working Time": wt_str,
                "Point 1 rows": (df_i["Point"] == "Point 1").sum(),
                "Point 2 rows": (df_i["Point"] == "Point 2").sum(),
                "Point 3 rows": (df_i["Point"] == "Point 3").sum(),
                "Alarm rows": df_i["Has Alarm"].sum(),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — GAS COMPOSITION
# ═══════════════════════════════════════════════════════════════
with tab_gas:
    st.header("Gas Composition")

    params_gas = ["CH4 [%]", "CO2 [%]", "O2 [%]", "Balance [%]"]
    for param in params_gas:
        fig = time_series_chart(
            combined_display, [param], f"{param} over Time", colour_by_point=colour_by_point
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("H₂S")
    fig_h2s = time_series_chart(
        combined_display,
        ["H2S High [ppm]", "H2S Low [ppm]"],
        "H₂S (High & Low sensors)",
        colour_by_point=False,
    )
    st.plotly_chart(fig_h2s, use_container_width=True)

    st.subheader("Gas Composition Statistics")
    render_stats_table(summary_stats(combined, None))


# ═══════════════════════════════════════════════════════════════
# TAB 3 — ENERGY
# ═══════════════════════════════════════════════════════════════
with tab_energy:
    st.header("Energy Values")

    fig_gcv = time_series_chart(
        combined_display, ["GCV [Kcal/SCM]"], "GCV over Time", colour_by_point=colour_by_point
    )
    st.plotly_chart(fig_gcv, use_container_width=True)

    fig_ncv = time_series_chart(
        combined_display, ["NCV [Kcal/SCM]"], "NCV over Time", colour_by_point=colour_by_point
    )
    st.plotly_chart(fig_ncv, use_container_width=True)

    # GCV vs NCV overlay
    fig_both = go.Figure()
    for param, colour in [("GCV [Kcal/SCM]", "#6366F1"), ("NCV [Kcal/SCM]", "#EC4899")]:
        sub = combined_display.dropna(subset=[param])
        fig_both.add_trace(
            go.Scatter(
                x=sub["timestamp"], y=sub[param], mode="lines", name=param,
                line=dict(color=colour, width=1.5),
            )
        )
    _style_fig(fig_both, "GCV vs NCV")
    st.plotly_chart(fig_both, use_container_width=True)

    # Bar chart: mean GCV per file
    if len(all_dfs) > 1:
        st.subheader("GCV Mean by File (Point comparison)")
        st.plotly_chart(
            bar_comparison_chart(all_dfs, "GCV [Kcal/SCM]", file_names),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — FLOW & PHYSICAL
# ═══════════════════════════════════════════════════════════════
with tab_flow:
    st.header("Flow & Physical Parameters")

    fig_flow = time_series_chart(
        combined_display, ["Flow [l/min]"], "Flow Rate (l/min)", colour_by_point=colour_by_point
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    fig_dew = time_series_chart(
        combined_display, ["Dew Point [°C]"], "Dew Point (°C)", colour_by_point=colour_by_point
    )
    st.plotly_chart(fig_dew, use_container_width=True)

    fig_dens = time_series_chart(
        combined_display, ["R.Density [kg/SCM]"], "Relative Density (kg/SCM)",
        colour_by_point=colour_by_point,
    )
    st.plotly_chart(fig_dens, use_container_width=True)

    # Density scatter
    st.subheader("Density vs CH4% Scatter")
    st.plotly_chart(density_scatter(combined), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — DERIVED PARAMETERS
# ═══════════════════════════════════════════════════════════════
with tab_derived:
    st.header("Derived Parameters")

    st.caption(
        "**CH4:CO2 Ratio** — higher = purer methane\n\n"
        "**H2S Total** — max of high/low sensor readings\n\n"
        "**Methane Quality Index (MQI)** — composite 0–100: "
        "MQI = CH4% × (1 − O2/21) × (1 − H2S/2000)\n\n"
        "**Cumulative Flow** — running total gas volume per file"
    )

    fig_ratio = time_series_chart(
        combined_display, ["CH4:CO2 Ratio"], "CH4:CO2 Ratio", colour_by_point=colour_by_point
    )
    st.plotly_chart(fig_ratio, use_container_width=True)

    fig_mqi = time_series_chart(
        combined_display, ["Methane Quality Index"], "Methane Quality Index (0–100)",
        colour_by_point=colour_by_point,
    )
    st.plotly_chart(fig_mqi, use_container_width=True)

    # Cumulative flow per file
    st.subheader("Cumulative Flow by File")
    fig_cf = go.Figure()
    colours = px.colors.qualitative.Set2
    for i, (df_i, fname) in enumerate(zip(all_dfs, file_names)):
        sub = df_i.dropna(subset=["Cumulative Flow [l]"])
        fig_cf.add_trace(
            go.Scatter(
                x=sub["timestamp"], y=sub["Cumulative Flow [l]"],
                mode="lines", name=fname,
                line=dict(color=colours[i % len(colours)], width=2),
            )
        )
    _style_fig(fig_cf, "Cumulative Flow per File")
    st.plotly_chart(fig_cf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 6 — POINT COMPARISON
# ═══════════════════════════════════════════════════════════════
with tab_points:
    st.header("Point-by-Point Comparison")
    st.caption(
        "Point 1 = Before Compressor · "
        "Point 2 = Sweet Gas · "
        "Point 3 = After Purification"
    )

    available_points = [
        p for p in ["Point 1", "Point 2", "Point 3"]
        if (combined["Point"] == p).sum() > 0
    ]

    if not available_points:
        st.info("No Point-tagged rows found in the loaded files.")
    else:
        # Per-point stats
        cols_pts = st.columns(len(available_points))
        for col, pt in zip(cols_pts, available_points):
            with col:
                colour = POINT_COLOURS.get(pt, "#94A3B8")
                st.markdown(
                    f'<div style="border-left: 4px solid {colour}; padding-left: 10px;">'
                    f"<b>{pt}</b></div>",
                    unsafe_allow_html=True,
                )
                stats = summary_stats(combined, pt)
                for param in ["CH4 [%]", "CO2 [%]", "O2 [%]", "GCV [Kcal/SCM]", "Flow [l/min]", "R.Density [kg/SCM]"]:
                    if param in stats:
                        s = stats[param]
                        st.metric(param, f"{s['mean']:.4g}", f"±{s['std']:.3g}")

        st.divider()

        # Box plots per point
        st.subheader("Distribution by Point")
        box_param = st.selectbox(
            "Select parameter for box plot",
            ["CH4 [%]", "CO2 [%]", "O2 [%]", "GCV [Kcal/SCM]",
             "NCV [Kcal/SCM]", "Flow [l/min]", "R.Density [kg/SCM]",
             "Dew Point [°C]", "H2S Total [ppm]", "Methane Quality Index"],
        )
        sub_pts = combined[combined["Point"].notna()]
        if not sub_pts.empty:
            fig_box = px.box(
                sub_pts,
                x="Point",
                y=box_param,
                color="Point",
                color_discrete_map=POINT_COLOURS,
                points="outliers",
                title=f"{box_param} Distribution by Point",
            )
            _style_fig(fig_box, f"{box_param} Distribution by Point")
            st.plotly_chart(fig_box, use_container_width=True)

        # Multi-file bar compare
        if len(all_dfs) > 1:
            st.subheader("Cross-File Point Comparison")
            compare_param = st.selectbox(
                "Parameter for cross-file compare",
                ["CH4 [%]", "CO2 [%]", "GCV [Kcal/SCM]", "Flow [l/min]",
                 "R.Density [kg/SCM]", "Methane Quality Index"],
                key="cross_file_param",
            )
            st.plotly_chart(
                bar_comparison_chart(all_dfs, compare_param, file_names),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════
# TAB 7 — ALARMS
# ═══════════════════════════════════════════════════════════════
with tab_alarms:
    st.header("Alarms")

    alarm_rows = combined[combined["Has Alarm"]]
    total_alarms = len(alarm_rows)
    total_rows = len(combined)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total alarm rows", total_alarms)
    c2.metric("Total rows", total_rows)
    c3.metric("Alarm %", f"{100*total_alarms/max(total_rows,1):.1f}%")

    if total_alarms == 0:
        st.success("No alarms found in loaded files.")
    else:
        col_pie, col_bar = st.columns(2)
        with col_pie:
            st.plotly_chart(alarm_pie(combined), use_container_width=True)
        with col_bar:
            alarm_counts = alarm_rows["Alarm"].value_counts().reset_index()
            alarm_counts.columns = ["Alarm Type", "Count"]
            fig_alarm_bar = px.bar(
                alarm_counts,
                x="Count",
                y="Alarm Type",
                orientation="h",
                title="Alarm Counts",
                color="Count",
                color_continuous_scale="Reds",
            )
            _style_fig(fig_alarm_bar, "Alarm Counts")
            st.plotly_chart(fig_alarm_bar, use_container_width=True)

        # Timeline: alarm overlay on CH4
        st.subheader("Alarm Events on CH4 Timeline")
        fig_alarm_ts = go.Figure()
        # CH4 background
        sub_ch4 = combined.dropna(subset=["CH4 [%]"])
        fig_alarm_ts.add_trace(
            go.Scatter(
                x=sub_ch4["timestamp"], y=sub_ch4["CH4 [%]"],
                mode="lines", name="CH4 [%]",
                line=dict(color="#3B82F6", width=1.2),
                opacity=0.7,
            )
        )
        # Alarm markers
        fig_alarm_ts.add_trace(
            go.Scatter(
                x=alarm_rows["timestamp"],
                y=alarm_rows["CH4 [%]"].fillna(0),
                mode="markers",
                name="Alarm",
                marker=dict(color="#EF4444", size=6, symbol="x"),
                hovertext=alarm_rows["Alarm"],
                hovertemplate="<b>Alarm:</b> %{hovertext}<br>%{x|%H:%M:%S}<extra></extra>",
            )
        )
        _style_fig(fig_alarm_ts, "CH4 with Alarm Events Highlighted")
        st.plotly_chart(fig_alarm_ts, use_container_width=True)

        st.subheader("Alarm Detail Table")
        display_alarm = alarm_rows[["timestamp", "source_file", "Alarm", "CH4 [%]", "CO2 [%]", "Flow [l/min]"]].copy()
        st.dataframe(display_alarm, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 8 — RAW DATA
# ═══════════════════════════════════════════════════════════════
with tab_raw:
    st.header("Raw Data")

    file_select = st.selectbox("Select file", file_names)
    df_raw = all_dfs[file_names.index(file_select)]

    st.caption(f"{len(df_raw)} rows · {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")

    # Filter by point
    point_filter_options = ["All"] + [
        p for p in ["Point 1", "Point 2", "Point 3"]
        if (df_raw["Point"] == p).sum() > 0
    ]
    pf = st.selectbox("Filter by Point", point_filter_options)
    df_show = df_raw if pf == "All" else df_raw[df_raw["Point"] == pf]

    display_cols = [
        "timestamp", "Point", "Alarm",
        "CH4 [%]", "CO2 [%]", "O2 [%]",
        "H2S High [ppm]", "H2S Low [ppm]",
        "Dew Point [°C]", "Flow [l/min]", "Balance [%]",
        "GCV [Kcal/SCM]", "NCV [Kcal/SCM]", "R.Density [kg/SCM]",
        "Methane Quality Index",
    ]
    st.dataframe(df_show[display_cols], use_container_width=True)

    # Download
    csv_bytes = df_show.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_bytes,
        file_name=f"{file_select}_filtered.csv",
        mime="text/csv",
    )
