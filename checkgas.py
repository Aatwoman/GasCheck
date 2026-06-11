"""
Biogas Plant Analytics Dashboard · Streamlit v11
=================================================
CHANGES v11:
 1.  KPI: Raw material split — Dung + Waste Potato in same card
 2.  KPI: Biogas Yield (m³/ton feedstock) card
 3.  KPI: Total CBG Sale card — Dispenser + Cascade combined
 4.  KPI: MFM Reading card (bg_mfm_kwh_total)
 5.  KPI: Electricity Consumed card (vpsa_kwh_total)
 6.  KPI: Optimum-range indicator pill shown where applicable
 7.  KPI layout: 5-column first row + 5-column second row, all visible above graphs
 8.  Graphs: All replaced with bar+line(rolling avg) via bar_line_fig()
 9.  Rolling avg window selectable 1–14 days (sidebar + per-tab override)
10.  Legend items clickable to show/hide series (Plotly interactive legend)
11.  Weekly date select added to sidebar (ISO week picker)
12.  Month vs Month comparison tab (same plant) — select any 2 months, overlay daily trends

CHANGES v10:
 1.  Lab tab per-plant per-parameter charts, vertical legend
 2.  Lab daily aggregation + IQR×3 outlier removal
 3.  Lab rolling-average slider and outlier-toggle checkbox
 4.  TS vs VS scatter with numpy polyfit R²
 5.  H₂S threshold lines removed
 6.  CO₂ removed from gas composition charts
 7.  Gas tab local MA-window slider
 8.  Formula / source annotations in chart titles
 9.  Compare lab sub-section per-parameter per-sample-point panels
"""

import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Biogas Plant Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (full light mode including sidebar) ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html,body,.stApp,[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],[data-testid="block-container"],
section[data-testid="stMain"] {
    background:#f4f7fb !important;
    color:#1e2d45 !important;
    font-family:'Inter',sans-serif !important;
}
:root {
    --background-color:#f4f7fb !important;
    --secondary-background-color:#eaf0f9 !important;
    --text-color:#1e2d45 !important;
    --primary-color:#1a56db !important;
}

/* ── SIDEBAR – full light mode ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebar"] section {
    background:#ffffff !important;
    border-right:1px solid #dde6f4 !important;
}
/* All sidebar text dark */
[data-testid="stSidebar"] *:not(button):not(input):not(.stSlider *) {
    color:#1e2d45 !important;
}
[data-testid="stSidebar"] label  { color:#374f6b !important; font-size:0.82rem !important; font-weight:500 !important; }
[data-testid="stSidebar"] p      { color:#374f6b !important; }
[data-testid="stSidebar"] small  { color:#5a7a9a !important; }
/* Slider track and thumb */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { background:#e2eaf5 !important; }
/* Selectbox / multiselect */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background:#f0f5fc !important;
    border-color:#c5d5eb !important;
    color:#1e2d45 !important;
}
/* Radio buttons */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label { color:#374f6b !important; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child {
    background:#1a56db !important;
}
/* File uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background:#f0f5fc !important;
    border:1.5px dashed #a8c0e0 !important;
    border-radius:8px !important;
}
/* Text inputs */
[data-testid="stSidebar"] input[type="text"] {
    background:#f0f5fc !important;
    border:1px solid #c5d5eb !important;
    color:#1e2d45 !important;
    border-radius:6px !important;
}

/* Sidebar header card */
.sb-header {
    background:linear-gradient(135deg,#1a56db 0%,#2563eb 100%);
    border-radius:10px;
    padding:14px 16px;
    margin-bottom:14px;
    color:#ffffff !important;
}
.sb-header h2 { color:#ffffff !important; font-family:'Space Mono',monospace !important;
    font-size:1rem !important; margin:0 !important; letter-spacing:0.05em !important; }
.sb-header p  { color:#c7d9ff !important; font-size:0.73rem !important; margin:3px 0 0 !important; }

/* Sidebar section dividers */
.sb-section {
    border-top:1px solid #dde6f4;
    margin:10px 0 6px;
    padding-top:8px;
}
.sb-section-label {
    font-size:0.68rem !important;
    font-weight:700 !important;
    letter-spacing:0.1em !important;
    text-transform:uppercase !important;
    color:#8aaac8 !important;
    margin-bottom:6px !important;
}

/* ── Main content ── */
[data-testid="stMainBlockContainer"] { padding:1.4rem 2rem !important; }

/* ── KPI cards ── */
.kpi-row { display:flex; gap:10px; margin:10px 0 16px; flex-wrap:wrap; }
.kpi-card {
    flex:1; min-width:160px;
    background:#fff;
    border:1px solid #dde6f4;
    border-top:3px solid #1a56db;
    border-radius:10px;
    padding:13px 15px 11px;
    box-shadow:0 2px 6px rgba(26,86,219,.07);
}
.kpi-icon  { font-size:1.15rem; margin-bottom:5px; }
.kpi-value { font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700;
    color:#1a56db; line-height:1.1; }
.kpi-label { font-size:0.67rem; font-weight:600; color:#5a7a9a;
    margin-top:4px; text-transform:uppercase; letter-spacing:.07em; }

/* ── Section headers ── */
.sec-hdr {
    background:#eaf0fc;
    border-left:3px solid #1a56db;
    color:#1e2d45;
    padding:8px 15px;
    border-radius:0 8px 8px 0;
    font-family:'Space Mono',monospace;
    font-size:0.78rem; font-weight:700;
    letter-spacing:0.06em; text-transform:uppercase;
    margin:18px 0 8px;
}

/* ── Badges ── */
.plant-badge {
    display:inline-block; background:#ddeeff; border:1px solid #90caf9;
    border-radius:20px; padding:2px 11px;
    font-family:'Space Mono',monospace; font-size:0.74rem; color:#1a56db;
    margin:2px 3px;
}
.compare-badge {
    display:inline-block; background:#fff3e0; border:1px solid #ffb74d;
    border-radius:20px; padding:2px 11px;
    font-family:'Space Mono',monospace; font-size:0.74rem; color:#c84b00;
    margin:2px 3px;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#e2eaf5; border-radius:10px; padding:3px; gap:2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'Inter',sans-serif; font-size:0.76rem; font-weight:500;
    padding:6px 11px; border-radius:7px; color:#4a6a8a; background:transparent;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:#fff !important; color:#1a56db !important;
    box-shadow:0 1px 4px rgba(26,86,219,.12);
}

/* ── Chart wrappers ── */
[data-testid="stPlotlyChart"] {
    background:#fff; border:1px solid #dde6f4;
    border-radius:10px; overflow:hidden;
    box-shadow:0 1px 4px rgba(26,86,219,.04);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border:1px solid #dde6f4; border-radius:8px; }

/* ── Expander ── */
[data-testid="stExpander"],[data-testid="stExpander"]>div {
    background:#fff; border:1px solid #dde6f4; border-radius:8px;
}

/* ── Download buttons ── */
.stDownloadButton button {
    background:#1a56db !important; color:#fff !important;
    border:none !important; border-radius:6px !important;
    font-size:0.82rem !important;
}

/* ── Mode banner ── */
.mode-banner {
    background:linear-gradient(90deg,#e8f0fe 0%,#f0f4fc 100%);
    border:1px solid #b8d0f8; border-radius:8px;
    padding:7px 14px; margin-bottom:10px;
    font-size:0.82rem; color:#1e2d45; font-weight:500;
}

hr { border-color:#dde6f4 !important; margin:.8rem 0 !important; }

/* Keep white background in alerts */
[data-testid="stInfo"]    { background:#e8f4fd !important; }
[data-testid="stWarning"] { background:#fff8e1 !important; }
[data-testid="stSuccess"] { background:#e8f5e9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Palette / theme ───────────────────────────────────────────────────────────
PALETTE = ["#1a56db","#2e7d32","#c84b00","#7b1fa2",
           "#00838f","#b71c1c","#4527a0","#558b2f","#ad1457","#00695c"]
CHART_BG   = "#ffffff"
CHART_GRID = "#eaf0f8"
FONT_COLOR = "#1e2d45"
AXIS_COLOR = "#7a96b2"

# ── Column definitions ────────────────────────────────────────────────────────
SEEK = {
    "date":                ["date"],
    "dung_tons":           ["dung (tons)","dung\n(tons)","dung"],
    "waste_potato_tons":   ["waste potato"],
    "total_feed_m3":       ["total feed to reactor"],
    "total_filter_water":  ["total filter water consumed"],
    "raw_ch4":             ["ch₄","ch4"],
    "raw_co2":             ["co₂","co2"],
    "raw_o2":              ["o₂","o2"],
    "raw_h2s":             ["h₂s","h2s"],
    "raw_bal":             ["bal (%)"],
    "total_generated_gas": ["total generated gas"],
    "total_raw_gas":       ["total raw gas"],
    "gen_inlet_diff":      ["gen-inlet"],
    "total_purified_gas":  ["total purified gas"],
    "expected_gas_kg":     ["expected gas"],
    "cbg_mass_fm_kg":      ["cbg mass fm"],
    "pure_gas_purity_fm":  ["pure gas purity in fm","pure gas purity"],
    "cbg_sales_kg":        ["total cbg sales dispenser","total cbg sales"],
    "num_vehicles":        ["no. of vehicles","no of vehicles"],
    "cascade_sales_kg":    ["cascade vehicle sales"],
    "purif_efficiency":    ["purification efficiency (%)"],
    "purif_running_hrs":   ["purification running hrs"],
    "compressor_hrs":      ["compressor running hrs"],
    "screw_press_hrs":     ["screw press running hrs"],
    "vibro_screen_hrs":    ["vibro screen running hrs"],
    "volute_press_hrs":    ["volute press running hrs"],
    "screw_moisture":      ["screw press moisture"],
    "volute_moisture":     ["volute press moisture"],
    "raw_water_m3":        ["raw water"],
    "digester_ph":         ["mid ph"],
    "digester_temp":       ["digester temp"],
    "flare_m3":            ["flare"],
    "poly_kg":             ["poly consumption"],
    "dg_hrs":              ["dg running hrs"],
    "dg_diesel_l":         ["dg diesel consumed"],
    "purif_eff_calc":      ["purif. eff."],
    "bg_recovery":         ["bg recovery"],
    "remarks":             ["remarks"],
}
_SECOND = {"pure_ch4":["ch₄","ch4"],"pure_co2":["co₂","co2"],"pure_h2s":["h₂s","h2s"]}

COL_LABELS = {
    "dung_tons":"Dung Collected (tons)","waste_potato_tons":"Waste Potato (tons)",
    "total_feed_m3":"Total Feed to Reactor (m³)","total_filter_water":"Filter Water (m³)",
    "raw_ch4":"Raw CH₄ (%)","raw_co2":"Raw CO₂ (%)","raw_o2":"Raw O₂ (%)","raw_h2s":"Raw H₂S (PPM)",
    "raw_bal":"Raw Balance (%)","total_generated_gas":"Total Generated Gas (m³)",
    "total_raw_gas":"Total Raw Gas (m³)","gen_inlet_diff":"Gen–Inlet Diff (m³)",
    "total_purified_gas":"Total Purified Gas (m³)","expected_gas_kg":"Expected Gas (kg)",
    "cbg_mass_fm_kg":"CBG Mass FM (kg)","pure_gas_purity_fm":"Pure Gas Purity FM (%)",
    "cbg_sales_kg":"CBG Sales Dispenser (kg)","num_vehicles":"No. of Vehicles",
    "cascade_sales_kg":"Cascade Sales (kg)","total_sales_kg":"Total CBG Sales (kg)",
    "purif_efficiency":"Purification Efficiency (%)","purif_running_hrs":"Purification Running Hrs",
    "compressor_hrs":"Compressor Running Hrs","screw_press_hrs":"Screw Press Running Hrs",
    "vibro_screen_hrs":"Vibro Screen Running Hrs","volute_press_hrs":"Volute Press Running Hrs",
    "screw_moisture":"Screw Press Moisture (%)","volute_moisture":"Volute Press Moisture (%)",
    "raw_water_m3":"Raw Water (m³)","digester_ph":"Digester pH","digester_temp":"Digester Temp (°C)",
    "flare_m3":"Flare Gas (m³)","poly_kg":"Poly Consumption (kg)","dg_hrs":"DG Running Hrs",
    "dg_diesel_l":"DG Diesel Consumed (L)","purif_eff_calc":"Purif. Eff. Calc (%)","bg_recovery":"BG Recovery (%)",
    "pure_ch4":"Pure CH₄ (%)","pure_co2":"Pure CO₂ (%)","pure_h2s":"Pure H₂S (PPM)",
    "vpsa_kwh_total":"VPSA KWH Total","bg_mfm_kwh_total":"BG MFM KWH Total",
    "hp_comp_kwh_init":"HP Compressor KWH Init","hp_comp_kwh_final":"HP Compressor KWH Final",
    "remarks":"Remarks",
}

# ── Outlier handling ──────────────────────────────────────────────────────────
def _iqr_clip(s: pd.Series, factor: float = 3.0) -> pd.Series:
    """Replace IQR-outliers with NaN (non-destructive)."""
    if s.dropna().empty:
        return s
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return s.where((s >= lo) & (s <= hi))


# Columns where 0 is physically impossible / clearly erroneous
_NONZERO_COLS = {
    "total_generated_gas", "total_purified_gas", "total_raw_gas",
    "raw_ch4", "pure_ch4", "dung_tons", "total_feed_m3",
}

def _clean_ops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean operational data:
    • Replace zeros that are physically impossible with NaN
    • IQR-clip extreme outliers per column
    • Fix purification efficiency entered as fraction (0.xx → xx.x)
    """
    df = df.copy()

    # Fix fractional purif_efficiency (e.g., 0.9635 instead of 96.35)
    if "purif_efficiency" in df.columns:
        mask_frac = df["purif_efficiency"].notna() & (df["purif_efficiency"] < 5)
        df.loc[mask_frac, "purif_efficiency"] = df.loc[mask_frac, "purif_efficiency"] * 100

    # Zero → NaN for physically impossible zeros
    for col in _NONZERO_COLS:
        if col in df.columns:
            df[col] = df[col].where(df[col] != 0)

    # IQR outlier clipping for key continuous columns
    _clip_cols = [
        "total_generated_gas", "total_purified_gas", "total_raw_gas",
        "gen_inlet_diff", "dung_tons", "total_feed_m3",
        "purif_efficiency", "digester_temp", "digester_ph",
        "cbg_sales_kg", "vpsa_kwh_total",
    ]
    for col in _clip_cols:
        if col in df.columns:
            df[col] = _iqr_clip(df[col])

    return df


# ── Header / column detection ─────────────────────────────────────────────────
def _find_header_rows(raw):
    for r in range(min(6, len(raw))):
        v = str(raw.iloc[r, 0]).replace("\n"," ").strip().lower()
        if v == "date":
            return max(0, r-1), r
    return 0, 1


def _build_col_index(raw):
    sec_row, hdr_row = _find_header_rows(raw)
    header  = [str(v).replace("\n"," ").strip().lower() if pd.notna(v) else "" for v in raw.iloc[hdr_row]]
    section = [str(v).replace("\n"," ").strip().lower() if pd.notna(v) else "" for v in raw.iloc[sec_row]]
    idx = {}
    skip = set(_SECOND.keys()) | {"vpsa_kwh_total","bg_mfm_kwh_total","hp_comp_kwh_init","hp_comp_kwh_final"}
    for key, needles in SEEK.items():
        if key in skip: continue
        for needle in needles:
            nl = needle.lower()
            for c, h in enumerate(header):
                if nl in h:
                    idx[key] = c; break
            if key in idx: break
    for pk, needles in _SECOND.items():
        rk = pk.replace("pure_","raw_")
        for needle in needles:
            nl = needle.lower()
            matches = [c for c,h in enumerate(header) if nl in h]
            if len(matches)>=1 and rk not in idx: idx[rk] = matches[0]
            if len(matches)>=2: idx[pk] = matches[1]
            if rk in idx and pk in idx: break
    kwh = [c for c,h in enumerate(header) if "total kwh consumed" in h]
    if len(kwh)>=1: idx["vpsa_kwh_total"] = kwh[0]
    if len(kwh)>=2: idx["bg_mfm_kwh_total"] = kwh[1]
    hp = [c for c,s in enumerate(section) if "hp compressor" in s]
    for c in hp:
        h = header[c] if c < len(header) else ""
        if "initial" in h:   idx["hp_comp_kwh_init"]  = c
        elif "final" in h:   idx["hp_comp_kwh_final"] = c
    return idx


def _to_num(s):
    if not isinstance(s, pd.Series): return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")


# ── Loaders ───────────────────────────────────────────────────────────────────
def _read_sheet(wb_bytes, sheet_name, fname=""):
    """Read an Excel sheet, choosing engine based on file extension."""
    fname_lower = (fname or "").lower()
    if fname_lower.endswith(".xlsb"):
        try:
            import pyxlsb  # noqa
            return pd.read_excel(io.BytesIO(wb_bytes), sheet_name=sheet_name,
                                 header=None, engine="pyxlsb")
        except ImportError:
            st.error("pyxlsb is not installed. Add `pyxlsb` to requirements.txt.")
            raise
    return pd.read_excel(io.BytesIO(wb_bytes), sheet_name=sheet_name, header=None)


def load_daily_operations(wb_bytes, plant_name, fname=""):
    import gc
    raw = _read_sheet(wb_bytes, "Daily Operations", fname=fname)
    _, hdr = _find_header_rows(raw)
    ds = hdr + 2
    for r in range(ds, min(ds+5, len(raw))):
        try:
            v = raw.iloc[r, 0]
            if pd.notna(v): pd.Timestamp(v); ds = r; break
        except Exception: pass
    col_idx = _build_col_index(raw)
    data = raw.iloc[ds:].reset_index(drop=True)
    del raw; gc.collect()  # free the full sheet immediately
    all_keys = list(SEEK.keys()) + list(_SECOND.keys()) + \
               ["vpsa_kwh_total","bg_mfm_kwh_total","hp_comp_kwh_init","hp_comp_kwh_final"]
    records = {}
    for key in all_keys:
        c = col_idx.get(key)
        records[key] = data.iloc[:,c].values if (c is not None and c < data.shape[1]) \
                       else np.full(len(data), np.nan, dtype=object)
    del data; gc.collect()  # free the sliced sheet
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in df.columns:
        if col not in ("date","remarks"): df[col] = _to_num(df[col])
    df["total_sales_kg"] = df["cbg_sales_kg"].fillna(0) + df["cascade_sales_kg"].fillna(0)
    df["plant"] = plant_name
    df = _clean_ops(df)
    # Downcast float64 → float32 to halve memory usage
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    return df


def load_lab_analysis(wb_bytes, plant_name, fname=""):
    import gc
    raw = _read_sheet(wb_bytes, "Lab & Slurry Analysis", fname=fname)
    data = raw.iloc[3:].reset_index(drop=True).copy()
    del raw; gc.collect()
    data.columns = range(data.shape[1])
    data.rename(columns={0:"date",1:"sample_point",2:"pH",3:"EC_mScm",
                          4:"TS_pct",5:"VS_pct",6:"Temp_C",7:"Carbon_pct"}, inplace=True)
    data["date"] = data["date"].ffill()
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["date","sample_point"])
    data["sample_point"] = data["sample_point"].astype(str).str.strip()
    data = data[~data["sample_point"].str.lower().str.contains("sample point|notes|nan",na=False)]
    for col in ["pH","EC_mScm","TS_pct","VS_pct","Temp_C","Carbon_pct"]:
        if col in data.columns: data[col] = _to_num(data[col])
    for col,lo,hi in [("TS_pct",0,100),("VS_pct",0,100),("pH",0,14)]:
        if col in data.columns:
            data = data[~(data[col].notna() & ~data[col].between(lo,hi))]
    data["plant"] = plant_name
    cols = ["date","plant","sample_point","pH","EC_mScm","TS_pct","VS_pct","Temp_C","Carbon_pct"]
    return data[cols].reset_index(drop=True)


def load_dung_quality(wb_bytes, plant_name, fname=""):
    import gc
    raw = _read_sheet(wb_bytes, "Dung Route Quality", fname=fname)
    route_row = raw.iloc[0].copy()
    subcol_row = raw.iloc[1].copy()
    data = raw.iloc[3:].reset_index(drop=True).copy()
    del raw; gc.collect()
    records, cur = [], None
    for c in range(1, data.shape[1], 4):
        if c < len(route_row) and pd.notna(route_row.iloc[c]):
            cur = str(route_row.iloc[c]).strip()
        if cur is None: continue
        sub = [str(subcol_row.iloc[c+k]).strip() if (c+k)<len(subcol_row) and pd.notna(subcol_row.iloc[c+k]) else f"sub{k}" for k in range(4)]
        for _, row in data.iterrows():
            dv = pd.to_datetime(row.iloc[0], dayfirst=True, errors="coerce")
            if pd.isna(dv): continue
            rec = {"date":dv,"route":cur,"plant":plant_name}
            for k,sn in enumerate(sub):
                v = row.iloc[c+k] if (c+k)<len(row) else np.nan
                rec[sn] = pd.to_numeric(v, errors="coerce")
            records.append(rec)
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True) if records else pd.DataFrame()


def load_fertilizer_quality(wb_bytes, plant_name, fname=""):
    import gc
    raw = _read_sheet(wb_bytes, "Fertilizer Quality", fname=fname)
    hi = 2
    for r in range(min(6, len(raw))):
        if str(raw.iloc[r,0]).replace("\n"," ").strip().lower().startswith("sr"):
            hi = r; break
    headers = [str(h).replace("\n"," ").strip() for h in raw.iloc[hi]]
    data = raw.iloc[hi+1:].reset_index(drop=True).copy()
    del raw; gc.collect()
    data.columns = headers
    sr_col = headers[0]
    data = data[pd.to_numeric(data[sr_col], errors="coerce").notna()].copy()
    non_num = {sr_col,"Sr. No.","Sr.\nNo.","Sample Date","Sample\nDate",
               "Material Name","Material\nName","Batch / Type","Batch /\nType",
               "Mfg Date / Month","Mfg Date\n/ Month","Remarks / Sampler","Remarks /\nSampler"}
    for col in data.columns:
        if col in non_num: continue
        if isinstance(data[col], pd.Series) and data[col].dtype==object:
            try: data[col] = pd.to_numeric(data[col], errors="coerce")
            except Exception: pass
    data["plant"] = plant_name
    return data.reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def load_plant(file_bytes, plant_name, fname=""):
    def _s(fn, label):
        try: return fn(file_bytes, plant_name, fname=fname)
        except Exception as e:
            st.warning(f"⚠ [{plant_name}] {label}: {e}"); return pd.DataFrame()
    return {"ops":_s(load_daily_operations,"Daily Operations"),
            "lab":_s(load_lab_analysis,"Lab & Slurry Analysis"),
            "dung":_s(load_dung_quality,"Dung Route Quality"),
            "fert":_s(load_fertilizer_quality,"Fertilizer Quality")}


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _pmap(plants):
    return {p:PALETTE[i%len(PALETTE)] for i,p in enumerate(sorted(plants))}

def _hex_rgba(hx, a=0.15):
    h=hx.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def _ma(s, w):
    return s.rolling(w, min_periods=1).mean()

def _xrange(df):
    """Compute x-axis range from the date filter stored in session_state."""
    df_flt = st.session_state.get("_date_filter",{})
    if not df_flt: return None
    s = df_flt.get("start"); e = df_flt.get("end")
    if s is None or e is None: return None
    return [(s-pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            (e+pd.Timedelta(days=1)).strftime("%Y-%m-%d")]

def _base(fig, height=500, xr=None):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, family="Inter,sans-serif", size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5, bgcolor="rgba(255,255,255,.9)",
                    bordercolor="#dde6f4", borderwidth=1, font=dict(size=11,color=FONT_COLOR)),
        hovermode="x unified", height=height, title_x=0,
        title_font=dict(size=13, color="#1e2d45", family="Space Mono,monospace"),
        margin=dict(l=10, r=10, t=44, b=80),
    )
    xkw = dict(showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
               zeroline=False, showline=True, linecolor="#dde6f4",
               tickfont=dict(size=11,color=AXIS_COLOR), tickcolor=AXIS_COLOR,
               title_font=dict(color=AXIS_COLOR), type="date", autorange=(xr is None))
    if xr is not None: xkw["range"] = xr
    fig.update_xaxes(**xkw)
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
                     zeroline=False, showline=False,
                     tickfont=dict(size=11,color=AXIS_COLOR), title_font=dict(color=AXIS_COLOR))
    return fig


def bar_line_fig(df, x, ycol, title, ylab="", ma=7, height=420, xr=None,
                 opt_low=None, opt_high=None):
    """
    Bar chart with daily values + rolling-average line on top.
    Legend items are click-to-hide (Plotly interactive legend).
    Optional optimum range band via opt_low / opt_high.
    """
    fig = go.Figure()
    if df.empty or ycol not in df.columns:
        return _base(fig, height, xr)
    cmap = _pmap(df["plant"].unique())
    for p, gdf in df.groupby("plant"):
        s = gdf[ycol]; c = cmap[p]; valid = s.notna().sum()
        if valid == 0:
            continue
        # ── Bar (daily) ──────────────────────────────────────────────────────
        fig.add_trace(go.Bar(
            x=gdf[x], y=s,
            name=f"{p} daily",
            marker_color=_hex_rgba(c, 0.55),
            marker_line_width=0,
            legendgroup=p,
        ))
        # ── Rolling avg line ─────────────────────────────────────────────────
        if valid >= 1:
            ma_w = max(1, min(ma, valid))
            fig.add_trace(go.Scatter(
                x=gdf[x], y=_ma(s, ma_w),
                name=f"{p} {ma_w}d avg",
                mode="lines",
                line=dict(color=c, width=2.6),
                legendgroup=p,
            ))
    # Optimum range band
    if opt_low is not None and opt_high is not None:
        fig.add_hrect(
            y0=opt_low, y1=opt_high,
            fillcolor="#2e7d32", opacity=0.06,
            line_width=0,
            annotation_text=f"Optimal {opt_low}–{opt_high}",
            annotation_font_color="#2e7d32",
            annotation_font_size=10,
        )
    fig.update_layout(
        title=title, yaxis_title=ylab,
        barmode="group",
        bargap=0.15, bargroupgap=0.05,
    )
    return _base(fig, height, xr)


def line_fig(df, x, ycol, title, ylab="", ma=7, height=500, xr=None):
    fig = go.Figure()
    if df.empty or ycol not in df.columns:
        return _base(fig, height, xr)
    cmap = _pmap(df["plant"].unique())
    for p, gdf in df.groupby("plant"):
        s = gdf[ycol]; c = cmap[p]; valid = s.notna().sum()
        if valid == 0: continue
        fig.add_trace(go.Scatter(x=gdf[x], y=s, mode="lines", name=p,
                                  line=dict(color=c,width=1.2), opacity=0.28,
                                  showlegend=(ma<=1)))
        if ma > 1 and valid >= ma:
            fig.add_trace(go.Scatter(x=gdf[x], y=_ma(s,ma), mode="lines",
                                      name=f"{p} ({ma}d avg)",
                                      line=dict(color=c,width=2.8), opacity=1.0))
        elif valid > 0 and ma <= 1:
            fig.data[-1].update(opacity=0.88, line=dict(width=2.2))
    fig.update_layout(title=title, yaxis_title=ylab)
    return _base(fig, height, xr)


def dual_fig(df, x, ca, la, cb, lb, title, height=500, xr=None):
    fig = go.Figure()
    if df.empty: return _base(fig, height, xr)
    cmap = _pmap(df["plant"].unique())
    for p, gdf in df.groupby("plant"):
        c = cmap[p]
        if ca in gdf.columns and gdf[ca].notna().any():
            fig.add_trace(go.Scatter(x=gdf[x],y=gdf[ca],name=f"{p} – {la}",
                                      line=dict(color=c,width=2.4)))
        if cb in gdf.columns and gdf[cb].notna().any():
            fig.add_trace(go.Scatter(x=gdf[x],y=gdf[cb],name=f"{p} – {lb}",
                                      line=dict(color=c,width=2.4,dash="dot")))
    fig.update_layout(title=title)
    return _base(fig, height, xr)


def bar_fig(df, x, y, title, color="plant", height=460):
    cmap = _pmap(df["plant"].unique()) if "plant" in df.columns else {}
    fig = px.bar(df, x=x, y=y, color=color, barmode="group",
                 title=title, color_discrete_map=cmap)
    fig.update_traces(marker_line_width=0)
    fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                      font=dict(color=FONT_COLOR,family="Inter,sans-serif"),
                      title_font=dict(size=13,color="#1e2d45",family="Space Mono,monospace"),
                      legend=dict(orientation="h",yanchor="top",y=-0.18,xanchor="center",x=0.5,
                                  bgcolor="rgba(255,255,255,.9)",bordercolor="#dde6f4",
                                  borderwidth=1,font=dict(color=FONT_COLOR)),
                      height=height, title_x=0, margin=dict(l=10,r=10,t=44,b=80))
    fig.update_xaxes(showgrid=False,linecolor="#dde6f4",tickfont=dict(color=AXIS_COLOR))
    fig.update_yaxes(showgrid=True,gridcolor=CHART_GRID,tickfont=dict(color=AXIS_COLOR))
    return fig


def scatter_fig(df, x, y, title, color="sample_point", height=520):
    """Scatter with per-group numpy trendlines — no statsmodels dependency."""
    cmap = _pmap(df[color].unique()) if color in df.columns else {}
    fig = go.Figure()
    groups = df.groupby(color) if color in df.columns else [("All", df)]
    color_list = list(cmap.values()) if cmap else PALETTE

    for i, (grp, gdf) in enumerate(groups):
        c = cmap.get(grp, color_list[i % len(color_list)])
        valid = gdf[[x, y]].dropna()
        # scatter points
        fig.add_trace(go.Scatter(
            x=valid[x], y=valid[y], mode="markers", name=str(grp),
            marker=dict(color=c, size=7, opacity=0.72,
                        line=dict(color="white", width=0.5)),
        ))
        # numpy trendline (no external dep)
        if len(valid) >= 3:
            try:
                m, b = np.polyfit(valid[x].values.astype(float),
                                  valid[y].values.astype(float), 1)
                xmin, xmax = valid[x].min(), valid[x].max()
                x_line = np.array([xmin, xmax])
                fig.add_trace(go.Scatter(
                    x=x_line, y=m * x_line + b,
                    mode="lines", name=f"{grp} trend",
                    line=dict(color=c, width=1.8, dash="dash"),
                    showlegend=False,
                ))
                # Add R² annotation in hover
                ss_res = np.sum((valid[y].values - (m * valid[x].values + b)) ** 2)
                ss_tot = np.sum((valid[y].values - valid[y].mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                fig.add_annotation(
                    x=xmax, y=m * xmax + b,
                    text=f"R²={r2:.2f}", showarrow=False,
                    font=dict(size=10, color=c), xanchor="right",
                )
            except Exception:
                pass

    fig.update_layout(
        title=title,
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, family="Inter,sans-serif"),
        title_font=dict(size=13, color="#1e2d45", family="Space Mono,monospace"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,.9)", bordercolor="#dde6f4",
                    borderwidth=1, font=dict(color=FONT_COLOR)),
        height=height, title_x=0, margin=dict(l=10, r=10, t=44, b=80),
    )
    fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID,
                     tickfont=dict(color=AXIS_COLOR), title_font=dict(color=AXIS_COLOR))
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID,
                     tickfont=dict(color=AXIS_COLOR), title_font=dict(color=AXIS_COLOR))
    return fig


def sec(text):
    st.markdown(f'<div class="sec-hdr">{text}</div>', unsafe_allow_html=True)

def _has_data(df, col):
    """Return True if col exists in df, has at least one non-null, non-zero value."""
    if df is None or df.empty or col not in df.columns:
        return False
    s = df[col].dropna()
    return len(s) > 0 and (s != 0).any()

def _pc(fig, key, height=None):
    if height: fig.update_layout(height=height)
    # Check whether the figure actually has any visible data
    has_visible = any(
        (tr.y is not None and len(tr.y) > 0) or
        (tr.r is not None and len(tr.r) > 0)
        for tr in fig.data
    )
    if not has_visible:
        return  # silently skip empty charts
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        # Header card
        st.markdown("""
<div class="sb-header">
  <h2>⚡ BIOGAS ANALYTICS</h2>
  <p>Unified Daily Report · Multi-Plant</p>
</div>""", unsafe_allow_html=True)

        # Upload section
        st.markdown('<div class="sb-section-label">📂 DATA SOURCE</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload Excel file(s)", type=["xlsx", "xlsb"], accept_multiple_files=True,
            help="One .xlsx or .xlsb per plant — Unified Daily Report format.",
            label_visibility="collapsed",
        )

        all_data = {}
        if uploaded:
            for f in uploaded:
                rb = f.read()
                default = f.name.replace(".xlsb","").replace(".xlsx","").replace("_"," ").title()
                pname = st.text_input(f"Label: {f.name[:28]}", value=default, key=f"pn_{f.name}")
                with st.spinner(f"Loading {pname}…"):
                    all_data[pname] = load_plant(rb, pname, fname=f.name)
                del rb  # free upload bytes immediately after caching

        if not all_data:
            st.info("⬆ Upload one or more plant Excel files to begin.")
            return {}, [], {}, "individual"

        # Plant / view section
        st.markdown('<div class="sb-section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">🏭 PLANT SELECTION</div>', unsafe_allow_html=True)
        plants = list(all_data.keys())

        if len(plants) > 1:
            view_mode = st.radio("View mode", ["individual","compare"],
                                 format_func=lambda x: "📋 Single plant" if x=="individual" else "📊 Compare plants",
                                 horizontal=True, key="view_mode")
        else:
            view_mode = "individual"

        if view_mode == "individual":
            sel_plant = st.selectbox("Select plant", plants, key="sel_ind")
            selected = [sel_plant]
        else:
            selected = st.multiselect("Plants to compare", plants, default=plants, key="sel_cmp")
            if not selected: selected = plants

        # Date filter section
        all_ops = [all_data[p]["ops"] for p in selected if p in all_data and not all_data[p]["ops"].empty]
        date_filter = {}

        if all_ops:
            combined = pd.concat(all_ops, ignore_index=True)
            data_min = combined["date"].min().date()
            data_max = combined["date"].max().date()

            st.markdown('<div class="sb-section"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-section-label">📅 DATE FILTER</div>', unsafe_allow_html=True)

            ftype = st.radio("Filter by", ["All Data","Month picker","Week picker","Custom range"],
                             key="ftype", label_visibility="collapsed")

            if ftype == "Month picker":
                all_months = [str(m) for m in pd.period_range(
                    pd.Period(data_min,"M"), pd.Period(data_max,"M"), freq="M")]
                chosen = st.multiselect("Months", all_months, default=all_months, key="months")
                if chosen:
                    periods = [pd.Period(m,"M") for m in chosen]
                    date_filter = {"start":min(p.start_time for p in periods),
                                   "end":  max(p.end_time   for p in periods),
                                   "months": chosen}
            elif ftype == "Week picker":
                # Build list of ISO weeks in data range
                all_weeks = pd.date_range(data_min, data_max, freq="W-MON")
                if len(all_weeks) == 0:
                    all_weeks = pd.date_range(data_min, data_max + pd.Timedelta(days=7), freq="W-MON")
                week_labels = [f"W{d.isocalendar()[1]:02d} {d.year}  ({d.strftime('%d %b')}–{(d+pd.Timedelta(days=6)).strftime('%d %b')})"
                               for d in all_weeks]
                week_map = {lbl: d for lbl, d in zip(week_labels, all_weeks)}
                chosen_weeks = st.multiselect("Select week(s)", week_labels,
                                               default=week_labels[-1:] if week_labels else [],
                                               key="weeks")
                if chosen_weeks:
                    starts = [week_map[w] for w in chosen_weeks]
                    ends   = [s + pd.Timedelta(days=6) for s in starts]
                    date_filter = {"start": pd.Timestamp(min(starts)),
                                   "end":   pd.Timestamp(min(max(ends), pd.Timestamp(data_max)))}
            elif ftype == "Custom range":
                dr = st.date_input("Range", value=(data_min, data_max),
                                   min_value=data_min, max_value=data_max, key="dr")
                if isinstance(dr,(list,tuple)) and len(dr)==2:
                    date_filter = {"start":pd.Timestamp(dr[0]),"end":pd.Timestamp(dr[1])}
                elif isinstance(dr,(list,tuple)) and len(dr)==1:
                    date_filter = {"start":pd.Timestamp(dr[0]),"end":pd.Timestamp(dr[0])}
            else:
                date_filter = {"start":pd.Timestamp(data_min),"end":pd.Timestamp(data_max)}

            # Show active range as pill
            if date_filter:
                s_str = date_filter["start"].strftime("%d %b %y")
                e_str = date_filter["end"].strftime("%d %b %y")
                st.markdown(f"<small style='color:#5a7a9a'>📆 {s_str} → {e_str}</small>",
                            unsafe_allow_html=True)

        # Chart options
        st.markdown('<div class="sb-section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">⚙ CHART OPTIONS</div>', unsafe_allow_html=True)
        st.slider("Moving average (days)", 1, 30, 7, key="ma_window")

        st.markdown('<div class="sb-section"></div>', unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#8aaac8'>Outliers are auto-removed via IQR · "
            "Zero-values in key columns treated as missing data</small>",
            unsafe_allow_html=True)

        # Store date_filter in session for _xrange()
        st.session_state["_date_filter"] = date_filter
        return all_data, selected, date_filter, view_mode


# ── Data filters ──────────────────────────────────────────────────────────────
def _flt(df, df_flt):
    if df.empty or not df_flt: return df
    if "months" in df_flt:
        ps = [pd.Period(m,"M") for m in df_flt["months"]]
        return df[df["date"].apply(lambda d: pd.Period(d,"M") in ps)]
    return df[(df["date"]>=df_flt["start"]) & (df["date"]<=df_flt["end"])]

def get_ops(all_data, selected, df_flt):
    frames = [_flt(all_data[p]["ops"], df_flt) for p in selected
              if p in all_data and not all_data[p]["ops"].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def get_lab(all_data, selected, df_flt):
    frames = [_flt(all_data[p]["lab"], df_flt) for p in selected
              if p in all_data and not all_data[p]["lab"].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── KPI row ───────────────────────────────────────────────────────────────────
def _kpi_cards(df, label_prefix=""):
    """Render 10 KPI cards (5+5) for a single plant's ops dataframe."""
    def sm(c): return df[c].dropna().mean() if c in df.columns else float("nan")
    def ss(c): return df[c].dropna().sum()  if c in df.columns else float("nan")
    import math
    def fmt(v, decimals=1):
        if isinstance(v, float) and math.isnan(v): return "–"
        return f"{v:,.{decimals}f}"

    # ── Raw material ──────────────────────────────────────────────────────────
    dung_avg    = sm("dung_tons")
    potato_avg  = sm("waste_potato_tons")
    rm_lines = []
    if not math.isnan(dung_avg):
        rm_lines.append(f"<span style='font-size:.9rem;font-weight:700;color:#1a56db'>{fmt(dung_avg, 1)}</span>"
                        f"<span style='font-size:.65rem;color:#5a7a9a'> t/d dung</span>")
    if not math.isnan(potato_avg):
        rm_lines.append(f"<span style='font-size:.9rem;font-weight:700;color:#2e7d32'>{fmt(potato_avg, 1)}</span>"
                        f"<span style='font-size:.65rem;color:#5a7a9a'> t/d potato</span>")
    rm_html = "<br>".join(rm_lines) if rm_lines else "–"

    # ── Yield ─────────────────────────────────────────────────────────────────
    total_gas = df["total_generated_gas"].dropna() if "total_generated_gas" in df.columns else pd.Series(dtype=float)
    total_feed = (df["dung_tons"].fillna(0) + df["waste_potato_tons"].fillna(0)
                  ) if "waste_potato_tons" in df.columns else df["dung_tons"].fillna(0) if "dung_tons" in df.columns else pd.Series(0, index=df.index)
    mask = total_feed > 0
    if mask.any() and "total_generated_gas" in df.columns:
        yield_series = df.loc[mask, "total_generated_gas"] / total_feed[mask]
        yield_val = fmt(yield_series.dropna().mean(), 1)
    else:
        yield_val = "–"

    # ── CBG sales (dispenser + cascade) ──────────────────────────────────────
    disp_sum  = ss("cbg_sales_kg")
    casc_sum  = ss("cascade_sales_kg")
    import math as _math
    cbg_disp  = disp_sum if not _math.isnan(disp_sum) else 0
    cbg_casc  = casc_sum if not _math.isnan(casc_sum) else 0
    cbg_total = cbg_disp + cbg_casc
    cbg_lines = []
    if cbg_total > 0:
        cbg_lines.append(f"<span style='font-size:.9rem;font-weight:700;color:#1a56db'>{cbg_total:,.0f}</span>"
                         f"<span style='font-size:.65rem;color:#5a7a9a'> kg total</span>")
        if cbg_casc > 0:
            cbg_lines.append(f"<span style='font-size:.78rem;color:#c84b00'>{cbg_disp:,.0f}</span>"
                             f"<span style='font-size:.63rem;color:#5a7a9a'> disp · </span>"
                             f"<span style='font-size:.78rem;color:#7b1fa2'>{cbg_casc:,.0f}</span>"
                             f"<span style='font-size:.63rem;color:#5a7a9a'> casc</span>")
    cbg_html = "<br>".join(cbg_lines) if cbg_lines else "–"

    # ── Gas flared total ──────────────────────────────────────────────────────
    flare_total = ss("flare_m3")
    flare_val   = fmt(flare_total, 0) if not _math.isnan(flare_total) else "–"

    # ── Electricity consumed ──────────────────────────────────────────────────
    elec_val = fmt(ss("vpsa_kwh_total"), 0)

    # ── Total gas generated (sum over period) ─────────────────────────────────
    total_gen_sum = ss("total_generated_gas")
    total_gen_val = fmt(total_gen_sum, 0) if not _math.isnan(total_gen_sum) else "–"

    # ── Build rows ────────────────────────────────────────────────────────────
    # Row 1:  Dung/Potato | Raw Gas Gen (avg) | Gas Yield | Pure Gas Gen (avg) | Flare (total)
    # Row 2:  Total Gas Gen (sum) | Vehicle+Cascade Sales | Purif Eff | CH₄% | Electricity

    def _card(icon, label, value_html, unit="", opt_note=""):
        opt_bar = f"<div style='font-size:.6rem;color:#2e7d32;margin-top:2px'>{opt_note}</div>" if opt_note else ""
        return f"""
<div class="kpi-card">
  <div class="kpi-icon">{icon}</div>
  <div class="kpi-value" style="font-size:1.25rem">{value_html}</div>
  <div class="kpi-label">{label}{(' · ' + unit) if unit else ''}</div>
  {opt_bar}
</div>"""

    row1 = [
        ("🐄🥔", "Raw Material",
         rm_html, "", ""),
        ("🌿",   "Raw Gas Gen",
         f"{fmt(sm('total_generated_gas'), 0)}", "m³/day", ""),
        ("📈",   "Gas Yield",
         yield_val, "m³/ton", ""),
        ("💨",   "Pure Gas Gen",
         f"{fmt(sm('total_purified_gas'), 0)}", "m³/day", ""),
        ("🔆",   "Gas Flared",
         flare_val, "m³", ""),
    ]
    row2 = [
        ("🌿",   "Total Gas Gen",
         total_gen_val, "m³", ""),
        ("🔥",   "Vehicle + Cascade Sales",
         cbg_html, "", ""),
        ("⚗",   "Purif. Eff.",
         f"{fmt(sm('purif_efficiency'), 1)}", "%",
         "✅ Optimal ≥ 95%" if not _math.isnan(sm("purif_efficiency")) and sm("purif_efficiency") >= 95
         else "⚠ Target: ≥95%" if not _math.isnan(sm("purif_efficiency")) else ""),
        ("✨",   "Avg CH₄ Pure",
         f"{fmt(sm('pure_ch4'), 1)}", "%",
         "✅ Optimal ≥ 90%" if not _math.isnan(sm("pure_ch4")) and sm("pure_ch4") >= 90
         else "⚠ Target: ≥90%" if not _math.isnan(sm("pure_ch4")) else ""),
        ("⚡",   "Electricity",
         elec_val, "KWH", ""),
    ]

    if label_prefix:
        st.markdown(
            f"<div style='font-family:Space Mono,monospace;font-size:0.75rem;"
            f"color:#1a56db;font-weight:700;letter-spacing:.06em;"
            f"margin:6px 0 4px;'>🏭 {label_prefix.upper()}</div>",
            unsafe_allow_html=True)

    r1_cols = st.columns(5)
    for i, (icon, label, val, unit, opt) in enumerate(row1):
        with r1_cols[i]:
            st.markdown(_card(icon, label, val, unit, opt), unsafe_allow_html=True)

    st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)
    r2_cols = st.columns(5)
    for i, (icon, label, val, unit, opt) in enumerate(row2):
        with r2_cols[i]:
            st.markdown(_card(icon, label, val, unit, opt), unsafe_allow_html=True)


def render_kpis(ops, all_data=None, selected=None, date_filter=None, view_mode="individual"):
    if ops.empty:
        st.warning("No operational data for the selected range."); return

    if view_mode == "compare" and all_data and selected and len(selected) >= 2:
        # One KPI block per plant — never averaged together
        for p in selected:
            pdf = all_data.get(p, {}).get("ops", pd.DataFrame())
            if not pdf.empty and date_filter:
                pdf = _flt(pdf, date_filter)
            if pdf.empty:
                st.markdown(
                    f"<div style='font-family:Space Mono,monospace;font-size:0.75rem;"
                    f"color:#c84b00;'>⚠ {p}: no data in selected range</div>",
                    unsafe_allow_html=True)
            else:
                _kpi_cards(pdf, label_prefix=p)
            st.markdown("")

        # ── Yield per day chart (compare mode only) ───────────────────────────
        _render_yield_chart(all_data, selected, date_filter)
    else:
        _kpi_cards(ops)
        st.markdown("")


def _render_yield_chart(all_data, selected, date_filter):
    """Bar+line chart of daily biogas yield (m³/ton feedstock) for each plant."""
    try:
        cmap = _pmap(selected)
        xr   = st.session_state.get("_xrange_cache")
        ma   = st.session_state.get("ma_window", 7)
        fig  = go.Figure()
        has_any = False

        for p in selected:
            pdf = all_data.get(p, {}).get("ops", pd.DataFrame())
            if pdf.empty:
                continue
            if date_filter:
                pdf = _flt(pdf, date_filter)
            if pdf.empty:
                continue

            feed = pdf["dung_tons"].fillna(0)
            if "waste_potato_tons" in pdf.columns:
                feed = feed + pdf["waste_potato_tons"].fillna(0)
            mask = feed > 0
            if not mask.any() or "total_generated_gas" not in pdf.columns:
                continue

            yield_s = (pdf.loc[mask, "total_generated_gas"] / feed[mask]).reindex(pdf.index)
            c = cmap[p]
            c_rgba = _hex_rgba(c, 0.72)

            fig.add_trace(go.Bar(
                x=pdf["date"], y=yield_s,
                name=p,
                marker_color=c_rgba,
                showlegend=True,
            ))
            ma_line = _ma(yield_s, ma)
            fig.add_trace(go.Scatter(
                x=pdf["date"], y=ma_line,
                name=f"{p} {ma}d avg",
                mode="lines",
                line=dict(color=c, width=2),
                showlegend=True,
            ))
            has_any = True

        if not has_any:
            return

        fig.update_layout(
            title="Daily Biogas Yield  [Total Generated Gas ÷ Total Feed]",
            barmode="group",
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            font=dict(color=FONT_COLOR, family="Inter,sans-serif", size=12),
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="center", x=0.5, bgcolor="rgba(255,255,255,.9)",
                        bordercolor="#dde6f4", borderwidth=1, font=dict(size=11, color=FONT_COLOR)),
            hovermode="x unified", height=400, title_x=0,
            title_font=dict(size=13, color="#1e2d45", family="Space Mono,monospace"),
            margin=dict(l=10, r=10, t=44, b=80),
            yaxis_title="m³/ton",
        )
        xkw = dict(showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
                   zeroline=False, showline=True, linecolor="#dde6f4",
                   tickfont=dict(size=11, color=AXIS_COLOR), tickcolor=AXIS_COLOR,
                   title_font=dict(color=AXIS_COLOR), type="date", autorange=(xr is None))
        if xr is not None:
            xkw["range"] = xr
        fig.update_xaxes(**xkw)
        fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
                         zeroline=False, tickfont=dict(size=11, color=AXIS_COLOR),
                         title_font=dict(color=AXIS_COLOR))
        _pc(fig, "kpi_yield_compare")
    except Exception as e:
        st.warning(f"Yield chart error: {e}")


# ── Tab helper: safe two-column layout ───────────────────────────────────────
def _cols2():
    cols = st.columns(2)
    return cols[0], cols[1]

def _cols3():
    cols = st.columns(3)
    return cols[0], cols[1], cols[2]


# ── Tabs ──────────────────────────────────────────────────────────────────────
def tab_gas(ops, ma, xr):
    try:
        sec("📊 GAS PRODUCTION & QUALITY")

        # ── Local MA override ─────────────────────────────────────────────────
        _ga, _gb = _cols2()
        with _ga:
            ma = st.slider("📉 Rolling average window (days)", 1, 14,
                            st.session_state.get("ma_window", 7),
                            key="gas_ma_local",
                            help="Bars = daily values · Line = rolling mean. Click legend to hide/show.")

        c1,c2 = _cols2()
        with c1:
            _pc(bar_line_fig(ops,"date","total_generated_gas",
                         "Raw Biogas Generated  [FM_final − FM_initial]",
                         "m³/day",ma,xr=xr),  "g1")
        with c2:
            _pc(bar_line_fig(ops,"date","total_purified_gas",
                         "Purified Gas Output  [Purifier FM_final − FM_initial]",
                         "m³/day",ma,xr=xr),  "g2")

        c3,c4 = _cols2()
        with c3:
            _pc(bar_line_fig(ops,"date","raw_ch4",
                         "Raw Gas CH₄ (%)  [Inline analyser, raw side]",
                         "%",ma,xr=xr, opt_low=55, opt_high=65),  "g3")
        with c4:
            _pc(bar_line_fig(ops,"date","pure_ch4",
                         "Purified Gas CH₄ (%)  [Inline analyser, pure side]",
                         "%",ma,xr=xr, opt_low=90, opt_high=100),  "g4")

        sec("⚠ H₂S LEVELS (PPM)  ·  Thresholds hidden for readability")
        c5,c6 = _cols2()
        with c5:
            _pc(bar_line_fig(ops,"date","raw_h2s",
                         "Raw Gas H₂S (PPM)  [Inline analyser, raw side]",
                         "PPM",ma,xr=xr),  "g5")
        with c6:
            _pc(bar_line_fig(ops,"date","pure_h2s",
                         "Purified Gas H₂S (PPM)  [Inline analyser, pure side]",
                         "PPM",ma,xr=xr),  "g6")

        sec("⚗ PURIFICATION EFFICIENCY & BIOGAS RECOVERY")
        c9,c10 = _cols2()
        with c9:
            fig = bar_line_fig(ops,"date","purif_efficiency",
                          "Purification Efficiency (%)  [Purified Gas ÷ Raw Gas × 100]",
                          "%",ma,xr=xr, opt_low=95, opt_high=100)
            fig.add_hline(y=95,line_dash="dot",line_color="#2e7d32",
                           annotation_text="Target 95%",annotation_font_color="#2e7d32")
            _pc(fig,"g9")
        with c10:
            _pc(bar_line_fig(ops,"date","bg_recovery",
                         "Biogas Recovery (%)  [Purified ÷ Total Generated × 100]",
                         "%",ma,xr=xr),  "g10")

        sec("📉 GAS BALANCE")
        c7,c8 = _cols2()
        with c7:
            _pc(bar_line_fig(ops,"date","flare_m3",
                         "Flare Gas  [Manual meter reading]",
                         "m³",ma,xr=xr),  "g7")
        with c8:
            _pc(bar_line_fig(ops,"date","gen_inlet_diff",
                         "Gen–Inlet Differential  [Generator FM − Inlet FM]",
                         "m³",ma,xr=xr),  "g8")
    except Exception as e:
        st.error(f"Gas tab error: {e}")


def tab_feed(ops, ma, xr):
    try:
        sec("🐄 FEEDSTOCK & FEEDING")
        c1,c2 = _cols2()
        with c1: _pc(bar_line_fig(ops,"date","dung_tons",
                               "Dung Collected  [Weighbridge / manual]","tons/day",ma,xr=xr),  "f1")
        with c2: _pc(bar_line_fig(ops,"date","total_feed_m3",
                               "Total Feed to Reactor  [Flow Meter Final − Initial]","m³/day",ma,xr=xr),  "f2")
        c3,c4 = _cols2()
        with c3: _pc(bar_line_fig(ops,"date","total_filter_water",
                               "Filter Water  [FM Final − FM Initial]","m³/day",ma,xr=xr),  "f3")
        with c4:
            if "waste_potato_tons" in ops.columns and ops["waste_potato_tons"].notna().any():
                _pc(bar_line_fig(ops,"date","waste_potato_tons","Waste Potato Added","tons/day",ma,xr=xr),  "f4")
            else:
                st.info("No waste-potato data for selected range.")
        sec("📈 SPECIFIC BIOGAS YIELD  [Total Generated Gas (m³) ÷ Total Feedstock (tons)]")
        o2=ops.copy()
        feed_total = o2["dung_tons"].fillna(0) + (o2["waste_potato_tons"].fillna(0) if "waste_potato_tons" in o2.columns else 0)
        o2["yield_m3_per_ton"]=np.where(feed_total>0, o2["total_generated_gas"]/feed_total, np.nan)
        _pc(bar_line_fig(o2,"date","yield_m3_per_ton","Biogas Yield (m³/ton feedstock)","m³/ton",ma,xr=xr),  "f5")
    except Exception as e:
        st.error(f"Feed tab error: {e}")


def tab_purif(ops, ma, xr):
    try:
        sec("⚗ PURIFICATION & CBG SALES")
        c1,c2 = _cols2()
        with c1:
            fig=bar_line_fig(ops,"date","purif_efficiency",
                          "Purification Efficiency (%)  [Purified Gas ÷ Raw Gas × 100]",
                          "%",ma,xr=xr, opt_low=95, opt_high=100)
            fig.add_hline(y=95,line_dash="dot",line_color="#2e7d32",annotation_text="Target 95%",annotation_font_color="#2e7d32")
            _pc(fig,"p1")
        with c2: _pc(bar_line_fig(ops,"date","bg_recovery",
                               "Biogas Recovery (%)  [Purified ÷ Total Generated × 100]",
                               "%",ma,xr=xr),  "p2")
        c3,c4 = _cols2()
        with c3: _pc(bar_line_fig(ops,"date","cbg_sales_kg",
                               "CBG Sales – Dispenser  [Dispenser mass flow meter]","kg/day",ma,xr=xr),  "p3")
        with c4: _pc(bar_line_fig(ops,"date","total_sales_kg",
                               "Total CBG Sales  [Dispenser + Cascade]","kg/day",ma,xr=xr),  "p4")
        # Cascade separately if exists
        if "cascade_sales_kg" in ops.columns and ops["cascade_sales_kg"].notna().any():
            c5,c6 = _cols2()
            with c5: _pc(bar_line_fig(ops,"date","cascade_sales_kg",
                               "Cascade Sales  [Cascade FM]","kg/day",ma,xr=xr),  "p4b")
            with c6: _pc(bar_line_fig(ops,"date","num_vehicles","Vehicles Served / Day","count",1,xr=xr),  "p6")
        else:
            _pc(bar_line_fig(ops,"date","num_vehicles","Vehicles Served / Day","count",1,xr=xr),  "p6")

        sec("📅 MONTHLY CBG SALES")
        monthly=(ops.assign(month=ops["date"].dt.to_period("M").astype(str))
                    .groupby(["month","plant"],as_index=False)[["cbg_sales_kg","cascade_sales_kg"]].sum())
        if not monthly.empty:
            _pc(bar_fig(monthly,"month","cbg_sales_kg","Monthly CBG Sales – Dispenser (kg)",height=400),"p5a")
            if "cascade_sales_kg" in monthly.columns and monthly["cascade_sales_kg"].notna().any():
                _pc(bar_fig(monthly,"month","cascade_sales_kg","Monthly CBG Sales – Cascade (kg)",height=400),"p5b")
        _pc(bar_line_fig(ops,"date","purif_running_hrs","Purification Running Hrs","hrs",1,xr=xr),  "p7")
    except Exception as e:
        st.error(f"Purif tab error: {e}")


def tab_power(ops, ma, xr):
    try:
        sec("⚡ POWER & UTILITY CONSUMPTION")
        c1,c2 = _cols2()
        with c1: _pc(bar_line_fig(ops,"date","vpsa_kwh_total",
                               "VPSA Power  [KWH meter Final − Initial]","KWH/day",ma,xr=xr),  "pw1")
        with c2: _pc(bar_line_fig(ops,"date","bg_mfm_kwh_total",
                               "Biogas MFM Power  [KWH meter Final − Initial]","KWH/day",ma,xr=xr),  "pw2")
        c3,c4 = _cols2()
        with c3: _pc(bar_line_fig(ops,"date","raw_water_m3",
                               "Raw Water  [FM Final − FM Initial]","m³/day",ma,xr=xr),  "pw3")
        with c4: _pc(bar_line_fig(ops,"date","poly_kg",
                               "Poly Consumption  [Manual dosing log]","kg/day",ma,xr=xr),  "pw4")
        c5,c6 = _cols2()
        with c5: _pc(bar_line_fig(ops,"date","dg_hrs","DG Running Hours","hrs",1,xr=xr),  "pw5")
        with c6: _pc(bar_line_fig(ops,"date","dg_diesel_l","DG Diesel Consumed","L/day",ma,xr=xr),  "pw6")
        sec("💡 SPECIFIC ENERGY INTENSITY  [VPSA KWH ÷ Purified Gas (m³)]")
        o2=ops.copy()
        o2["kwh_per_m3"]=np.where(o2["total_purified_gas"]>0,o2["vpsa_kwh_total"]/o2["total_purified_gas"],np.nan)
        _pc(bar_line_fig(o2,"date","kwh_per_m3","VPSA Specific Energy (KWH/m³ purified)","KWH/m³",ma,xr=xr),  "pw7")
    except Exception as e:
        st.error(f"Power tab error: {e}")


def tab_digester(ops, ma, xr):
    try:
        sec("🌡 DIGESTER CONDITIONS")
        c1,c2 = _cols2()
        with c1:
            fig=bar_line_fig(ops,"date","digester_temp",
                          "Digester Temperature (°C)  [9–10 AM manual reading]","°C",ma,xr=xr,
                          opt_low=35, opt_high=40)
            fig.add_hline(y=37,line_dash="dash",line_color="#c84b00",annotation_text="Mesophilic 37°C",annotation_font_color="#c84b00")
            _pc(fig,"d1")
        with c2:
            fig=bar_line_fig(ops,"date","digester_ph",
                          "Digester pH  [9–10 AM manual / probe reading]","pH",ma,xr=xr,
                          opt_low=6.8, opt_high=7.5)
            _pc(fig,"d2")
        sec("💧 DEWATERING  ·  Moisture (%) = (Wet – Dry) ÷ Wet × 100")
        c3,c4 = _cols2()
        with c3:
            _pc(bar_line_fig(ops,"date","screw_moisture","Screw Press Moisture (%)","% ",ma,xr=xr),  "d3a")
        with c4:
            _pc(bar_line_fig(ops,"date","volute_moisture","Volute Press Moisture (%)","% ",ma,xr=xr),  "d3b")
        c_extra,_ = _cols2()
        with c_extra:
            _pc(bar_line_fig(ops,"date","flare_m3","Flare Gas  [Manual meter]","m³",ma,xr=xr),  "d4")
        c5,c6,c7 = _cols3()
        with c5: _pc(bar_line_fig(ops,"date","screw_press_hrs","Screw Press Hrs","hrs",1,height=380,xr=xr),  "d5")
        with c6: _pc(bar_line_fig(ops,"date","vibro_screen_hrs","Vibro Screen Hrs","hrs",1,height=380,xr=xr),  "d6")
        with c7: _pc(bar_line_fig(ops,"date","volute_press_hrs","Volute Press Hrs","hrs",1,height=380,xr=xr),  "d7")
    except Exception as e:
        st.error(f"Digester tab error: {e}")


def tab_lab(all_data, selected, df_flt):
    try:
        sec("🔬 LAB & SLURRY ANALYSIS")
        lab = get_lab(all_data, selected, df_flt)

        if lab.empty:
            st.info("No lab data for this plant/date range. "
                    "Lab sheets may not have been filled in yet."); return

        num_cols_lab = ["pH","EC_mScm","TS_pct","VS_pct","Temp_C","Carbon_pct"]
        has_data = any(lab[c].notna().any() for c in num_cols_lab if c in lab.columns)
        if not has_data:
            st.info("Lab sheet found but all measurement cells are empty."); return

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl_a, ctrl_b, ctrl_c = st.columns([2, 1, 1])
        with ctrl_a:
            pts = sorted(lab["sample_point"].dropna().unique())
            defaults = [s for s in ["RCD (Raw Cattle Dung)","Digester Mid Sampling Point",
                                     "Mixing Tank","Slurry Tank"] if s in pts] or pts[:3]
            chosen = st.multiselect("Sample Points", pts, default=defaults, key="lab_pts")
        with ctrl_b:
            lab_ma = st.slider("Rolling avg (days)", 1, 14, 1, key="lab_ma",
                                help="Set >1 to smooth noisy lab readings (7 samples/day averaged to daily).")
        with ctrl_c:
            rm_lab_outliers = st.checkbox("Remove outliers (IQR×3)", value=True, key="lab_iqr")

        if not chosen:
            st.info("Select at least one sample point."); return

        lab_f = lab[lab["sample_point"].isin(chosen)].copy()

        # Daily-aggregate: average all same-day readings per (date, plant, sample_point)
        # This collapses the 7 daily sub-samples into one point per day per zone
        id_cols = ["date","plant","sample_point"]
        agg_cols = [c for c in num_cols_lab if c in lab_f.columns]
        lab_f["date_only"] = lab_f["date"].dt.normalize()
        lab_day = (lab_f.groupby(["date_only","plant","sample_point"], as_index=False)[agg_cols]
                         .mean())
        lab_day.rename(columns={"date_only":"date"}, inplace=True)

        # IQR outlier removal per (param, sample_point)
        if rm_lab_outliers:
            for col in agg_cols:
                for sp, gdf in lab_day.groupby("sample_point"):
                    mask = lab_day["sample_point"] == sp
                    lab_day.loc[mask, col] = _iqr_clip(lab_day.loc[mask, col], factor=3.0)

        # Apply moving average to the daily-aggregated series
        if lab_ma > 1:
            result_parts = []
            for (plant_val, sp_val), gdf in lab_day.groupby(["plant","sample_point"]):
                gdf = gdf.sort_values("date").copy()
                for col in agg_cols:
                    gdf[col] = _ma(gdf[col], lab_ma)
                result_parts.append(gdf)
            lab_day = pd.concat(result_parts, ignore_index=True) if result_parts else lab_day

        # ── PARAMETER DEFINITIONS with formulas / descriptions ────────────────
        PARAM_META = {
            "pH":         ("pH",         "pH",       "Acidity / alkalinity — optimal digester range 6.8–7.5"),
            "TS_pct":     ("TS (%)",      "%",        "Total Solids  [TS % = (Dry weight ÷ Wet weight) × 100]"),
            "VS_pct":     ("VS (%)",      "%",        "Volatile Solids  [VS % = (Dry loss on ignition ÷ Wet wt) × 100]"),
            "EC_mScm":    ("EC (mS/cm)",  "mS/cm",    "Electrical Conductivity — indicator of salt / nutrient load"),
            "Temp_C":     ("Temp (°C)",   "°C",       "Slurry temperature at sampling point"),
            "Carbon_pct": ("Carbon (%)",  "%",        "Total organic carbon"),
        }

        # ── Per-plant separate charts (in Compare mode) ──────────────────────
        # This prevents legend clutter when multiple plants × multiple sample points
        plant_list = sorted(lab_day["plant"].unique())

        params_to_show = [(k, v) for k, v in PARAM_META.items()
                          if k in lab_day.columns and lab_day[k].dropna().shape[0] > 0]
        rendered = 0

        for param_key, (param_label, param_unit, param_desc) in params_to_show:
            sec(f"📌 {param_label}  ·  {param_desc}")

            # One column per plant — avoids all-on-one-axis mess
            if len(plant_list) > 1:
                plant_cols = st.columns(len(plant_list))
            else:
                plant_cols = [st.container()]

            for pi, plant_val in enumerate(plant_list):
                pdata = lab_day[lab_day["plant"] == plant_val].dropna(subset=[param_key])
                if pdata.empty:
                    with plant_cols[pi]:
                        st.caption(f"No {param_label} data for {plant_val}")
                    continue

                fig = go.Figure()
                sp_list = sorted(pdata["sample_point"].unique())
                sp_colors = {sp: PALETTE[i % len(PALETTE)] for i, sp in enumerate(sp_list)}

                for sp in sp_list:
                    spd = pdata[pdata["sample_point"] == sp].sort_values("date")
                    c = sp_colors[sp]
                    # raw faint dots
                    fig.add_trace(go.Scatter(
                        x=spd["date"], y=spd[param_key],
                        mode="markers", name=sp,
                        marker=dict(color=c, size=5, opacity=0.45),
                        showlegend=True,
                    ))
                    # smooth line if enough points
                    if lab_ma > 1 or len(spd) >= 3:
                        fig.add_trace(go.Scatter(
                            x=spd["date"], y=spd[param_key],
                            mode="lines", name=f"{sp} (line)",
                            line=dict(color=c, width=1.8),
                            opacity=0.9, showlegend=False,
                        ))

                # reference bands
                if param_key == "pH":
                    fig.add_hrect(y0=6.8, y1=7.5, fillcolor="#2e7d32",
                                   opacity=0.06, annotation_text="Optimal 6.8–7.5",
                                   annotation_font_color="#2e7d32")
                elif param_key == "Temp_C":
                    fig.add_hrect(y0=35, y1=40, fillcolor="#c84b00",
                                   opacity=0.06, annotation_text="Mesophilic zone",
                                   annotation_font_color="#c84b00")

                fig.update_layout(
                    title=f"{plant_val} — {param_label} ({param_unit})",
                    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                    font=dict(color=FONT_COLOR, family="Inter,sans-serif"),
                    title_font=dict(size=12, color="#1e2d45", family="Space Mono,monospace"),
                    legend=dict(
                        orientation="v", yanchor="top", y=1.0,
                        xanchor="left", x=1.01,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#dde6f4", borderwidth=1,
                        font=dict(size=10, color=FONT_COLOR),
                    ),
                    height=380,
                    margin=dict(l=10, r=150, t=44, b=40),
                    hovermode="x unified",
                    yaxis_title=param_unit,
                )
                fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID,
                                  tickfont=dict(size=10, color=AXIS_COLOR), type="date")
                fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID,
                                  tickfont=dict(size=10, color=AXIS_COLOR))
                with plant_cols[pi]:
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"lab_{param_key}_{plant_val}")
            rendered += 1

        if rendered == 0:
            st.info("No numeric lab measurements found for the selected sample points.")
            return

        # ── TS vs VS correlation ──────────────────────────────────────────────
        sec("📊 TS vs VS CORRELATION  [VS % is always ≤ TS %; ratio indicates organic fraction]")
        valid = lab_day.dropna(subset=["TS_pct","VS_pct"])
        if not valid.empty:
            # Filter obvious data-entry outliers (VS cannot exceed TS)
            valid = valid[valid["VS_pct"] <= valid["TS_pct"]]
            if not valid.empty:
                _pc(scatter_fig(valid, "TS_pct", "VS_pct",
                                "TS (%) vs VS (%)  —  slope ≈ organic fraction",
                                color="sample_point"), "lab_scatter")
            else:
                st.info("No valid TS/VS pairs found (VS ≤ TS constraint).")
        else:
            st.info("Not enough TS/VS data for correlation plot.")

    except Exception as e:
        import traceback
        st.error(f"Lab tab error: {e}")
        with st.expander("Details"): st.code(traceback.format_exc())


def tab_month_compare(all_data, selected, date_filter):
    """Month vs Month comparison for the same plant — overlay daily trends."""
    try:
        sec("📅 MONTH vs MONTH COMPARISON  ·  Same plant, daily overlay")

        plants_avail = [p for p in selected if p in all_data and not all_data[p]["ops"].empty]
        if not plants_avail:
            st.info("No plant data available."); return

        ctrl_a, ctrl_b = st.columns([1, 3])
        with ctrl_a:
            plant_sel = st.selectbox("Plant", plants_avail, key="mvc_plant")
        ops_full = all_data[plant_sel]["ops"].copy()
        if ops_full.empty:
            st.info(f"No data for {plant_sel}."); return

        # Get all available months for this plant
        all_months = sorted(ops_full["date"].dt.to_period("M").unique(), reverse=True)
        all_month_str = [str(m) for m in all_months]
        if len(all_month_str) < 2:
            st.info("Need at least 2 months of data to compare."); return

        with ctrl_b:
            selected_months = st.multiselect(
                "Select months to compare (max 6)",
                all_month_str,
                default=all_month_str[:2],
                key="mvc_months",
                max_selections=6,
            )
        if len(selected_months) < 2:
            st.info("Select at least 2 months."); return

        ma_mvc = st.slider("Rolling avg (days)", 1, 14, 7, key="mvc_ma")

        # Metrics to compare
        COMPARE_METRICS = [
            ("total_generated_gas",  "Raw Biogas (m³/day)",         "m³",  None, None),
            ("total_purified_gas",   "Purified Gas (m³/day)",        "m³",  None, None),
            ("purif_efficiency",     "Purification Efficiency (%)",  "%",   95,   100),
            ("cbg_sales_kg",         "CBG Sales – Dispenser (kg/d)", "kg",  None, None),
            ("total_sales_kg",       "Total CBG Sales (kg/d)",       "kg",  None, None),
            ("dung_tons",            "Dung Input (tons/day)",        "t",   None, None),
            ("raw_ch4",              "Raw CH₄ (%)",                  "%",   55,   65),
            ("pure_ch4",             "Purified CH₄ (%)",             "%",   90,   100),
            ("digester_temp",        "Digester Temp (°C)",           "°C",  35,   40),
            ("digester_ph",          "Digester pH",                  "pH",  6.8,  7.5),
            ("vpsa_kwh_total",       "Electricity (KWH/day)",        "KWH", None, None),
            ("bg_mfm_kwh_total",     "MFM Reading (KWH/day)",        "KWH", None, None),
        ]

        MVC_PALETTE = ["#1a56db","#c84b00","#2e7d32","#7b1fa2","#00838f","#b71c1c"]

        for col, title, ylab, opt_lo, opt_hi in COMPARE_METRICS:
            if col not in ops_full.columns:
                continue
            frames = []
            for m_str in selected_months:
                period = pd.Period(m_str, "M")
                mdf = ops_full[ops_full["date"].dt.to_period("M") == period].copy()
                mdf = mdf.dropna(subset=[col])
                if mdf.empty:
                    continue
                # day-of-month as x-axis so months overlay
                mdf["dom"] = mdf["date"].dt.day
                mdf["month_label"] = m_str
                frames.append(mdf[["dom", col, "month_label"]])
            if not frames:
                continue

            all_m = pd.concat(frames, ignore_index=True)
            if all_m[col].dropna().empty:
                continue

            fig = go.Figure()
            for i, m_str in enumerate(selected_months):
                mdata = all_m[all_m["month_label"] == m_str].sort_values("dom")
                if mdata.empty:
                    continue
                c = MVC_PALETTE[i % len(MVC_PALETTE)]
                s = mdata[col].astype(float)
                # Bars
                fig.add_trace(go.Bar(
                    x=mdata["dom"], y=s,
                    name=f"{m_str}",
                    marker_color=_hex_rgba(c, 0.45),
                    marker_line_width=0,
                    legendgroup=m_str,
                ))
                # Rolling avg line
                if len(mdata) >= 1:
                    ma_w = max(1, min(ma_mvc, len(mdata)))
                    fig.add_trace(go.Scatter(
                        x=mdata["dom"],
                        y=s.rolling(ma_w, min_periods=1).mean(),
                        name=f"{m_str} avg",
                        mode="lines",
                        line=dict(color=c, width=2.4),
                        legendgroup=m_str,
                    ))

            # Optimum band
            if opt_lo is not None and opt_hi is not None:
                fig.add_hrect(
                    y0=opt_lo, y1=opt_hi,
                    fillcolor="#2e7d32", opacity=0.06, line_width=0,
                    annotation_text=f"Optimal {opt_lo}–{opt_hi}",
                    annotation_font_color="#2e7d32", annotation_font_size=10,
                )

            fig.update_layout(
                title=f"{title}  ·  {plant_sel}",
                yaxis_title=ylab,
                xaxis_title="Day of month",
                barmode="overlay",
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                font=dict(color=FONT_COLOR, family="Inter,sans-serif", size=12),
                title_font=dict(size=13, color="#1e2d45", family="Space Mono,monospace"),
                legend=dict(orientation="h", yanchor="top", y=-0.18,
                            xanchor="center", x=0.5,
                            bgcolor="rgba(255,255,255,.9)",
                            bordercolor="#dde6f4", borderwidth=1,
                            font=dict(size=11, color=FONT_COLOR)),
                height=400, title_x=0, margin=dict(l=10, r=10, t=44, b=80),
                hovermode="x unified",
            )
            fig.update_xaxes(
                tickmode="linear", dtick=1,
                showgrid=True, gridcolor=CHART_GRID,
                tickfont=dict(size=10, color=AXIS_COLOR),
                title_font=dict(color=AXIS_COLOR),
            )
            fig.update_yaxes(
                showgrid=True, gridcolor=CHART_GRID,
                tickfont=dict(size=11, color=AXIS_COLOR),
                title_font=dict(color=AXIS_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"mvc_{col}")

    except Exception as e:
        import traceback
        st.error(f"Month comparison tab error: {e}")
        with st.expander("Details"): st.code(traceback.format_exc())


def tab_compare(all_data, selected, df_flt):
    try:
        sec("📊 CROSS-PLANT COMPARISON")
        if len(selected) < 2:
            st.info("Select **Compare** mode with ≥ 2 plants from the sidebar."); return

        ops = get_ops(all_data, selected, df_flt)
        st.markdown("**Data availability:**")
        dcols = st.columns(len(selected))
        plants_ok = []
        for i,p in enumerate(selected):
            pd_f = _flt(all_data.get(p,{}).get("ops",pd.DataFrame()), df_flt)
            if len(pd_f)>0:
                plants_ok.append(p)
                dcols[i].success(f"**{p}**\n{len(pd_f)} rows\n"
                                  f"{pd_f['date'].min().strftime('%d %b %y')} → "
                                  f"{pd_f['date'].max().strftime('%d %b %y')}")
            else:
                dcols[i].warning(f"**{p}**\nNo data in range")

        if len(plants_ok) < 2:
            st.warning("Need ≥ 2 plants with data. Adjust the date filter."); return

        monthly = (
            ops.assign(month=ops["date"].dt.to_period("M").astype(str))
               .groupby(["month","plant"],as_index=False)
               .agg(total_generated_gas=("total_generated_gas","sum"),
                    total_purified_gas =("total_purified_gas","sum"),
                    cbg_sales_kg       =("cbg_sales_kg","sum"),
                    avg_purif_eff      =("purif_efficiency","mean"),
                    avg_ch4_raw        =("raw_ch4","mean"),
                    avg_ch4_pure       =("pure_ch4","mean"),
                    avg_digester_temp  =("digester_temp","mean"),
                    dung_tons          =("dung_tons","sum"))
        )
        for col,title in [
            ("total_generated_gas","Monthly Raw Biogas (m³)"),
            ("total_purified_gas", "Monthly Purified Gas (m³)"),
            ("cbg_sales_kg",       "Monthly CBG Sales (kg)"),
            ("avg_purif_eff",      "Avg Purification Efficiency (%)  [Purified ÷ Raw × 100]"),
            ("avg_ch4_raw",        "Avg Raw CH₄ (%)  [Inline analyser]"),
            ("avg_digester_temp",  "Avg Digester Temp (°C)  [9–10 AM reading]"),
            ("dung_tons",          "Total Dung Collected (tons)"),
        ]:
            if col in monthly.columns and monthly[col].notna().any():
                _pc(bar_fig(monthly,"month",col,title),"cmp_"+col)

        sec("📈 DAILY OVERLAY")
        xr = _xrange(None); ma = st.session_state.get("ma_window",7)
        # Point 2: raw_ch4 compare opt band max = 65%
        OVERLAY_METRICS = [
            ("total_generated_gas", "Raw Biogas Generated (m³/day)  [FM_final − FM_initial]",  "m³/day", None, None),
            ("total_purified_gas",  "Purified Gas (m³/day)  [Purifier FM_final − FM_initial]", "m³/day", None, None),
            ("purif_efficiency",    "Purification Efficiency (%)  [Purified ÷ Raw × 100]",     "% ",     95,   100),
            ("cbg_sales_kg",        "CBG Sales (kg/day)",                                       "kg",     None, None),
            ("raw_ch4",             "Raw CH₄ (%)  [Inline analyser, raw side]",                "%",      55,   65),
            ("pure_ch4",            "Purified CH₄ (%)  [Inline analyser, pure side]",          "%",      90,   100),
        ]
        for col, title, ylab, opt_lo, opt_hi in OVERLAY_METRICS:
            if col in ops.columns and ops[col].notna().any():
                _pc(bar_line_fig(ops, "date", col, title, ylab, ma,
                                 xr=xr, opt_low=opt_lo, opt_high=opt_hi),
                    "cmpov_" + col)

        # ── Lab sub-section: per-plant per-parameter, clearly separated ──────
        sec("🔬 LAB DATA COMPARISON  ·  Daily averages per plant & sample point")
        lab_all = get_lab(all_data, selected, df_flt)
        if lab_all.empty:
            st.info("No lab data available for comparison."); return

        # Controls
        lab_ctrl_a, lab_ctrl_b = st.columns([3, 1])
        with lab_ctrl_a:
            lab_pts_all = sorted(lab_all["sample_point"].dropna().unique())
            lab_chosen = st.multiselect(
                "Lab sample points to compare",
                lab_pts_all,
                default=[s for s in ["Digester Mid Sampling Point","Mixing Tank"]
                         if s in lab_pts_all] or lab_pts_all[:2],
                key="cmp_lab_pts",
            )
        with lab_ctrl_b:
            cmp_lab_ma = st.slider("Avg window (days)", 1, 14, 7, key="cmp_lab_ma")

        if not lab_chosen:
            st.info("Select sample points above."); return

        lab_f = lab_all[lab_all["sample_point"].isin(lab_chosen)].copy()

        # Daily aggregate (collapses 7 sub-samples per day into one per zone)
        agg_cols_lab = [c for c in ["pH","TS_pct","VS_pct","EC_mScm","Temp_C","Carbon_pct"]
                        if c in lab_f.columns]
        lab_f["date_only"] = lab_f["date"].dt.normalize()
        lab_day = (lab_f.groupby(["date_only","plant","sample_point"], as_index=False)[agg_cols_lab]
                         .mean())
        lab_day.rename(columns={"date_only":"date"}, inplace=True)

        # IQR outlier removal
        for col in agg_cols_lab:
            for sp, gdf in lab_day.groupby("sample_point"):
                mask = lab_day["sample_point"] == sp
                lab_day.loc[mask, col] = _iqr_clip(lab_day.loc[mask, col], factor=3.0)

        # MA smoothing
        if cmp_lab_ma > 1:
            parts = []
            for (p_val, sp_val), gdf in lab_day.groupby(["plant","sample_point"]):
                gdf = gdf.sort_values("date").copy()
                for col in agg_cols_lab:
                    gdf[col] = _ma(gdf[col], cmp_lab_ma)
                parts.append(gdf)
            lab_day = pd.concat(parts, ignore_index=True) if parts else lab_day

        LAB_META = {
            "pH":         ("pH",        "pH"),
            "TS_pct":     ("TS (%)",    "%"),
            "VS_pct":     ("VS (%)",    "%"),
            "EC_mScm":    ("EC mS/cm",  "mS/cm"),
            "Temp_C":     ("Temp °C",   "°C"),
            "Carbon_pct": ("Carbon %",  "%"),
        }

        for param_key, (param_label, param_unit) in LAB_META.items():
            if param_key not in lab_day.columns or lab_day[param_key].dropna().empty:
                continue

            st.markdown(f"**{param_label}** — one panel per sample point")
            panel_cols = st.columns(len(lab_chosen))

            for pi, sp_val in enumerate(lab_chosen):
                spdata = lab_day[lab_day["sample_point"] == sp_val].dropna(subset=[param_key])
                if spdata.empty:
                    with panel_cols[pi]:
                        st.caption(f"No {param_label} for {sp_val[:30]}")
                    continue

                fig = go.Figure()
                for plant_val, gdf in spdata.groupby("plant"):
                    idx = sorted(lab_day["plant"].unique()).index(plant_val)
                    c = PALETTE[idx % len(PALETTE)]
                    gdf = gdf.sort_values("date")
                    fig.add_trace(go.Scatter(
                        x=gdf["date"], y=gdf[param_key],
                        mode="lines+markers", name=plant_val,
                        line=dict(color=c, width=2.2),
                        marker=dict(size=4, opacity=0.6),
                    ))

                sp_short = sp_val[:28] + ("…" if len(sp_val) > 28 else "")
                fig.update_layout(
                    title=f"{sp_short}",
                    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                    font=dict(color=FONT_COLOR, family="Inter,sans-serif", size=11),
                    title_font=dict(size=11, color="#1e2d45", family="Space Mono,monospace"),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.04,
                        xanchor="left", x=0,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#dde6f4", borderwidth=1,
                        font=dict(size=9, color=FONT_COLOR),
                    ),
                    height=320,
                    margin=dict(l=10, r=10, t=60, b=30),
                    hovermode="x unified",
                    yaxis_title=param_unit,
                )
                fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID,
                                  tickfont=dict(size=9, color=AXIS_COLOR), type="date")
                fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID,
                                  tickfont=dict(size=9, color=AXIS_COLOR))
                with panel_cols[pi]:
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"cmp_lab_{param_key}_{sp_val[:20]}")

    except Exception as e:
        import traceback
        st.error(f"Compare tab error: {e}")
        with st.expander("Details"): st.code(traceback.format_exc())


def tab_dung(all_data, selected, df_flt):
    try:
        sec("🚛 DUNG ROUTE QUALITY")
        frames = [_flt(all_data[p]["dung"],df_flt) for p in selected
                  if p in all_data and not all_data[p]["dung"].empty]
        if not frames:
            st.info("No dung route quality data for this selection."); return
        dung = pd.concat(frames, ignore_index=True)
        routes = sorted(dung["route"].dropna().unique())
        chosen = st.multiselect("Routes", routes, default=routes[:6] if len(routes)>6 else routes, key="dung_r")
        if not chosen: return
        dung_f = dung[dung["route"].isin(chosen)]
        rendered = 0
        needles = ["Sand (%)","pH","EC","TS (%)"]
        for i in range(0, len(needles), 2):
            pair = needles[i:i+2]
            cl = st.columns(len(pair))
            for j,needle in enumerate(pair):
                matching = [c for c in dung_f.columns if c.strip().lower()==needle.strip().lower()]
                if not matching: continue
                rc = matching[0]
                if dung_f[rc].dropna().empty: continue
                fig = px.box(dung_f.dropna(subset=[rc]), x="route", y=rc, color="plant",
                             title=f"Dung Route – {needle}")
                fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                    font=dict(color=FONT_COLOR,family="Inter,sans-serif"),
                    title_font=dict(size=13,color="#1e2d45",family="Space Mono,monospace"),
                    legend=dict(orientation="h",yanchor="top",y=-0.18,xanchor="center",x=0.5,
                        bgcolor="rgba(255,255,255,.9)",bordercolor="#dde6f4",borderwidth=1,font=dict(color=FONT_COLOR)),
                    height=460, margin=dict(l=10,r=10,t=44,b=80))
                fig.update_xaxes(showgrid=False, tickfont=dict(color=AXIS_COLOR))
                fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, tickfont=dict(color=AXIS_COLOR))
                with cl[j]: st.plotly_chart(fig, use_container_width=True, key=f"dung_{rc}_{j}")
                rendered += 1
        if rendered == 0: st.info("No matching metric columns found.")
    except Exception as e:
        st.error(f"Dung routes tab error: {e}")


def tab_fert(all_data, selected):
    try:
        sec("🌱 ORGANIC FERTILIZER QUALITY")
        frames = [all_data[p]["fert"] for p in selected
                  if p in all_data and not all_data[p]["fert"].empty]
        if not frames:
            st.info("No fertilizer quality data."); return
        fert = pd.concat(frames, ignore_index=True)
        num_cols = fert.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.dataframe(fert.head(30), use_container_width=True); return
        c1,_ = st.columns([1,3])
        with c1: param = st.selectbox("Parameter", num_cols, key="fert_param")
        mat_col = next((c for c in fert.columns if "material" in str(c).lower()), None)
        fert_plot = fert.dropna(subset=[param])
        if fert_plot.empty:
            st.info(f"No data for {param}."); return
        fig = (px.box(fert_plot, x=mat_col, y=param, color="plant",
                      title=f"{param} by Material Type", points="all")
               if mat_col else
               px.box(fert_plot, y=param, color="plant", title=f"{param}", points="all"))
        fig.update_traces(marker=dict(size=4,opacity=0.6))
        fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            font=dict(color=FONT_COLOR,family="Inter,sans-serif"),
            title_font=dict(size=13,color="#1e2d45",family="Space Mono,monospace"),
            legend=dict(orientation="h",yanchor="top",y=-0.18,xanchor="center",x=0.5,
                bgcolor="rgba(255,255,255,.9)",bordercolor="#dde6f4",borderwidth=1,font=dict(color=FONT_COLOR)),
            height=500, margin=dict(l=10,r=10,t=44,b=80))
        fig.update_xaxes(showgrid=False, tickfont=dict(color=AXIS_COLOR))
        fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, tickfont=dict(color=AXIS_COLOR))
        _pc(fig,"fert_chart")
        with st.expander("📋 Raw fertilizer data"):
            st.dataframe(fert, use_container_width=True)
    except Exception as e:
        st.error(f"Fertilizer tab error: {e}")


def tab_raw(ops, all_data, selected, df_flt):
    try:
        sec("🗄 RAW DATA EXPLORER")
        if ops.empty:
            st.info("No data for the selected range."); return

        plants_avail = [p for p in selected if p in all_data and not all_data[p]["ops"].empty]
        if not plants_avail:
            st.info("No plant data available."); return

        raw_sel = st.multiselect("Plants to include", plants_avail,
                                  default=plants_avail, key="raw_psel") if len(plants_avail)>1 else plants_avail
        if not raw_sel:
            st.info("Select at least one plant."); return

        frames = [_flt(all_data[p]["ops"], df_flt) for p in raw_sel if not all_data[p]["ops"].empty]
        if not frames:
            st.info("No data for selected plants in this date range."); return

        ops_raw = pd.concat(frames, ignore_index=True).sort_values("date", ascending=False).reset_index(drop=True)

        ALWAYS = ["date","plant"]
        data_cols = [c for c in ops_raw.columns if c not in ALWAYS]
        nonempty  = [c for c in data_cols if ops_raw[c].notna().any()]

        col_disp  = {c: COL_LABELS.get(c, c) for c in nonempty}
        disp_to_c = {v: k for k,v in col_disp.items()}

        preferred = ["dung_tons","total_generated_gas","total_purified_gas",
                     "cbg_sales_kg","purif_efficiency","raw_ch4","pure_ch4",
                     "digester_temp","digester_ph","raw_h2s","pure_h2s","vpsa_kwh_total"]
        def_labels = [col_disp[c] for c in preferred if c in col_disp]

        st.markdown(f"**{len(ops_raw):,} rows** · {len(nonempty)} columns with data")

        sel_labels = st.multiselect("Columns to display",
                                     options=list(col_disp.values()),
                                     default=def_labels,
                                     key="raw_col_sel")
        sel_cols = [disp_to_c[l] for l in sel_labels if l in disp_to_c]
        show_cols = ALWAYS + [c for c in sel_cols if c in ops_raw.columns]

        display_df = ops_raw[show_cols].copy()
        display_df.rename(columns={c: COL_LABELS.get(c,c) for c in show_cols if c not in ALWAYS}, inplace=True)
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

        st.dataframe(display_df, use_container_width=True, height=520)

        dl1,dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇ Download CSV",
                ops_raw[show_cols].to_csv(index=False).encode("utf-8"),
                "biogas_data.csv","text/csv",key="dl_csv")
        with dl2:
            buf = io.BytesIO()
            ops_raw[show_cols].to_excel(buf, index=False, engine="openpyxl")
            st.download_button("⬇ Download Excel", buf.getvalue(),
                "biogas_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_xlsx")

        with st.expander("📊 Summary Statistics"):
            num_c = [c for c in sel_cols if pd.api.types.is_float_dtype(ops_raw[c])]
            if num_c:
                stats = ops_raw[num_c].describe().T.round(2)
                stats.index = [COL_LABELS.get(c,c) for c in stats.index]
                st.dataframe(stats, use_container_width=True)
            else:
                st.info("No numeric columns selected.")
    except Exception as e:
        import traceback
        st.error(f"Raw data tab error: {e}")
        with st.expander("Details"): st.code(traceback.format_exc())


# ── Summary statistics (Point 1: min CH₄% for avg calc = 80%) ────────────────
def summary_stats(ops: pd.DataFrame, df_flt) -> pd.DataFrame:
    """
    Compute a summary statistics table for key operational columns.
    For raw_ch4 and pure_ch4 averages, only rows where the value >= 80%
    are included in the mean calculation (Point 1/3 rule).
    """
    if ops is None or ops.empty:
        return pd.DataFrame()

    MIN_CH4_FOR_AVG = 80.0  # Point 1: min CH₄% threshold for average calculation

    STAT_COLS = [
        ("total_generated_gas",  "Raw Gas Generated (m³/day)"),
        ("total_purified_gas",   "Purified Gas (m³/day)"),
        ("purif_efficiency",     "Purification Efficiency (%)"),
        ("raw_ch4",              "Raw CH₄ (%)"),
        ("pure_ch4",             "Pure CH₄ (%)"),
        ("cbg_sales_kg",         "CBG Sales – Dispenser (kg/day)"),
        ("total_sales_kg",       "Total CBG Sales (kg/day)"),
        ("dung_tons",            "Dung Input (tons/day)"),
        ("digester_temp",        "Digester Temp (°C)"),
        ("digester_ph",          "Digester pH"),
        ("flare_m3",             "Flare Gas (m³/day)"),
        ("vpsa_kwh_total",       "Electricity (KWH/day)"),
        ("bg_mfm_kwh_total",     "MFM Reading (KWH/day)"),
    ]

    rows = []
    for col, label in STAT_COLS:
        if col not in ops.columns:
            continue
        s = ops[col].dropna()
        if s.empty:
            continue

        # Apply minimum CH₄% filter for mean calculation on gas quality columns
        if col in ("raw_ch4", "pure_ch4"):
            s_for_avg = s[s >= MIN_CH4_FOR_AVG]
        else:
            s_for_avg = s

        rows.append({
            "Parameter":  label,
            "Count":      int(s.notna().sum()),
            "Mean":       round(s_for_avg.mean(), 2) if not s_for_avg.empty else float("nan"),
            "Median":     round(s.median(), 2),
            "Std Dev":    round(s.std(), 2),
            "Min":        round(s.min(), 2),
            "Max":        round(s.max(), 2),
            "Note":       f"Avg uses values ≥{MIN_CH4_FOR_AVG}% only" if col in ("raw_ch4", "pure_ch4") else "",
        })

    return pd.DataFrame(rows)


def render_stats_table(stats_df: pd.DataFrame):
    """Render the summary statistics table inside an expander."""
    if stats_df is None or stats_df.empty:
        st.info("No statistics available for the selected range.")
        return
    with st.expander("📊 Summary Statistics Table", expanded=False):
        st.caption(
            "⚠ CH₄ averages include only readings ≥ 80% "
            "(Point 1 rule — low / zero readings excluded from mean)."
        )
        st.dataframe(
            stats_df.set_index("Parameter"),
            use_container_width=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_data, selected, date_filter, view_mode = sidebar()

    if not all_data or not selected:
        st.markdown("## ⚡ Biogas Plant Analytics Dashboard")
        st.markdown("""
**Getting started:**
1. Upload one `.xlsx` per plant (Unified Daily Report format) in the sidebar.
2. Name each plant, pick a date filter, set moving-average window.
3. Switch between **Single plant** and **Compare plants** views.

**Data quality:** Outliers are automatically removed via IQR · Physical-zero values in gas/dung columns are treated as missing.
        """)
        return

    # Header
    if view_mode == "compare" and len(selected) >= 2:
        badges = " ".join(f'<span class="compare-badge">⚖ {p}</span>' for p in selected)
        mode_label, mode_color = "COMPARE MODE", "#c84b00"
    else:
        badges = " ".join(f'<span class="plant-badge">🏭 {p}</span>' for p in selected)
        mode_label, mode_color = "INDIVIDUAL VIEW", "#1a56db"

    st.markdown(
        f"<h2 style='font-family:Space Mono,monospace;color:#1e2d45;"
        f"font-size:1.22rem;letter-spacing:.04em;margin-bottom:4px'>"
        f"⚡ BIOGAS ANALYTICS &nbsp;"
        f"<span style='font-size:.68rem;color:{mode_color}'>[{mode_label}]</span>"
        f"&nbsp;{badges}</h2>",
        unsafe_allow_html=True)

    if date_filter:
        if "months" in date_filter:
            st.markdown(
                f'<div class="mode-banner">📅 Showing: <strong>'
                f'{", ".join(date_filter["months"])}</strong></div>',
                unsafe_allow_html=True)
        else:
            s = date_filter["start"].strftime("%d %b %Y")
            e = date_filter["end"].strftime("%d %b %Y")
            st.markdown(
                f'<div class="mode-banner">📅 <strong>{s}</strong> → <strong>{e}</strong></div>',
                unsafe_allow_html=True)

    ma  = st.session_state.get("ma_window", 7)
    ops = get_ops(all_data, selected, date_filter)
    xr  = _xrange(None)
    # cache xr for sub-functions that don't receive it
    st.session_state["_xrange_cache"] = xr

    render_kpis(ops, all_data=all_data, selected=selected,
                  date_filter=date_filter, view_mode=view_mode)

    # Summary stats table (uses ≥80% CH₄ filter per Point 1 rule)
    if not ops.empty:
        render_stats_table(summary_stats(ops, date_filter))

    st.markdown("---")

    if view_mode == "compare" and len(selected) >= 2:
        tabs = st.tabs(["📊 Compare","📅 Month vs Month","📊 Gas","🐄 Feedstock","⚗ Purification",
                         "⚡ Power","🌡 Digester","🔬 Lab","🚛 Dung Routes","🌱 Fertilizer","🗄 Raw Data"])
        with tabs[0]: tab_compare(all_data, selected, date_filter)
        with tabs[1]: tab_month_compare(all_data, selected, date_filter)
        with tabs[2]: tab_gas(ops, ma, xr)
        with tabs[3]: tab_feed(ops, ma, xr)
        with tabs[4]: tab_purif(ops, ma, xr)
        with tabs[5]: tab_power(ops, ma, xr)
        with tabs[6]: tab_digester(ops, ma, xr)
        with tabs[7]: tab_lab(all_data, selected, date_filter)
        with tabs[8]: tab_dung(all_data, selected, date_filter)
        with tabs[9]: tab_fert(all_data, selected)
        with tabs[10]: tab_raw(ops, all_data, selected, date_filter)
    else:
        tabs = st.tabs(["📊 Gas","🐄 Feedstock","⚗ Purification","⚡ Power",
                         "🌡 Digester","📅 Month vs Month","🔬 Lab","🚛 Dung Routes","🌱 Fertilizer","🗄 Raw Data"])
        with tabs[0]: tab_gas(ops, ma, xr)
        with tabs[1]: tab_feed(ops, ma, xr)
        with tabs[2]: tab_purif(ops, ma, xr)
        with tabs[3]: tab_power(ops, ma, xr)
        with tabs[4]: tab_digester(ops, ma, xr)
        with tabs[5]: tab_month_compare(all_data, selected, date_filter)
        with tabs[6]: tab_lab(all_data, selected, date_filter)
        with tabs[7]: tab_dung(all_data, selected, date_filter)
        with tabs[8]: tab_fert(all_data, selected)
        with tabs[9]: tab_raw(ops, all_data, selected, date_filter)


if __name__ == "__main__":
    main()
