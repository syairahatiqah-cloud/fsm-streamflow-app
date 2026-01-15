# app.py
# ============================================================
# Streamlit App: FSM Imputation for WL/SF (Upload File Only)
# - Upload Excel/CSV
# - Choose datetime + value column
# - Plot RAW time series (HTML + PNG download)
# - Monthly missing summary (CSV + HTML + PNG download)
# - FSM Imputation (CSV download)
# - Plot RAW vs Imputed (HTML + PNG download)
# - Monthly seasonality original vs imputed (PNG download)
# - Auto y-axis label: Water Level (m) vs Streamflow (m³/s)
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ============================================================
# Helper: infer y-axis label from selected column name
# ============================================================

def infer_yaxis_label(col_name: str) -> str:
    s = str(col_name).lower()

    wl_keys = [
        "wl", "water level", "waterlevel", "stage",
        "river level", "level (m)", "waterlevel (m)"
    ]
    sf_keys = [
        "sf", "streamflow", "stream flow", "discharge",
        "flow", "m3/s", "m³/s"
    ]

    if any(k in s for k in wl_keys):
        return "Water Level (m)"
    if any(k in s for k in sf_keys):
        return "Streamflow (m³/s)"
    return "Value"

# ============================================================
# FSM helper functions (from your Colab script; unchanged logic)
# ============================================================

def find_na_gaps(x: pd.Series):
    is_na = x.isna().to_numpy()
    n = len(is_na)

    gaps = []
    in_gap = False
    start = None

    for i in range(n):
        if is_na[i] and not in_gap:
            in_gap = True
            start = i
        elif (not is_na[i]) and in_gap:
            in_gap = False
            gaps.append((start, i - 1))

    if in_gap:
        gaps.append((start, n - 1))

    return gaps


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))


def fsm_find_best_match(values: np.ndarray,
                        gap_start: int,
                        gap_end: int,
                        m: int,
                        n: int,
                        const_c: float = 0.0,
                        max_candidates: int = None):
    N = len(values)
    gap_len = gap_end - gap_start + 1

    left_start = max(0, gap_start - m)
    left_len = gap_start - left_start

    right_end = min(N - 1, gap_end + n)
    right_len = right_end - gap_end

    if left_len == 0 and right_len == 0:
        return None, 0, gap_len, 0

    I_parts = []
    if left_len > 0:
        I_parts.append(values[left_start:gap_start])
    I_parts.append(np.full(gap_len, const_c, dtype=float))
    if right_len > 0:
        I_parts.append(values[gap_end + 1:right_end + 1])
    I_full = np.concatenate(I_parts).astype(float)
    z = len(I_full)

    valid_mask = ~np.isnan(I_full)
    if not np.any(valid_mask):
        return None, left_len, gap_len, right_len

    I_valid = I_full[valid_mask]

    best_dist = np.inf
    best_S = None

    indices = np.arange(0, N - z + 1) if N >= z else np.array([], dtype=int)

    if max_candidates is not None and len(indices) > max_candidates:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(indices, size=max_candidates, replace=False))

    gap_pos_in_I = np.zeros(z, dtype=bool)
    gap_pos_in_I[left_len:left_len + gap_len] = True

    for start in indices:
        end = start + z - 1

        if not (end < gap_start or start > gap_end):
            continue

        S_window = values[start:end + 1].astype(float)

        if np.any(np.isnan(S_window[valid_mask])):
            continue

        S_for_dist = S_window.copy()
        S_for_dist[gap_pos_in_I] = const_c

        d = euclidean_distance(I_valid, S_for_dist[valid_mask])

        if d < best_dist:
            best_dist = d
            best_S = S_window

    if best_S is None:
        return None, left_len, gap_len, right_len

    return best_S, left_len, gap_len, right_len


def fsm_impute_gap_diff(values: np.ndarray,
                        gap_start: int,
                        gap_end: int,
                        S_window: np.ndarray,
                        left_len: int,
                        gap_len: int,
                        right_len: int):
    x = values.copy()

    if left_len > 0:
        prev_idx = gap_start - 1
        if np.isnan(x[prev_idx]):
            return None

        s_pos = left_len
        current = x[prev_idx]
        for k in range(gap_len):
            if s_pos + k - 1 < 0:
                return None
            diff = S_window[s_pos + k] - S_window[s_pos + k - 1]
            current = current + diff
            x[gap_start + k] = current
        return x

    if right_len > 0:
        next_idx = gap_end + 1
        if np.isnan(x[next_idx]):
            return None

        s_pos_last = left_len + gap_len - 1
        current = x[next_idx]
        for k in range(gap_len):
            offset = gap_len - 1 - k
            if s_pos_last + 1 >= len(S_window):
                return None
            diff = S_window[s_pos_last + 1] - S_window[s_pos_last]
            current = current - diff
            x[gap_start + offset] = current
            s_pos_last -= 1
        return x

    return None


def fsm_impute_gap_scale(values: np.ndarray,
                         gap_start: int,
                         gap_end: int,
                         S_window: np.ndarray,
                         left_len: int,
                         gap_len: int,
                         right_len: int):
    x = values.copy()

    parts = []
    if left_len > 0:
        parts.append(x[gap_start - left_len:gap_start])
    parts.append(np.full(gap_len, np.nan))
    if right_len > 0:
        parts.append(x[gap_end + 1:gap_end + 1 + right_len])
    I_full = np.concatenate(parts).astype(float)

    known_mask = ~np.isnan(I_full)
    if not np.any(known_mask):
        return None

    query_known = I_full[known_mask]
    S_known = S_window[known_mask]

    q_min, q_max = np.nanmin(query_known), np.nanmax(query_known)
    s_min, s_max = np.nanmin(S_known), np.nanmax(S_known)

    if np.isclose(s_max - s_min, 0.0):
        scale = 1.0
    else:
        scale = (q_max - q_min) / (s_max - s_min)

    shift = q_min - s_min * scale
    S_scaled = S_window * scale + shift

    s_gap_start = left_len
    s_gap_end = left_len + gap_len
    x[gap_start:gap_end + 1] = S_scaled[s_gap_start:s_gap_end]
    return x


def impute_series_fsm(series: pd.Series,
                      mode: str = "FSM_scale",
                      m_factor: float = 1.0,
                      const_c: float = 0.0,
                      max_candidates: int = None,
                      verbose: bool = True) -> pd.Series:
    x = series.astype(float).to_numpy()
    original_index = series.index

    gaps = find_na_gaps(series)
    if verbose:
        st.write(f"Found {len(gaps)} gaps.")

    prog = st.progress(0.0)
    total = max(1, len(gaps))

    for gi, (start, end) in enumerate(gaps, 1):
        gap_len = end - start + 1
        m = max(1, int(m_factor * gap_len))
        n = m

        if verbose:
            st.write(f"[Gap {gi}] indices {start}–{end} (len={gap_len}), m=n={m}")

        S_window, left_len, g_len, right_len = fsm_find_best_match(
            x, gap_start=start, gap_end=end, m=m, n=n,
            const_c=const_c, max_candidates=max_candidates
        )

        if S_window is None:
            if verbose:
                st.write("  -> No valid match found, gap left as NaN.")
            prog.progress(gi / total)
            continue

        if mode == "FSM_diff":
            x_new = fsm_impute_gap_diff(x, start, end, S_window, left_len, g_len, right_len)
        elif mode == "FSM_scale":
            x_new = fsm_impute_gap_scale(x, start, end, S_window, left_len, g_len, right_len)
        else:
            raise ValueError("mode must be 'FSM_scale' or 'FSM_diff'")

        if x_new is None:
            if verbose:
                st.write("  -> Imputation failed for this gap, leaving as NaN.")
            prog.progress(gi / total)
            continue

        x = x_new
        prog.progress(gi / total)

    return pd.Series(x, index=original_index, name=series.name)

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="FSM WL/SF App", layout="wide")
st.title("FSM Imputation App (WL / SF) — Upload File Only")

uploaded = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload an Excel (.xlsx/.xls) or CSV file to start.")
    st.stop()

# Read file
try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

# Clean column names
df.columns = df.columns.astype(str).str.strip()

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

cols = list(df.columns)
if len(cols) < 2:
    st.error("Your file must have at least 2 columns (datetime + WL/SF).")
    st.stop()

time_col = st.selectbox("Select datetime column", cols, index=0)
val_col  = st.selectbox("Select WL/SF column", cols, index=1)

# Parse datetime + numeric
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

series = pd.to_numeric(df[val_col], errors="coerce")

y_label = infer_yaxis_label(val_col)
data_type = "Water Level" if y_label.startswith("Water Level") else ("Streamflow" if y_label.startswith("Streamflow") else "Value")

st.write(f"Rows after datetime parsing: {len(df)}")
st.write(f"Missing values in selected series: {int(series.isna().sum())}")
st.write(f"Detected variable type: **{data_type}**  → y-axis label: **{y_label}**")

# ============================================================
# 1) Plot raw time series (download HTML + PNG)
# ============================================================
st.header("1) Raw Time Series Plot")

raw_fig = go.Figure()
raw_fig.add_trace(go.Scatter(x=df[time_col], y=series, mode="lines", name="Raw"))

raw_fig.update_layout(
    title=f"{data_type} Time Series: {val_col}",
    xaxis_title="Date and Time",
    yaxis_title=y_label,
    hovermode="x unified"
)

st.plotly_chart(raw_fig, use_container_width=True)

raw_html = raw_fig.to_html(include_plotlyjs="cdn")
st.download_button(
    "Download raw plot (HTML)",
    data=raw_html.encode("utf-8"),
    file_name=f"{val_col}_raw_time_series.html",
    mime="text/html"
)

# PNG download needs kaleido in requirements.txt
raw_png_bytes = raw_fig.to_image(format="png")
st.download_button(
    "Download raw plot (PNG)",
    data=raw_png_bytes,
    file_name=f"{val_col}_raw_time_series.png",
    mime="image/png"
)

# ============================================================
# 2) Monthly missing summary (download CSV + HTML + PNG)
# ============================================================
st.header("2) Monthly Missing Data Summary")

tmp = pd.DataFrame({time_col: df[time_col], val_col: series})
tmp["Year"] = tmp[time_col].dt.year
tmp["Month"] = tmp[time_col].dt.month

missing_values_per_month = (
    tmp.groupby(["Year", "Month"])[val_col]
    .apply(lambda x: x.isnull().sum())
    .reset_index(name="Missing_Count")
)

total_observations_per_month = (
    tmp.groupby(["Year", "Month"])
    .size()
    .reset_index(name="Total_Observations_Month")
)

monthly_summary = pd.merge(
    missing_values_per_month,
    total_observations_per_month,
    on=["Year", "Month"],
    how="left"
)

monthly_summary["Missing_Percentage"] = (
    monthly_summary["Missing_Count"] / monthly_summary["Total_Observations_Month"]
) * 100

monthly_summary["YearMonth"] = (
    monthly_summary["Year"].astype(str)
    + "-"
    + monthly_summary["Month"].astype(str).str.zfill(2)
)

st.dataframe(monthly_summary.head(50), use_container_width=True)

miss_fig = go.Figure()
miss_fig.add_trace(go.Bar(x=monthly_summary["YearMonth"], y=monthly_summary["Missing_Percentage"], name="Missing %"))
miss_fig.update_layout(
    title=f"Monthly Missing Data Percentage: {val_col}",
    xaxis_title="Year-Month",
    yaxis_title="Missing Percentage (%)",
    xaxis_tickangle=-45,
    hovermode="x unified"
)
st.plotly_chart(miss_fig, use_container_width=True)

monthly_csv = monthly_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download monthly missing summary (CSV)",
    data=monthly_csv,
    file_name=f"{val_col}_monthly_missing_summary.csv",
    mime="text/csv"
)

miss_html = miss_fig.to_html(include_plotlyjs="cdn")
st.download_button(
    "Download monthly missing plot (HTML)",
    data=miss_html.encode("utf-8"),
    file_name=f"{val_col}_monthly_missing_plot.html",
    mime="text/html"
)

miss_png_bytes = miss_fig.to_image(format="png")
st.download_button(
    "Download monthly missing plot (PNG)",
    data=miss_png_bytes,
    file_name=f"{val_col}_monthly_missing_plot.png",
    mime="image/png"
)

# ============================================================
# 3) FSM Imputation (download CSV + plots)
# ============================================================
st.header("3) FSM Imputation")

mode = st.selectbox("FSM mode", ["FSM_scale", "FSM_diff"], index=0)
m_factor = st.number_input("m_factor (context length factor)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
const_c  = st.number_input("const_c (constant inside dummy gap)", value=0.0)
max_candidates = st.number_input("max_candidates (0 = no cap)", min_value=0, value=0, step=1000)
max_candidates = None if max_candidates == 0 else int(max_candidates)

run = st.button("Run FSM Imputation")

if run:
    with st.spinner("Running FSM imputation..."):
        imputed = impute_series_fsm(
            series,
            mode=mode,
            m_factor=float(m_factor),
            const_c=float(const_c),
            max_candidates=max_candidates,
            verbose=True
        )

    out_df = df.copy()
    imputed_col = f"{val_col}_FSM_{mode}"
    out_df[val_col] = series.values
    out_df[imputed_col] = imputed.values

    st.success("FSM imputation completed.")
    st.subheader("Preview imputed data")
    st.dataframe(out_df[[time_col, val_col, imputed_col]].head(50), use_container_width=True)

    out_csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download imputed CSV",
        data=out_csv,
        file_name=f"{val_col}_FSM_imputed.csv",
        mime="text/csv"
    )

    st.subheader("Raw vs FSM Imputed Plot")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=out_df[time_col], y=out_df[val_col], mode="lines", name="Original"))
    fig2.add_trace(go.Scatter(x=out_df[time_col], y=out_df[imputed_col], mode="lines", name="FSM Imputed", line=dict(dash="dot")))
    fig2.update_layout(
        title=f"{data_type}: Original vs FSM Imputed ({mode})",
        xaxis_title="Date and Time",
        yaxis_title=y_label,
        hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig2_html = fig2.to_html(include_plotlyjs="cdn")
    st.download_button(
        "Download raw vs imputed plot (HTML)",
        data=fig2_html.encode("utf-8"),
        file_name=f"{val_col}_raw_vs_fsm_imputed.html",
        mime="text/html"
    )

    fig2_png_bytes = fig2.to_image(format="png")
    st.download_button(
        "Download raw vs imputed plot (PNG)",
        data=fig2_png_bytes,
        file_name=f"{val_col}_raw_vs_fsm_imputed.png",
        mime="image/png"
    )

    st.subheader("Monthly Seasonality (Original vs FSM Imputed)")

    season_df = out_df[[time_col, val_col, imputed_col]].copy()
    season_df["Month"] = pd.to_datetime(season_df[time_col]).dt.month

    avg_orig = season_df.groupby("Month")[val_col].mean().reset_index()
    avg_imp  = season_df.groupby("Month")[imputed_col].mean().reset_index()

    figm, ax = plt.subplots(figsize=(10, 4))
    ax.plot(avg_orig["Month"], avg_orig[val_col], marker="o", label="Original")
    ax.plot(avg_imp["Month"], avg_imp[imputed_col], marker="x", linestyle="--", label="FSM Imputed")
    ax.set_title(f"Monthly Seasonality ({data_type}): Original vs FSM Imputed")
    ax.set_xlabel("Month")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)
    ax.grid(True)
    ax.legend()

    st.pyplot(figm)

    buf = io.BytesIO()
    figm.tight_layout()
    figm.savefig(buf, format="png", dpi=300)
    buf.seek(0)

    st.download_button(
        "Download monthly seasonality plot (PNG)",
        data=buf.getvalue(),
        file_name=f"{val_col}_monthly_seasonality_original_vs_fsm.png",
        mime="image/png"
    )