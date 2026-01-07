import io
from typing import Optional
import pandas as pd
import streamlit as st

AUTHORIZED_USERS = st.secrets.get("users", {})
if not AUTHORIZED_USERS:
    st.error("Secrets not configured. Add [users] in Streamlit Cloud Secrets.")
    st.stop()

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.rerun()
        else:
            st.error("Invalid credentials")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

st.set_page_config(page_title="Filter and Remove", layout="wide")
st.title("Filter and Remove")

def to_excel_bytes(df: pd.DataFrame, sheet_name: str) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return out.getvalue()

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    name = filename.lower()
    bio = io.BytesIO(file_bytes)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        if sheet_name is None:
            return pd.read_excel(bio, engine="openpyxl")
        return pd.read_excel(bio, engine="openpyxl", sheet_name=sheet_name)
    raise ValueError("Unsupported file type")

def try_parse_datetime(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_object_dtype(series):
        return series
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return series
    parsed = pd.to_datetime(sample, errors="coerce")
    if parsed.notna().mean() >= 0.7:
        return pd.to_datetime(series, errors="coerce")
    return series

def classify_series(series: pd.Series, threshold: int) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.dropna().nunique() <= threshold:
        return "categorical"
    return "text"

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["_row_id"] = range(len(df))
    for c in df.columns:
        if c != "_row_id":
            df[c] = try_parse_datetime(df[c])
    return df

uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls", "xlsm"])
if not uploaded:
    st.stop()

file_bytes = uploaded.getvalue()

sheet = None
if uploaded.name.lower().endswith((".xlsx", ".xls", ".xlsm")):
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        if len(xls.sheet_names) > 1:
            sheet = st.selectbox("Select sheet", xls.sheet_names)
    except Exception:
        sheet = None

raw = load_data(file_bytes, uploaded.name, sheet)
if raw is None or raw.empty:
    st.warning("No data found.")
    st.stop()

df = prepare_df(raw)
cols = [c for c in df.columns if c != "_row_id"]

with st.sidebar:
    threshold = st.slider("Categorical threshold", 5, 200, 50, 5)
    filter_cols = st.multiselect("Columns to filter (rows matching will be removed)", cols, cols[: min(6, len(cols))])
    output_cols = st.multiselect("Columns to keep in output", cols, default=filter_cols if filter_cols else cols)
    preview_n = st.slider("Preview rows", 50, 2000, 500, 50)

if "applied" not in st.session_state:
    st.session_state["applied"] = False

st.subheader("Set filters (rows that match will be removed)")
filters = {}
any_enabled = False

for col in filter_cols:
    s = df[col]
    t = classify_series(s, threshold)

    with st.expander(f"{col} ({t})", expanded=False):
        enabled = st.checkbox("Enable", value=False, key=f"{col}_enabled")
        if not enabled:
            continue
        any_enabled = True

        include_blank = st.checkbox("Include blank values in removal", value=False, key=f"{col}_blank")

        if t == "numeric":
            s_non = s.dropna()
            if s_non.empty:
                st.warning("No numeric values.")
                continue
            mn = float(s_non.min())
            mx = float(s_non.max())
            c1, c2 = st.columns(2)
            with c1:
                lo = st.number_input("From", value=mn, format="%.10g", key=f"{col}_from")
            with c2:
                hi = st.number_input("To", value=mx, format="%.10g", key=f"{col}_to")
            lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
            filters[col] = ("numeric", lo2, hi2, include_blank)

        elif t == "datetime":
            s_non = s.dropna()
            if s_non.empty:
                st.warning("No date values.")
                continue
            start, end = st.date_input(
                "Date range",
                (s_non.min().date(), s_non.max().date()),
                key=f"{col}_date"
            )
            filters[col] = ("datetime", pd.to_datetime(start), pd.to_datetime(end), include_blank)

        elif t == "categorical":
            values = sorted(s.dropna().astype(str).unique().tolist())
            selected = st.multiselect("Values to remove", values, default=[], key=f"{col}_cat")
            filters[col] = ("categorical", selected, include_blank)

        else:
            query = st.text_input("Text contains (rows matching will be removed)", value="", key=f"{col}_txt")
            filters[col] = ("text", query, include_blank)

apply = st.button("Apply filters")

if apply:
    st.session_state["applied"] = True

if st.session_state["applied"] and any_enabled:
    remove_mask = pd.Series(True, index=df.index)

    for col, spec in filters.items():
        kind = spec[0]
        if kind == "numeric":
            _, lo, hi, include_blank = spec
            m = df[col].between(lo, hi)
            if include_blank:
                m = m | df[col].isna()
            remove_mask = remove_mask & m

        elif kind == "datetime":
            _, start_ts, end_ts, include_blank = spec
            end_ts2 = end_ts + pd.Timedelta(days=1)
            m = (df[col] >= start_ts) & (df[col] < end_ts2)
            if include_blank:
                m = m | df[col].isna()
            remove_mask = remove_mask & m

        elif kind == "categorical":
            _, selected, include_blank = spec
            if not selected and not include_blank:
                m = pd.Series(False, index=df.index)
            else:
                m = pd.Series(False, index=df.index)
                if selected:
                    m = m | df[col].astype(str).isin(selected)
                if include_blank:
                    m = m | df[col].isna()
            remove_mask = remove_mask & m

        else:
            _, query, include_blank = spec
            if not query and not include_blank:
                m = pd.Series(False, index=df.index)
            else:
                m = pd.Series(False, index=df.index)
                if query:
                    m = m | df[col].astype(str).str.contains(query, case=False, na=False)
                if include_blank:
                    m = m | df[col].isna()
            remove_mask = remove_mask & m

    final_cols = output_cols if output_cols else cols
    final_df = df.loc[~remove_mask, final_cols].copy()

    removed_count = int(remove_mask.sum())
    st.subheader("Preview of dataset to be downloaded (filtered rows removed)")
    st.write(f"Original rows: {len(df):,}")
    st.write(f"Rows removed by filters: {removed_count:,}")
    st.write(f"Rows remaining: {len(final_df):,}")
    st.dataframe(final_df.head(preview_n), use_container_width=True)

    st.subheader("Download")
    filename_base = st.text_input("Filename base", "filtered_output")
    fmt = st.selectbox("Format", ["CSV", "Excel"])

    if fmt == "CSV":
        st.download_button(
            "Download final dataset",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename_base}.csv",
            mime="text/csv"
        )
    else:
        st.download_button(
            "Download final dataset",
            to_excel_bytes(final_df, "final"),
            file_name=f"{filename_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Enable at least one filter and click Apply filters to preview the output.")
