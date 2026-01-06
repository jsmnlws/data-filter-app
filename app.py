import io
from typing import Optional
import pandas as pd
import streamlit as st

AUTHORIZED_USERS = st.secrets["users"]

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

st.set_page_config(page_title="Universal Data Filter", layout="wide")
st.title("Universal Data Filter")

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

df = load_data(file_bytes, uploaded.name, sheet)
if df.empty:
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

original = df.copy()
original["_row_id"] = range(len(original))

df_parsed = original.copy()
for c in df_parsed.columns:
    if c != "_row_id":
        df_parsed[c] = try_parse_datetime(df_parsed[c])

with st.sidebar:
    threshold = st.slider("Categorical threshold", 5, 200, 50, 5)
    cols = [c for c in df_parsed.columns if c != "_row_id"]
    filter_cols = st.multiselect("Columns to filter", cols, cols[: min(10, len(cols))])

filtered = df_parsed.copy()

for col in filter_cols:
    s = df_parsed[col]
    t = classify_series(s, threshold)

    with st.expander(f"{col} ({t})", expanded=False):
        if t == "numeric":
            s_non = s.dropna()
            if s_non.empty:
                continue
            mn = float(s_non.min())
            mx = float(s_non.max())
            c1, c2 = st.columns(2)
            with c1:
                lo = st.number_input("From", value=mn, format="%.10g", key=f"{col}_from")
            with c2:
                hi = st.number_input("To", value=mx, format="%.10g", key=f"{col}_to")
            keep_na = st.checkbox("Keep blank values", True, key=f"{col}_na")
            lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
            mask = filtered[col].between(lo2, hi2)
            if keep_na:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

        elif t == "datetime":
            s_non = s.dropna()
            if s_non.empty:
                continue
            start, end = st.date_input(
                "Date range",
                (s_non.min().date(), s_non.max().date()),
                key=f"{col}_date"
            )
            keep_na = st.checkbox("Keep blank values", True, key=f"{col}_na")
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end) + pd.Timedelta(days=1)
            mask = (filtered[col] >= start_ts) & (filtered[col] < end_ts)
            if keep_na:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

        elif t == "categorical":
            values = s.dropna().astype(str).unique().tolist()
            selected = st.multiselect("Keep values", values, values, key=f"{col}_cat")
            keep_na = st.checkbox("Keep blank values", True, key=f"{col}_na")
            mask = filtered[col].astype(str).isin(selected)
            if keep_na:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

        else:
            query = st.text_input("Text contains", key=f"{col}_txt")
            keep_na = st.checkbox("Keep blank values", True, key=f"{col}_na")
            if query:
                mask = filtered[col].astype(str).str.contains(query, case=False, na=False)
                if keep_na:
                    mask = mask | filtered[col].isna()
                filtered = filtered[mask]

filtered_ids = set(filtered["_row_id"])
cleaned = df_parsed[~df_parsed["_row_id"].isin(filtered_ids)]

format_choice = st.selectbox("Download format", ["CSV", "Excel"])
filename = st.text_input("Filename", "original_minus_filtered")

if format_choice == "CSV":
    st.download_button(
        "Download",
        cleaned[cols].to_csv(index=False).encode("utf-8"),
        file_name=f"{filename}.csv",
        mime="text/csv"
    )
else:
    st.download_button(
        "Download",
        to_excel_bytes(cleaned[cols], "remaining"),
        file_name=f"{filename}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
