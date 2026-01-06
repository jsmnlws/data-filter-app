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

st.set_page_config(page_title="Row Picker", layout="wide")
st.title("Row Picker")

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

def ensure_working_df(df: pd.DataFrame) -> pd.DataFrame:
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
file_key = f"{uploaded.name}-{len(file_bytes)}"

sheet = None
if uploaded.name.lower().endswith((".xlsx", ".xls", ".xlsm")):
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        if len(xls.sheet_names) > 1:
            sheet = st.selectbox("Select sheet", xls.sheet_names)
    except Exception:
        sheet = None

if "loaded_key" not in st.session_state or st.session_state["loaded_key"] != (file_key, sheet):
    df0 = load_data(file_bytes, uploaded.name, sheet)
    if df0 is None or df0.empty:
        st.warning("No data found.")
        st.stop()
    st.session_state["working_df"] = ensure_working_df(df0)
    st.session_state["loaded_key"] = (file_key, sheet)
    st.session_state["kept_ids"] = set()

df_work = st.session_state["working_df"].copy()
cols = [c for c in df_work.columns if c != "_row_id"]

if "kept_ids" not in st.session_state:
    st.session_state["kept_ids"] = set()

with st.sidebar:
    threshold = st.slider("Categorical threshold", 5, 200, 50, 5)
    preview_rows = st.slider("Top preview rows", 10, 300, 50, 10)
    default_cols = cols[: min(10, len(cols))]
    filter_cols = st.multiselect("Columns to filter", cols, default_cols)
    page_size = st.selectbox("Candidates per page", [50, 100, 200, 500], index=2)
    output_preview_n = st.slider("Output preview rows", 50, 2000, 500, 50)

st.subheader("Current dataset preview")
st.write(f"Rows: {len(df_work):,} | Columns: {len(cols):,}")
st.dataframe(df_work[cols].head(preview_rows), use_container_width=True)

filtered = df_work.copy()

st.subheader("Filters (narrow down candidates)")
for col in filter_cols:
    s = df_work[col]
    t = classify_series(s, threshold)

    with st.expander(f"{col} ({t})", expanded=False):
        enabled = st.checkbox("Enable", value=False, key=f"{col}_enabled")
        if not enabled:
            continue

        include_blank = st.checkbox("Include blank values", value=False, key=f"{col}_blank")

        if t == "numeric":
            s_non = s.dropna()
            if s_non.empty:
                filtered = filtered.iloc[0:0]
                continue
            mn = float(s_non.min())
            mx = float(s_non.max())
            c1, c2 = st.columns(2)
            with c1:
                lo = st.number_input("From", value=mn, format="%.10g", key=f"{col}_from")
            with c2:
                hi = st.number_input("To", value=mx, format="%.10g", key=f"{col}_to")
            lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
            mask = filtered[col].between(lo2, hi2)
            if include_blank:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

        elif t == "datetime":
            s_non = s.dropna()
            if s_non.empty:
                filtered = filtered.iloc[0:0]
                continue
            start, end = st.date_input(
                "Date range",
                (s_non.min().date(), s_non.max().date()),
                key=f"{col}_date"
            )
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end) + pd.Timedelta(days=1)
            mask = (filtered[col] >= start_ts) & (filtered[col] < end_ts)
            if include_blank:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

        elif t == "categorical":
            values = sorted(s.dropna().astype(str).unique().tolist())
            selected = st.multiselect("Keep values", values, default=values, key=f"{col}_cat")
            mask = filtered[col].astype(str).isin(selected)
            if include_blank:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

        else:
            query = st.text_input("Text contains", value="", key=f"{col}_txt")
            if query:
                mask = filtered[col].astype(str).str.contains(query, case=False, na=False)
            else:
                mask = pd.Series(True, index=filtered.index)
            if include_blank:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

st.subheader("Candidates (select rows to keep)")
st.write(f"Candidate rows: {len(filtered):,}")
st.write(f"Selected to keep: {len(st.session_state['kept_ids']):,}")

max_pages = max(1, (len(filtered) + page_size - 1) // page_size)
page = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size

page_df = filtered.iloc[start_idx:end_idx][["_row_id"] + cols].copy()

event = st.dataframe(
    page_df,
    use_container_width=True,
    hide_index=True,
    selection_mode="multi-row",
    on_select="rerun",
    key="candidate_table"
)

selected_positions = event.selection.rows if event and event.selection else []
selected_ids_this_page = page_df.iloc[selected_positions]["_row_id"].tolist() if selected_positions else []

b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("Add selected (this page)"):
        st.session_state["kept_ids"] = set(st.session_state["kept_ids"]).union(selected_ids_this_page)
        st.rerun()
with b2:
    if st.button("Remove selected (this page)"):
        st.session_state["kept_ids"] = set(st.session_state["kept_ids"]).difference(selected_ids_this_page)
        st.rerun()
with b3:
    if st.button("Select all filtered"):
        st.session_state["kept_ids"] = set(filtered["_row_id"].tolist())
        st.rerun()
with b4:
    if st.button("Clear selection"):
        st.session_state["kept_ids"] = set()
        st.rerun()

kept_ids = set(st.session_state["kept_ids"])
kept_df = df_work[df_work["_row_id"].isin(list(kept_ids))][cols].copy()
remaining_df = df_work[~df_work["_row_id"].isin(list(kept_ids))][cols].copy()

st.subheader("Preview of dataset to be downloaded")
st.write(f"Rows in final dataset: {len(remaining_df):,}")
st.dataframe(remaining_df.head(output_preview_n), use_container_width=True)

st.subheader("Downloads")
filename_base = st.text_input("Filename base", "output")
fmt = st.selectbox("Format", ["CSV", "Excel"])

if fmt == "CSV":
    st.download_button(
        "Download final dataset",
        remaining_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{filename_base}.csv",
        mime="text/csv"
    )
else:
    st.download_button(
        "Download final dataset",
        to_excel_bytes(remaining_df, "final"),
        file_name=f"{filename_base}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.subheader("Remove selected rows from the app dataset")
if st.button("Remove kept rows now"):
    df_new = df_work[~df_work["_row_id"].isin(list(kept_ids))].copy()
    df_new = df_new.drop(columns=["_row_id"], errors="ignore")
    df_new = ensure_working_df(df_new)
    st.session_state["working_df"] = df_new
    st.session_state["kept_ids"] = set()
    st.rerun()
