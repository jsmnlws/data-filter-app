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
    with pd.ExcelWriter(out, engine="openpyxl")
