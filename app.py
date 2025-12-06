import streamlit as st
import numpy as np
import pandas as pd

# page setting
st.set_page_config(
    page_title='Home',
    page_icon=':rocket:',
    layout='wide'
)

st.markdown(
    """
    <div style="text-align:center;">
        <h1>Prognosen Projekt</h1>
        <h1>Stromverbrauch in Baden-Württemberg</h1>
    </div>
    """,
    unsafe_allow_html=True
)