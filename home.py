import streamlit as st
import plotly.graph_objects as go
import numpy as np

def home_section():
    # Judul
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 20px;">Selamat Datang 🙌😊</h1>
        <h3 style="text-align: center; margin-bottom: 20px;">Flourish: "Grow stronger, every day."</h3>
        """, unsafe_allow_html=True
    )
    st.write("Sistem ini membantu Anda dalam memantau kesehatan mental Anda")

    # Menambahkan fun fact menarik tentang PTM
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <h3 style="font-weight: bold;">Fun Facts 💭</h3>
    </div>
    """, unsafe_allow_html=True)