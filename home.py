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
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
               Yuk cek kesehatan mentalmu disini.
            </div>
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                Dapatkan rekoemndasi yang sesuai.
            </div>
            <div style="display: inline-block; width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
                Ada Chatbot AI untuk ngobrol santuy.
        </div>
    </div>
    """, unsafe_allow_html=True)