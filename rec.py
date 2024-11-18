import pandas as pd
import streamlit as st
from langchain.schema import HumanMessage

from init import init_rag
from chat import rag_chatbot

# Fungsi untuk membuat prompt rekomendasi
def recommendation_prompt(query, context, recommendation_type):
    if recommendation_type == "Pengobatan":
        prompt = f"""
        Berdasarkan informasi berikut, saya akan memberikan rekomendasi terkait pengobatan.

        **Keluhan Utama**: {query}
        **Riwayat Medis dan Informasi Tambahan**: {context}

        Berdasarkan data di atas, rekomendasi pengobatan yang sesuai adalah...
        """
    elif recommendation_type == "Pola Hidup":
        prompt = f"""
        Berdasarkan informasi berikut, saya akan memberikan rekomendasi terkait pola hidup.

        **Keluhan Utama**: {query}
        **Riwayat Medis dan Informasi Tambahan**: {context}

        Berdasarkan data di atas, rekomendasi pola hidup yang sesuai adalah...
        """
    elif recommendation_type == "Penangan Lanjutan":
        prompt = f"""
        Berdasarkan informasi berikut, saya akan memberikan rekomendasi terkait penanganan lanjutan.

        **Keluhan Utama**: {query}
        **Riwayat Medis dan Informasi Tambahan**: {context}

        Berdasarkan data di atas, rekomendasi penanganan lanjutan yang sesuai adalah...
        """
    return prompt

#Fungsi rekomendasi
def show_recommendation():
    # Memuat LLM dan retrievers
    llm, retriever = init_rag()

    # Header halaman rekomendasi
    st.header("Rekomendasi Kesehatan Mental Anda")

    # Menyimpan percakapan dalam session state jika belum ada
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Menampilkan percakapan sebelumnya
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Fungsi untuk mengirim pesan chat
    def send_message(role, content):
        st.session_state.messages.append({"role": role, "content": content})
        st.chat_message(role).markdown(content)

    # Tombol untuk membersihkan percakapan
    if st.button("Mulai Ulang"):
        st.session_state.clear()
        st.rerun()

    # Menampilkan percakapan awal jika belum ada
    if len(st.session_state.messages) == 0:
        send_message("assistant", "Hallo! Yuk lihat rekomendasinya berdasarkan profil Anda. Silakan pilih rekomendasi yang ingin Anda lihat:")
        send_message("assistant", "1. Pengobatan\n2. Pola Hidup\n3. Penanganan Lanjutan")

    # Memastikan data prediksi ada di session state
    if "prediction" not in st.session_state:
        st.error("Data prediksi tidak ditemukan. Pastikan prediksi sudah dilakukan.")
        return

    pred_data = st.session_state["prediction"]

    # Menampilkan data prediksi dalam bentuk tabel
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([pred_data]), width=700)

    # Menyediakan pilihan rekomendasi
    rekomendasi = st.radio("Pilih jenis rekomendasi", ["Pengobatan", "Pola Hidup", "Penanganan Lanjutan"], key="rekomendasi_option")

    if st.button("Tampilkan Rekomendasi"):
        # Menyimpan pilihan pengguna
        send_message("user", f"Saya memilih {rekomendasi}")

        # Membuat query berdasarkan data prediksi
        query = "\n".join([f"{key}: {value}" for key, value in pred_data.items()])

        # Mendapatkan dokumen yang relevan
        relevant_documents = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_documents])

        # Membuat prompt berdasarkan rekomendasi
        prompt = recommendation_prompt(query=query, context=context, recommendation_type=rekomendasi)
        messages = [HumanMessage(content=prompt)]

        # Mendapatkan hasil dari model LLM
        try:
            answer = llm(messages=messages)
            send_message("assistant", f"Rekomendasi {rekomendasi} Anda:")
            send_message("assistant", answer.content)
        except Exception as e:
            send_message("assistant", "Maaf, terjadi kesalahan saat memproses rekomendasi.")
            st.error(f"Error: {e}")

        # Menanyakan apakah ingin melihat rekomendasi lain
        if st.button("Lihat rekomendasi lainnya"):
            st.session_state.rekomendasi_option = None
            st.session_state.clear()
            st.rerun()