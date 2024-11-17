import glob
import os
import numpy as np
import pandas as pd
import streamlit as st
from time import sleep
from streamlit_option_menu import option_menu
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
from dotenv import load_dotenv

from home import home_section

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title='Flourish',
        options=[
            'Home',
            'Mental Illness Prediction',
            'Recommendation',
            'AI Chatbot'
        ],
        icons=['house', 'activity', 'lightbulb', 'robot'],
        default_index=0
    )

@st.cache_resource
# Fungsi untuk memuat dokumen berbasis penyakit dalam format .txt
def init_rag():
    # Load all PDFs from the specified folder
    txt_folder_path = "./Data/"
    all_txt_paths = glob.glob(os.path.join(txt_folder_path, "*.txt"))
    
    # Load each PDF document and split text
    documents = []
    for txt_path in all_txt_paths:
        loader = TextLoader(txt_path, encoding='utf-8')
        txt_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents.extend(text_splitter.split_documents(txt_docs))
    
    # Set up embeddings and LLM with Google Gemini API
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    
    # Create FAISS vector database from documents
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return llm, retriever

# Fungsi untuk membuat prompt untuk prediksi (sama seperti sebelumnya)
def prediction_prompt(query, context):
    prompt = f"""
    Anda adalah seorang ahli kesehatan mental yang berpengalaman. Berdasarkan informasi berikut, bantu menganalisis kondisi saya untuk memberikan diagnosis awal.

    **Keluhan Utama**:
    {query}

    **Riwayat Medis dan Informasi Tambahan**:
    {context}

    Berdasarkan data di atas, berikan jawaban dengan gaya bahasa santai dan interaktif. Format jawaban adalah sebagai berikut:
    1. Awali dengan analisis singkat, misalnya: "Berdasarkan informasi Anda, sepertinya Anda mengalami..." atau "Sepertinya tidak ada tanda-tanda gangguan mental pada Anda, karena...".
    2. Jika ada gangguan, sebutkan jenis gangguan mental yang mungkin Anda alami (misalnya: gangguan kecemasan, depresi, PTSD, dll.).
    3. Sebutkan tingkat keparahan kondisi Anda (ringan, sedang, atau berat).
    4. Akhiri dengan kata motivasi singkat random yang berkaitan dengan machine learning/AI.
    """
    return prompt

# Fungsi utama untuk menampilkan percakapan chatassistant
def show_prediction():
    # Inisialisasi LLM dan retrievers
    llm, retrievers = init_rag()

    # Header aplikasi
    st.header("Cek Kondisi Kesehatan Mental Anda")

    # Menyimpan percakapan dalam session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Menampilkan percakapan sebelumnya
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Fungsi untuk mengirim pesan chatassistant
    def send_message(role, content):
        st.session_state.messages.append({"role": role, "content": content})
        st.chat_message(role).markdown(content)

    # Menambahkan tombol untuk membersihkan chat dan memulai ulang
    if st.button("Mulai Ulang"):
        st.session_state.clear()  # Menghapus seluruh session state
        st.rerun()  # Memulai ulang aplikasi

    # Mulai percakapan jika belum dimulai
    if len(st.session_state.messages) == 0:
        send_message("assistant", "Halo! Saya di sini untuk membantu Anda memeriksa kesehatan mental Anda. Mari mulai dengan beberapa pertanyaan.")
        send_message("assistant", "Siapa Anda? (Misalnya: Pria, 30 tahun)")

    # Input profil pengguna
    if "profil" not in st.session_state:
        profil = st.text_input("Masukkan profil Anda:")
        if profil:
            st.session_state.profil = profil
            send_message("user", profil)
            send_message("assistant", "Terima kasih! Sekarang, apa keluhan utama Anda? Ceritakan lebih lanjut.")

    # Input keluhan pengguna
    if "keluhan" not in st.session_state and "profil" in st.session_state:
        keluhan = st.text_area("Masukkan keluhan utama Anda:")
        if keluhan:
            st.session_state.keluhan = keluhan
            send_message("user", keluhan)
            send_message("assistant", "Terima kasih! Berapa jam tidur Anda rata-rata setiap hari? (0 - 24 jam)")

    # Input jam tidur
    if "jam_tidur" not in st.session_state and "keluhan" in st.session_state:
        jam_tidur = st.text_input("Jam tidur per hari (jam)")
        if jam_tidur:
            st.session_state.jam_tidur = jam_tidur
            send_message("user", f"{jam_tidur} jam/hari")
            send_message("assistant", "Terima kasih! Seberapa cemas Anda merasa akhir-akhir ini? (0: Tidak Cemas, 10: Sangat Cemas)")

    # Input tingkat kecemasan
    if "tingkat_kecemasan" not in st.session_state and "jam_tidur" in st.session_state:
        tingkat_kecemasan = st.text_input("Tingkat kecemasan")
        if tingkat_kecemasan:
            st.session_state.tingkat_kecemasan = tingkat_kecemasan
            send_message("user", f"{tingkat_kecemasan}/10")
            send_message("assistant", "Terima kasih! Seberapa tinggi tingkat stres Anda dalam seminggu terakhir? (0: Tidak Stres, 10: Sangat Stres)")

    # Input tingkat stres
    if "tingkat_stres" not in st.session_state and "tingkat_kecemasan" in st.session_state:
        tingkat_stres = st.text_input("Tingkat stres")
        if tingkat_stres:
            st.session_state.tingkat_stres = tingkat_stres
            send_message("user", f"{tingkat_stres}/10")
            send_message("assistant", "Terima kasih! Apakah Anda merasa memiliki dukungan sosial yang cukup? (0: Tidak Pernah, 10: Selalu)")

    # Input dukungan sosial
    if "dukungan_sosial" not in st.session_state and "tingkat_stres" in st.session_state:
        dukungan_sosial = st.text_input("Dukungan sosial")
        if dukungan_sosial:
            st.session_state.dukungan_sosial = dukungan_sosial
            send_message("user", f"{dukungan_sosial}/10")
            send_message("assistant", "Apakah Anda pernah mengalami trauma emosional atau fisik? (Misalnya, kehilangan orang terdekat, kecelakaan, dll.)")

    # Input riwayat trauma
    if "riwayat_trauma" not in st.session_state and "dukungan_sosial" in st.session_state:
        riwayat_trauma = st.text_input("Riwayat trauma (Ya/Tidak). Apa traumanya")
        if riwayat_trauma:
            st.session_state.riwayat_trauma = riwayat_trauma
            send_message("user", riwayat_trauma)
            send_message("assistant", "Terima kasih! Terakhir, bagaimana pola hidup Anda sehari-hari? (Misalnya, apakah Anda merokok? Berolahraga rutin?)")

    # Input pola hidup
    if "pola_hidup" not in st.session_state and "riwayat_trauma" in st.session_state:
        pola_hidup = st.text_area("Masukkan pola hidup Anda:")
        if pola_hidup:
            st.session_state.pola_hidup = pola_hidup
            send_message("user", pola_hidup)
            send_message("assistant", "Terima kasih atas informasinya! Klik tombol di bawah ini untuk melakukan prediksi kondisi kesehatan mental Anda.")

    # Tombol prediksi
    if all(key in st.session_state for key in ["profil", "keluhan", "jam_tidur", "tingkat_kecemasan", "tingkat_stres", "dukungan_sosial", "riwayat_trauma", "pola_hidup"]):
        if st.button("Lihat Hasil Prediksi"):
            # Gabungkan data pasien sebagai query untuk prediksi
            query = (
                f"Profil: {st.session_state.profil}. "
                f"Keluhan Utama: {st.session_state.keluhan}. "
                f"Jam tidur: {st.session_state.jam_tidur} jam/hari. "
                f"Tingkat kecemasan: {st.session_state.tingkat_kecemasan}/10. "
                f"Tingkat stres: {st.session_state.tingkat_stres}/10. "
                f"Dukungan sosial: {st.session_state.dukungan_sosial}/10. "
                f"Riwayat trauma: {st.session_state.riwayat_trauma}. "
                f"Pola hidup: {st.session_state.pola_hidup}."
            )

            # Dapatkan dokumen relevan menggunakan retriever
            context = "\n".join([result.page_content for result in retrievers.get_relevant_documents(query)])

            # Buat prompt untuk model LLM
            prompt = prediction_prompt(query=query, context=context)

            # Buat pesan HumanMessage dan dapatkan hasil dari model LLM
            messages = [HumanMessage(content=prompt)]
            answer = llm(messages=messages)

            # Tampilkan hasil prediksi
            send_message("assistant", "Hasil prediksi kesehatan mental Anda: ")
            send_message("assistant", answer.content)

            # Arahkan pengguna ke halaman rekomendasi
            if st.button("Dapatkan Rekomendasi"):
                st.session_state.page = 'Recommendation'
                st.rerun()

# Multipage logic
def main():
    if selected == 'Home':
        home_section()
    elif selected == 'Mental Illness Prediction':
        show_prediction()
    # elif selected == 'Recommendation':
    #     show_recommendation()
    # elif selected == 'Cahatassistant':
    #     chatassistant()

if __name__ == "__main__":
    main()