from langchain.schema import HumanMessage
import streamlit as st
from init import init_rag

# Fungsi utama chatbot berbasis RAG
def rag_chatbot():
    st.title("Chatbot Kesehatan Mental Berbasis RAG")
    
    # Inisialisasi model LLM dan retriever
    llm, retriever = init_rag()

    # Tombol untuk menghapus chat
    clear_chat = st.button("🗑️ Clear Chat", help="Hapus semua percakapan")
    if clear_chat:
        st.session_state.messages = []  # Hapus percakapan
        st.rerun()  # Refresh halaman

    # Menyimpan percakapan dalam session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Menampilkan pesan sebelumnya
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    
    # Input dari pengguna
    user_input = st.chat_input("Ketik pertanyaan Anda di sini...")
    if user_input:
        # Simpan input pengguna ke dalam sesi
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)
        
        # Proses pertanyaan dengan retriever untuk mendapatkan dokumen relevan
        relevant_docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Mendapatkan metadata (judul, penulis, jenis) dari dokumen relevan
        sources = []
        for doc in relevant_docs:
            metadata = doc.metadata
            sources.append(f"**Judul**: {metadata['title']}\n**Penulis**: {metadata['author']}\n**Jenis**: {metadata['type']}\n**Sumber**: {metadata['source']}")
        
        # Jika tidak ada konteks yang ditemukan
        if not context:
            st.chat_message("assistant").markdown("Maaf, saya tidak dapat menemukan informasi yang relevan. Coba pertanyaan lain.")
            return
        
        # Membuat prompt untuk LLM
        prompt = f"""
        Anda adalah asisten kesehatan berbasis AI. Berikut adalah pertanyaan dari pengguna:

        **Pertanyaan**: {user_input}

        **Informasi terkait (dari dokumen yang relevan)**:
        {context}

        Jawablah dengan informasi yang singkat dan jelas.
        """
        
        # Menyampaikan prompt ke model LLM untuk mendapatkan jawaban
        messages = [HumanMessage(content=prompt)]
        answer = llm(messages=messages)

        # Format respons dengan sumber dokumen yang lebih lengkap
        sources_text = "\n\n".join(sources)
        formatted_response = f"**Jawaban**:\n{answer.content}\n\n**Sumber Dokumen yang Digunakan**:\n\n{sources_text}"
        
        # Simpan respons asisten ke dalam sesi
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        st.chat_message("assistant").markdown(formatted_response)