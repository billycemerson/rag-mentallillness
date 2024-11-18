import streamlit as st
from streamlit_option_menu import option_menu

from home import home_section
from pred import show_prediction
from rec import show_recommendation
from chat import rag_chatbot

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

# Multipage logic
def main():
    if selected == 'Home':
        home_section()
    elif selected == 'Mental Illness Prediction':
        show_prediction()
    elif selected == 'Recommendation':
        show_recommendation()
    elif selected == 'AI Chatbot':
        rag_chatbot()

if __name__ == "__main__":
    main()